import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from time import sleep
from datetime import datetime
import requests


TRITON_URL = "localhost:8001"
HTTP_URL = TRITON_URL.replace(":8001", ":8000")
MODEL_NAME = "clip_image_encoder"
CONFIG_MATRIX_PATH = "performance/config_matrix.json"
CONFIG_PATHS = [
    "model_repository/clip_image_encoder/config.pbtxt",
]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = Path(f"performance/results/{timestamp}")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def backup_config(path: Path):
    backup_path = path.with_suffix(".bak")
    if not backup_path.exists():
        shutil.copy(path, backup_path)
        print(f"[BACKUP] {path} → {backup_path}")


def restore_config(path: Path):
    backup_path = path.with_suffix(".bak")
    if not backup_path.exists():
        return

    shutil.copy(backup_path, path)
    print(f"[RESTORE] {path} restored from backup.")

    text = path.read_text()
    text = clean_backend_field(text)

    text = re.sub(
        r'^\s*default_model_filename:\s*".*"\s*$',
        "",
        text,
        flags=re.MULTILINE,
    )

    if "platform:" not in text and "backend:" not in text:
        text = 'platform: "tensorrt_plan"\n' + text

    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"

    path.write_text(text)
    print(f"[CLEAN] backend/default_model_filename cleaned in {path.name}")


def clean_backend_field(text: str) -> str:
    return re.sub(r'^\s*backend:\s*".*"\s*$', "", text, flags=re.MULTILINE)


def insert_dynamic_batching(config_path: Path, preferred, delay):
    text = config_path.read_text()
    text = clean_backend_field(text)

    text = re.sub(
        r"dynamic_batching\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
        "",
        text,
        flags=re.MULTILINE,
    )

    block = (
        f"dynamic_batching {{\n"
        f"  preferred_batch_size: [{', '.join(map(str, preferred))}]\n"
        f"  max_queue_delay_microseconds: {delay}\n"
        f"}}\n"
    )

    if "max_batch_size" in text:
        updated = re.sub(
            r"(max_batch_size\s*:\s*\d+)",
            r"\1\n" + block,
            text,
            count=1,
        )
    else:
        updated = block + "\n" + text

    updated = re.sub(r"\n{3,}", "\n\n", updated).strip() + "\n"
    config_path.write_text(updated)
    print(f"[UPDATE] dynamic_batching inserted into {config_path.name}")


def wait_for_triton_ready(timeout=120):
    url = f"http://{HTTP_URL}/v2/health/ready"
    for i in range(timeout):
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                print(f"[READY] Triton is ready after {i}s.")
                return
        except Exception:
            pass
        sleep(2)
    raise TimeoutError("Triton did not become ready in time.")


def reload_model(model_name=MODEL_NAME, max_retries=3):
    url_load = f"http://{HTTP_URL}/v2/repository/models/{model_name}/load"
    url_unload = f"http://{HTTP_URL}/v2/repository/models/{model_name}/unload"

    print(f"[SAFE RELOAD] Unloading {model_name} ...")
    try:
        requests.post(url_unload, timeout=5)
    except Exception as e:
        print(f"[WARN] Unload failed: {e}")

    for i in range(max_retries):
        print(f"[SAFE RELOAD] Attempt {i+1}")
        time.sleep(2)
        try:
            r = requests.post(url_load, timeout=10)
            if r.status_code == 200:
                print(f"[OK] {model_name} reloaded successfully")
                return
            else:
                print(f"[ERROR] Reload failed ({r.status_code}): {r.text}")
        except Exception as e:
            print(f"[WARN] Reload exception: {e}")
    print(f"[FAIL] {model_name} reload failed after {max_retries} attempts")


def run_perf_analyzer(batch, concurrency, tag):
    out_dir = Path("/workspace/perf_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_file = out_dir / f"{MODEL_NAME}_b{batch}_c{concurrency}_{tag}.csv"
    log_file = RESULTS_DIR / f"{MODEL_NAME}_b{batch}_c{concurrency}_{tag}.log"

    cmd = (
        f"perf_analyzer "
        f"-m {MODEL_NAME} "
        f"-u {HTTP_URL} "
        f"-b {batch} "
        f"--shape text:3,244,244 "
        f"--concurrency-range {concurrency} "
        f"--percentile 95 "
        f"--input-data zero "
        f"--collect-metrics "
        f"--verbose-csv "
        f"-f {csv_file}"
    )

    print(f"[RUN] {cmd}")
    with open(log_file, "w") as f:
        subprocess.run(
            ["bash", "-lc", cmd], stdout=f, stderr=subprocess.STDOUT, check=False
        )

    if csv_file.exists():
        print(f"[OK] CSV saved → {csv_file}")
    else:
        print("[WARN] No CSV generated (check metrics endpoint)")

    print(f"[SAVED] Log → {log_file}")
    sleep(max(2, batch * concurrency / 300))


def main():
    with open(CONFIG_MATRIX_PATH) as f:
        all_cfg = json.load(f)

    for exp_name, cfg in all_cfg.items():
        print(f"\n========== [EXPERIMENT: {exp_name}] ==========")

        for enable in cfg["enable_dynamic_batching"]:
            for preferred in cfg["preferred_batch_sizes"]:
                for delay in cfg["max_queue_delay_microseconds"]:
                    tag = (
                        f"{exp_name}_dynamic_{delay}"
                        if enable
                        else f"{exp_name}_no_dynamic"
                    )

                    # Update config and reload
                    for path in CONFIG_PATHS:
                        p = Path(path)
                        backup_config(p)
                        if enable:
                            insert_dynamic_batching(p, preferred, delay)
                        else:
                            restore_config(p)

                    reload_model()
                    wait_for_triton_ready()

                    for batch in cfg["batch_sizes"]:
                        for conc in cfg["concurrency"]:
                            print(
                                f"[TEST] batch={batch}, concurrency={conc}, tag={tag}"
                            )
                            run_perf_analyzer(batch=batch, concurrency=conc, tag=tag)

                    # Restore after each block
                    for path in CONFIG_PATHS:
                        restore_config(Path(path))
                    reload_model()
                    wait_for_triton_ready()

    print("[DONE] All matrix experiments completed.")


if __name__ == "__main__":
    main()
