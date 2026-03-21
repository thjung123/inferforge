import asyncio
import logging
import tempfile
from pathlib import Path

import yaml

from builder.config import get_builder_settings
from builder.schemas import JobState
from builder.services.job_tracker import JobTracker
from model_builder.scripts.convert_to_onnx import convert_to_onnx
from model_builder.scripts.generate_triton_config import generate_triton_config

logger = logging.getLogger("builder")


def _build_trtexec_command(cfg: dict, onnx_path: str) -> list[str]:
    engine_dir = Path(cfg["paths"]["engine_model_dir"]) / "1"
    engine_dir.mkdir(parents=True, exist_ok=True)
    engine_path = engine_dir / "model.plan"

    precision = cfg.get("precision", {}).get("default", "fp16")

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
    ]

    if precision == "fp16":
        cmd.append("--fp16")
    elif precision == "int8":
        cmd.append("--int8")

    dynamic = cfg.get("dynamic_shapes", {})
    if dynamic.get("enabled"):
        min_shapes = []
        opt_shapes = []
        max_shapes = []

        for inp in cfg["inputs"]:
            name = inp["name"]
            if name in dynamic:
                profile = dynamic[name]
                min_shapes.append(f"{name}:" + "x".join(str(d) for d in profile["min"]))
                opt_shapes.append(f"{name}:" + "x".join(str(d) for d in profile["opt"]))
                max_shapes.append(f"{name}:" + "x".join(str(d) for d in profile["max"]))
            else:
                shape = inp["shape"]
                static = "x".join(str(abs(d)) for d in shape)
                min_shapes.append(f"{name}:{static}")
                opt_shapes.append(f"{name}:{static}")
                max_shapes.append(f"{name}:{static}")

        if min_shapes:
            cmd.append(f"--minShapes={','.join(min_shapes)}")
            cmd.append(f"--optShapes={','.join(opt_shapes)}")
            cmd.append(f"--maxShapes={','.join(max_shapes)}")

    cmd.append("--verbose")
    return cmd


async def run_build_pipeline(
    job_id: str,
    preset: dict,
    tracker: JobTracker,
) -> None:
    settings = get_builder_settings()
    cfg = dict(preset)
    model_name = cfg["model_name"]

    repo = settings.model_repository
    cfg.setdefault("paths", {})
    cfg["paths"]["onnx_model"] = f"{repo}/{model_name}/{model_name}.onnx"
    cfg["paths"]["engine_model_dir"] = f"{repo}/{model_name}"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(cfg, tmp)
        tmp_path = tmp.name

    try:
        await tracker.update_status(job_id, JobState.BUILDING_ONNX)
        logger.info(f"[{job_id}] Converting {model_name} to ONNX ...")
        onnx_path = await asyncio.to_thread(convert_to_onnx, tmp_path)
        logger.info(f"[{job_id}] ONNX export done → {onnx_path}")

        await tracker.update_status(job_id, JobState.BUILDING_TRT)
        logger.info(f"[{job_id}] Building TensorRT engine ...")
        trt_cmd = _build_trtexec_command(cfg, str(onnx_path))
        logger.info(f"[{job_id}] trtexec command: {' '.join(trt_cmd)}")

        proc = await asyncio.create_subprocess_exec(
            *trt_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(
                f"trtexec failed (rc={proc.returncode}): {stdout.decode()[-2000:]}"
            )
        logger.info(f"[{job_id}] TRT engine built successfully")

        await tracker.update_status(job_id, JobState.GENERATING_CONFIG)
        logger.info(f"[{job_id}] Generating Triton config ...")
        await asyncio.to_thread(generate_triton_config, tmp_path)
        logger.info(f"[{job_id}] config.pbtxt generated")

        await tracker.update_status(job_id, JobState.READY)
        logger.info(f"[{job_id}] Build pipeline complete for {model_name}")

    except Exception as exc:
        logger.error(f"[{job_id}] Build failed: {exc}")
        await tracker.set_failed(job_id, str(exc))
    finally:
        Path(tmp_path).unlink(missing_ok=True)
