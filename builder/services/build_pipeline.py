import asyncio
import logging
from pathlib import Path

from builder.config import get_builder_settings
from builder.schemas import JobState
from builder.services.config_generator import generate_config_pbtxt
from builder.services.job_tracker import JobTracker
from builder.services.onnx_exporter import export_onnx
from builder.services.triton_deployer import load_model

logger = logging.getLogger("builder")


def _build_trtexec_command(cfg: dict, onnx_path: Path) -> list[str]:
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


async def _build_single_model(
    job_id: str,
    cfg: dict,
    repo: str,
    tracker: JobTracker,
) -> None:
    model_name = cfg["model_name"]

    cfg.setdefault("paths", {})
    onnx_path = Path(f"{repo}/{model_name}/{model_name}.onnx")
    engine_dir = Path(f"{repo}/{model_name}")
    cfg["paths"]["engine_model_dir"] = str(engine_dir)

    await tracker.update_status(job_id, JobState.BUILDING_ONNX)
    logger.info(f"[{job_id}] Converting {model_name} to ONNX ...")
    onnx_path = await asyncio.to_thread(export_onnx, cfg, onnx_path)
    logger.info(f"[{job_id}] ONNX export done → {onnx_path}")

    await tracker.update_status(job_id, JobState.BUILDING_TRT)
    logger.info(f"[{job_id}] Building TensorRT engine for {model_name} ...")
    trt_cmd = _build_trtexec_command(cfg, onnx_path)
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
    logger.info(f"[{job_id}] TRT engine built for {model_name}")

    await tracker.update_status(job_id, JobState.GENERATING_CONFIG)
    logger.info(f"[{job_id}] Generating Triton config for {model_name} ...")
    await asyncio.to_thread(generate_config_pbtxt, cfg, engine_dir)
    logger.info(f"[{job_id}] config.pbtxt generated for {model_name}")


async def run_build_pipeline(
    job_id: str,
    preset: dict,
    tracker: JobTracker,
) -> None:
    settings = get_builder_settings()
    repo = settings.model_repository

    submodels = preset.get("submodels")
    if submodels:
        targets = []
        for sub in submodels:
            sub_cfg = dict(sub)
            sub_cfg["source"] = preset["source"]
            sub_cfg["model_type"] = preset["model_type"]
            targets.append(sub_cfg)
    else:
        targets = [dict(preset)]

    try:
        for cfg in targets:
            await _build_single_model(job_id, cfg, repo, tracker)

        await tracker.update_status(job_id, JobState.DEPLOYING)
        model_label = preset.get("model_name") or preset["model_type"]
        logger.info(f"[{job_id}] Loading models into Triton ...")

        for cfg in targets:
            await load_model(cfg["model_name"])

        await tracker.update_status(job_id, JobState.READY)
        logger.info(f"[{job_id}] Build pipeline complete for {model_label}")

    except Exception as exc:
        logger.error(f"[{job_id}] Build failed: {exc}")
        await tracker.set_failed(job_id, str(exc))
