import asyncio
import logging
from pathlib import Path

import httpx
import numpy as np

from builder.config import get_builder_settings
from builder.schemas import JobState
from builder.services.config_generator import (
    generate_config_pbtxt,
    generate_ensemble_config,
    generate_processor_config,
)
from builder.services.dag_validator import validate_ensemble_dag
from builder.services.job_tracker import JobTracker
from builder.services.onnx_exporter import export_onnx
from builder.services.precision_validator import (
    PrecisionThresholds,
    build_sample_inputs,
    compare_outputs,
    evaluate,
    run_onnx_reference,
)
from builder.services.triton_deployer import load_model

logger = logging.getLogger("builder")

_build_semaphore: asyncio.Semaphore | None = None


def _get_build_semaphore() -> asyncio.Semaphore:
    global _build_semaphore
    if _build_semaphore is None:
        _build_semaphore = asyncio.Semaphore(
            get_builder_settings().max_concurrent_builds
        )
    return _build_semaphore


_KSERVE_NUMPY = {
    "INT32": np.int32,
    "INT64": np.int64,
    "FP32": np.float32,
    "FP16": np.float16,
}


async def _query_triton_engine(
    cfg: dict, samples: dict, output_names: list[str], triton_http_url: str
) -> dict:
    """Run the deployed fp16 engine over the sample inputs via Triton (KServe v2)."""
    input_dtypes = {i["name"]: i["datatype"].upper() for i in cfg["inputs"]}
    payload_inputs = []
    for name, arr in samples.items():
        dt = input_dtypes.get(name, "FP32")
        cast = arr.astype(_KSERVE_NUMPY[dt])
        payload_inputs.append(
            {
                "name": name,
                "shape": list(cast.shape),
                "datatype": dt,
                "data": cast.flatten().tolist(),
            }
        )

    url = f"{triton_http_url}/v2/models/{cfg['model_name']}/infer"
    body = {"inputs": payload_inputs, "outputs": [{"name": n} for n in output_names]}
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, json=body)
        resp.raise_for_status()
        data = resp.json()

    out = {}
    for o in data["outputs"]:
        out[o["name"]] = np.array(o["data"], dtype=np.float32).reshape(o["shape"])
    return out


async def _validate_precision(cfg: dict, onnx_path: Path, triton_http_url: str):
    """Compare the deployed fp16 engine against the fp32 ONNX reference."""
    output_names = [o["name"] for o in cfg.get("outputs", [])]
    if not output_names:
        return True, []

    thresholds = PrecisionThresholds.from_cfg(cfg)
    samples = build_sample_inputs(cfg, num_samples=thresholds.num_samples)

    reference = await asyncio.to_thread(
        run_onnx_reference, onnx_path, samples, output_names
    )
    candidate = await _query_triton_engine(cfg, samples, output_names, triton_http_url)

    reports = []
    passed = True
    for name in output_names:
        report = compare_outputs(reference[name], candidate[name], name)
        reports.append(report)
        ok, reasons = evaluate(report, thresholds)
        if not ok:
            passed = False
            logger.error(f"[Precision] {cfg['model_name']}/{name} FAILED: {reasons}")
    return passed, reports


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

    async with _get_build_semaphore():
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


async def _upload_artifacts(job_id, targets, ensemble_cfg, repo, settings) -> None:
    from builder.services.model_uploader import get_minio_client, upload_model_dir

    client = get_minio_client(settings)
    names = [cfg["model_name"] for cfg in targets]
    if ensemble_cfg:
        names += [
            s["model_name"]
            for s in ensemble_cfg["steps"]
            if s.get("backend") == "python"
        ]
        names.append(ensemble_cfg["name"])
    for name in names:
        await asyncio.to_thread(
            upload_model_dir, name, Path(f"{repo}/{name}"), client=client
        )
    logger.info(f"[{job_id}] Uploaded {len(names)} model dirs to object store")


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

    ensemble_cfg = preset.get("ensemble")
    repo_path = Path(repo)

    try:
        for cfg in targets:
            await _build_single_model(job_id, cfg, repo, tracker)

        if ensemble_cfg:
            await tracker.update_status(job_id, JobState.GENERATING_CONFIG)
            logger.info(f"[{job_id}] Validating ensemble DAG ...")
            model_outputs = {
                cfg["model_name"]: {o["name"] for o in cfg.get("outputs", [])}
                for cfg in targets
            }
            for step in ensemble_cfg.get("steps", []):
                if step.get("backend") == "python":
                    model_outputs[step["model_name"]] = {
                        o["name"] for o in step.get("outputs", [])
                    }
            validate_ensemble_dag(ensemble_cfg, model_outputs)

            max_batch = ensemble_cfg.get("max_batch_size", 256)
            for step in ensemble_cfg["steps"]:
                if step.get("backend") == "python":
                    await asyncio.to_thread(
                        generate_processor_config, step, repo_path, max_batch
                    )

            await asyncio.to_thread(generate_ensemble_config, ensemble_cfg, repo_path)
            logger.info(f"[{job_id}] Ensemble configs generated")

        await tracker.update_status(job_id, JobState.DEPLOYING)
        model_label = preset.get("model_name") or preset["model_type"]
        logger.info(f"[{job_id}] Loading models into Triton ...")

        for cfg in targets:
            await load_model(cfg["model_name"])

        if ensemble_cfg:
            for step in ensemble_cfg["steps"]:
                if step.get("backend") == "python":
                    await load_model(step["model_name"])
            await load_model(ensemble_cfg["name"])

        await tracker.update_status(job_id, JobState.VALIDATING_PRECISION)
        for cfg in targets:
            onnx_path = Path(f"{repo}/{cfg['model_name']}/{cfg['model_name']}.onnx")
            try:
                passed, reports = await _validate_precision(
                    cfg, onnx_path, settings.triton_http_url
                )
            except (ValueError, KeyError) as exc:
                raise RuntimeError(
                    f"precision validation error for {cfg['model_name']}: {exc}"
                )
            except Exception as exc:
                raise RuntimeError(
                    f"precision validation could not be completed for "
                    f"{cfg['model_name']}: {exc}"
                )
            for report in reports:
                logger.info(f"[{job_id}] [Precision] {report.as_dict()}")
            if not passed:
                raise RuntimeError(
                    f"fp16 precision validation failed for {cfg['model_name']}"
                )

        if settings.push_to_object_store:
            await _upload_artifacts(job_id, targets, ensemble_cfg, repo, settings)

        await tracker.update_status(job_id, JobState.READY)
        logger.info(f"[{job_id}] Build pipeline complete for {model_label}")

    except Exception as exc:
        logger.error(f"[{job_id}] Build failed: {exc}")
        await tracker.set_failed(job_id, str(exc))
