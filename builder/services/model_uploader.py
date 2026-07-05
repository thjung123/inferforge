"""Push built model artifacts to the object-storage (MinIO/S3) model repository."""

import logging
from pathlib import Path

from minio import Minio

from builder.config import BuilderSettings, get_builder_settings

logger = logging.getLogger("builder")


def get_minio_client(settings: BuilderSettings | None = None) -> Minio:
    settings = settings or get_builder_settings()
    return Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )


def upload_model_dir(
    model_name: str,
    local_dir: Path,
    *,
    client: Minio | None = None,
    bucket: str | None = None,
    settings: BuilderSettings | None = None,
) -> str:
    """Upload local_dir to <bucket>/<model_name>/ and return the s3:// URI."""
    settings = settings or get_builder_settings()
    client = client or get_minio_client(settings)
    bucket = bucket or settings.model_bucket

    if not local_dir.is_dir():
        raise FileNotFoundError(f"model dir not found: {local_dir}")

    if not client.bucket_exists(bucket):
        try:
            client.make_bucket(bucket)
        except Exception:
            pass

    uploaded = 0
    for path in sorted(local_dir.rglob("*")):
        if path.is_file() and path.suffix != ".onnx":
            rel = path.relative_to(local_dir).as_posix()
            client.fput_object(bucket, f"{model_name}/{rel}", str(path))
            uploaded += 1

    uri = f"s3://{bucket}/{model_name}/"
    logger.info(f"[Upload] {model_name} → {uri} ({uploaded} objects)")
    return uri
