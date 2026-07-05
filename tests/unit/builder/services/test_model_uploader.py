from pathlib import Path

from builder.services.model_uploader import upload_model_dir


class _FakeMinio:
    def __init__(self, bucket_exists=False):
        self._exists = bucket_exists
        self.made = []
        self.put = []

    def bucket_exists(self, bucket):
        return self._exists

    def make_bucket(self, bucket):
        self.made.append(bucket)

    def fput_object(self, bucket, obj, path):
        self.put.append((bucket, obj, path))


def _build_model_dir(root: Path) -> Path:
    (root / "1").mkdir(parents=True)
    (root / "config.pbtxt").write_text('name: "bert_encoder"')
    (root / "1" / "model.plan").write_bytes(b"engine-bytes")
    return root


def test_upload_uploads_all_files_and_returns_uri(tmp_path):
    model_dir = _build_model_dir(tmp_path / "bert_encoder")
    fake = _FakeMinio(bucket_exists=False)

    uri = upload_model_dir(
        "bert_encoder", model_dir, client=fake, bucket="model-repository"
    )

    assert uri == "s3://model-repository/bert_encoder/"
    # bucket created because it did not exist
    assert fake.made == ["model-repository"]
    objs = sorted(o for _, o, _ in fake.put)
    assert objs == ["bert_encoder/1/model.plan", "bert_encoder/config.pbtxt"]


def test_upload_skips_bucket_creation_when_exists(tmp_path):
    model_dir = _build_model_dir(tmp_path / "m")
    fake = _FakeMinio(bucket_exists=True)

    upload_model_dir("m", model_dir, client=fake, bucket="b")

    assert fake.made == []  # not created
    assert len(fake.put) == 2
