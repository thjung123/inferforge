from gateway.services.embedding_cache import _make_key


def test_make_key_deterministic():
    k1 = _make_key("bert", {"texts": ["hello"]})
    k2 = _make_key("bert", {"texts": ["hello"]})
    assert k1 == k2


def test_make_key_different_inputs():
    k1 = _make_key("bert", {"texts": ["hello"]})
    k2 = _make_key("bert", {"texts": ["world"]})
    assert k1 != k2


def test_make_key_different_models():
    k1 = _make_key("bert", {"texts": ["hello"]})
    k2 = _make_key("clip", {"texts": ["hello"]})
    assert k1 != k2


def test_make_key_order_independent():
    k1 = _make_key("bert", {"a": 1, "b": 2})
    k2 = _make_key("bert", {"b": 2, "a": 1})
    assert k1 == k2


def test_make_key_prefix():
    key = _make_key("bert", {"texts": ["test"]})
    assert key.startswith("emb_cache:bert:")
