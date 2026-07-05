import numpy as np
from builder.processors.clip.feature_merger import ClipFeatureMerger


def test_clip_feature_merger_cosine_range():
    merger = ClipFeatureMerger()
    img = np.random.rand(2, 512).astype(np.float32)
    txt = np.random.rand(3, 512).astype(np.float32)

    sim = merger.merge(img, txt)
    assert sim.shape == (2, 3)
    # cosine similarity lives in [-1, 1]
    assert np.all(sim >= -1.0 - 1e-5)
    assert np.all(sim <= 1.0 + 1e-5)


def test_clip_feature_merger_identical_vectors_score_one():
    merger = ClipFeatureMerger()
    v = np.array([[1.0, 2.0, 2.0, 4.0]], dtype=np.float32)
    sim = merger.merge(v, v)
    np.testing.assert_allclose(sim, [[1.0]], rtol=1e-5, atol=1e-5)


def test_clip_feature_merger_batch_independent():
    """A pair's score must not change when other texts are added to the batch
    (the old softmax version coupled scores across the batch)."""
    merger = ClipFeatureMerger()
    img = np.random.rand(1, 16).astype(np.float32)
    txt = np.random.rand(3, 16).astype(np.float32)

    full = merger.merge(img, txt)
    single = merger.merge(img, txt[:1])
    np.testing.assert_allclose(full[0, 0], single[0, 0], rtol=1e-5, atol=1e-5)
