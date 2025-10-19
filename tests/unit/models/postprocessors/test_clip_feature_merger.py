import numpy as np
from model_builder.postprocessors.clip_feature_merger import ClipFeatureMerger


def test_clip_feature_merger_merge():
    merger = ClipFeatureMerger()
    img = np.random.rand(2, 512).astype(np.float32)
    txt = np.random.rand(3, 512).astype(np.float32)

    sim = merger.merge(img, txt)
    assert sim.shape == (2, 3)

    row_sums = np.sum(sim, axis=1)
    np.testing.assert_allclose(row_sums, np.ones_like(row_sums), rtol=1e-5, atol=1e-5)

    assert np.all(sim >= 0.0)
    assert np.all(sim <= 1.0)
