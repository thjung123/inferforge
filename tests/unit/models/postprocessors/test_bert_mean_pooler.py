import numpy as np

from builder.processors.bert.postprocessor import BertMeanPooler


def test_mean_pool_ignores_padding():
    pooler = BertMeanPooler()
    # batch=1, seq=3, hidden=2; 3rd token is padding (mask=0) with a huge value
    last_hidden_state = np.array(
        [[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]], dtype=np.float32
    )
    attention_mask = np.array([[1, 1, 0]], dtype=np.int32)

    emb = pooler.run(last_hidden_state, attention_mask)

    # Only the first two tokens count → mean of (1,2) and (3,4)
    np.testing.assert_allclose(emb, [[2.0, 3.0]], rtol=1e-6)
    assert emb.shape == (1, 2)
    assert emb.dtype == np.float32


def test_mean_pool_all_valid_tokens():
    pooler = BertMeanPooler()
    last_hidden_state = np.array([[[2.0, 0.0], [4.0, 8.0]]], dtype=np.float32)
    attention_mask = np.array([[1, 1]], dtype=np.int32)

    emb = pooler.run(last_hidden_state, attention_mask)

    np.testing.assert_allclose(emb, [[3.0, 4.0]], rtol=1e-6)


def test_mean_pool_batch_shape():
    pooler = BertMeanPooler()
    last_hidden_state = np.random.rand(4, 128, 768).astype(np.float32)
    attention_mask = np.ones((4, 128), dtype=np.int32)

    emb = pooler.run(last_hidden_state, attention_mask)

    assert emb.shape == (4, 768)


def test_mean_pool_all_padding_does_not_divide_by_zero():
    pooler = BertMeanPooler()
    last_hidden_state = np.ones((1, 3, 2), dtype=np.float32)
    attention_mask = np.zeros((1, 3), dtype=np.int32)

    emb = pooler.run(last_hidden_state, attention_mask)

    assert np.all(np.isfinite(emb))
