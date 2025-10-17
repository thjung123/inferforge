import numpy as np
from model_builder.preprocessors.bert_preprocessor import BertPreprocessor


def test_bert_preprocessor_basic():
    preprocessor = BertPreprocessor()
    texts = [
        "NVIDIA builds powerful AI infrastructure.",
        "Transformers make NLP much easier.",
    ]
    result = preprocessor.run(texts)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"input_ids", "attention_mask", "token_type_ids"}

    for key in result:
        assert isinstance(result[key], np.ndarray)
        assert result[key].ndim == 2
        assert result[key].shape[0] == len(texts)
        assert result[key].dtype == np.int64
