import pytest
import numpy as np
from builder.processors.clip.image_preprocessor import ClipImagePreprocessor
from builder.processors.clip.text_preprocessor import ClipTextPreprocessor


@pytest.mark.asyncio
async def test_clip_image_preprocessor(tmp_path):
    preprocessor = ClipImagePreprocessor()
    urls = [
        "https://dummyimage.com/224x224/000/fff.png&text=test1",
        "https://dummyimage.com/224x224/000/fff.png&text=test2",
    ]

    result = await preprocessor.run(urls)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 3, 224, 224)
    assert result.dtype == np.float32


def test_clip_text_preprocessor():
    preprocessor = ClipTextPreprocessor()
    texts = ["a photo of a cat", "a dog sitting on a chair"]

    result = preprocessor.run(texts)

    assert "input_ids" in result
    assert "attention_mask" in result
    assert result["input_ids"].shape[0] == len(texts)
    assert result["input_ids"].shape[1] == preprocessor.max_length
    assert result["input_ids"].dtype == np.int32
