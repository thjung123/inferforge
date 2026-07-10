import asyncio
import io
import logging

import aiohttp
import numpy as np
from PIL import Image

logger = logging.getLogger("builder")

_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


class ClipImagePreprocessor:
    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    async def _fetch_image(
        self, session: aiohttp.ClientSession, url: str
    ) -> np.ndarray | None:
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with session.get(url, timeout=timeout) as resp:
                if resp.status != 200:
                    logger.warning(
                        f"[Preprocessor] Failed to fetch image: {url} ({resp.status})"
                    )
                    return None
                content = await resp.read()
                return self._process_image_bytes(content)
        except Exception as e:
            logger.error(f"[Preprocessor] Error fetching image {url}: {e}")
            return None

    def _process_image_bytes(self, image_bytes: bytes) -> np.ndarray:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image = image.resize((self.image_size, self.image_size))
            arr = np.array(image).astype(np.float32) / 255.0
            arr = (arr - _CLIP_MEAN) / _CLIP_STD
            arr = arr.transpose(2, 0, 1)
            return arr
        except Exception as e:
            logger.error(f"[Preprocessor] Failed to process image bytes: {e}")
            return np.zeros((3, self.image_size, self.image_size), dtype=np.float32)

    async def run(self, image_urls: list[str]) -> np.ndarray:
        logger.info(
            f"[Preprocessor] Starting async image preprocessing for {len(image_urls)} images"
        )

        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_image(session, url) for url in image_urls]
            results = await asyncio.gather(*tasks)

        valid_images = [img for img in results if img is not None]
        if not valid_images:
            raise ValueError("No valid images were processed")

        batch = np.stack(valid_images, axis=0)
        logger.info(f"[Preprocessor] Completed. Batch shape: {batch.shape}")
        return batch
