import io
import asyncio
import aiohttp
import numpy as np
from PIL import Image
from typing import List, Union
from gateway.utils.logger import logger


class ClipImagePreprocessor:
    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    async def _fetch_image(
        self, session: aiohttp.ClientSession, url: str
    ) -> Union[np.ndarray, None]:
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession() as session:
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
            arr = np.array(image).astype(np.float32) / 255.0  # normalize to [0,1]
            arr = arr.transpose(2, 0, 1)  # HWC → CHW
            return arr
        except Exception as e:
            logger.error(f"[Preprocessor] Failed to process image bytes: {e}")
            return np.zeros((3, self.image_size, self.image_size), dtype=np.float32)

    async def run(self, image_urls: List[str]) -> np.ndarray:
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
