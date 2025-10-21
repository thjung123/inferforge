import asyncio
import httpx

API_URL = "http://localhost:8080/infer/"
API_KEY = "test-key"

IMAGE_URLS = [
    "https://picsum.photos/id/237/512/512",
    "https://picsum.photos/id/1025/512/512",
    "https://picsum.photos/id/1003/512/512",
    "https://picsum.photos/id/1018/512/512",
    "https://picsum.photos/id/1020/512/512",
]

TEXTS = [
    "a photo of a cat",
    "a small bird",
    "a golden retriever dog",
    "a red flower",
    "a yellow sunflower",
]


async def run_clip_example():
    payload = {
        "model_name": "clip_ensemble",
        "inputs": {"image_urls": IMAGE_URLS, "texts": TEXTS},
    }

    headers = {"x-api-key": API_KEY}

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(API_URL, json=payload, headers=headers)
        print("Status:", response.status_code)
        if response.status_code != 200:
            print("Error:", response.text)
            return

        data = response.json()
        print(data)
        print("\n--- CLIP Similarity Matrix ---")
        sim = data["similarity"]

        for i, img_url in enumerate(IMAGE_URLS):
            best_idx = max(range(len(TEXTS)), key=lambda j: sim[i][j])
            print(
                f"[{i}] {img_url.split('/')[-1]}  →  {TEXTS[best_idx]} (score={sim[i][best_idx]:.3f})"
            )


if __name__ == "__main__":
    asyncio.run(run_clip_example())
