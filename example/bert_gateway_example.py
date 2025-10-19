import asyncio
import httpx
import numpy as np

API_URL = "http://localhost:8080/infer/"
API_KEY = "test-key"

TEXTS = [
    "Artificial intelligence is transforming the world.",
    "The quick brown fox jumps over the lazy dog.",
    "A large language model trained on diverse data.",
    "The cat sat on the mat.",
    "Machine learning enables predictive analytics.",
]


async def run_bert_example():
    payload = {"model_name": "bert_ensemble", "inputs": {"texts": TEXTS}}

    headers = {"x-api-key": API_KEY}

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(API_URL, json=payload, headers=headers)
        print("Status:", response.status_code)
        if response.status_code != 200:
            print("Error:", response.text)
            return

        data = response.json()
        print("\n--- BERT Embedding Result ---")
        emb = np.array(data["bert_emb"], dtype=np.float32)

        print(f"Embedding shape: {emb.shape}")
        print(
            f"First row (first text embedding sample):\n{np.round(emb[0][:10], 4)} ..."
        )

        sim = (
            emb
            @ emb.T
            / (
                np.linalg.norm(emb, axis=1, keepdims=True)
                * np.linalg.norm(emb, axis=1, keepdims=True).T
            )
        )

        print("\n--- Sentence Similarity Matrix ---")
        for i in range(len(TEXTS)):
            row = " | ".join(f"{sim[i][j]:.2f}" for j in range(len(TEXTS)))
            print(f"[{i}] {row}")

        np.fill_diagonal(sim, -1.0)
        idx = np.unravel_index(np.argmax(sim), sim.shape)
        print(f'\nMost similar pair:\n  "{TEXTS[idx[0]]}"\n  ↔ "{TEXTS[idx[1]]}"')
        print(f"Cosine similarity = {sim[idx]:.3f}")


if __name__ == "__main__":
    asyncio.run(run_bert_example())
