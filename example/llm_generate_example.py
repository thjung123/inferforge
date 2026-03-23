import asyncio

import httpx

API_URL = "http://localhost:8080/generate"
API_KEY = "test-key"


async def run_generate():
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain what TensorRT is in 2 sentences."},
        ],
        "max_tokens": 128,
        "temperature": 0.7,
    }

    headers = {"x-api-key": API_KEY}

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(API_URL, json=payload, headers=headers)
        print("Status:", resp.status_code)
        if resp.status_code != 200:
            print("Error:", resp.text)
            return

        data = resp.json()
        print(f"\nModel: {data['model']}")
        print(f"Response: {data['content']}")
        if data.get("usage"):
            print(f"Usage: {data['usage']}")


async def run_stream():
    payload = {
        "messages": [
            {"role": "user", "content": "Count from 1 to 10."},
        ],
        "max_tokens": 64,
        "stream": True,
    }

    headers = {"x-api-key": API_KEY}

    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream(
            "POST", API_URL, json=payload, headers=headers
        ) as resp:
            print("\n--- Streaming ---")
            async for line in resp.aiter_lines():
                if line:
                    print(line)


if __name__ == "__main__":
    asyncio.run(run_generate())
    asyncio.run(run_stream())
