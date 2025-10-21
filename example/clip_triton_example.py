import requests
import json

TRITON_URL = "http://localhost:8000/v2/models/clip_ensemble/infer"

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

payload = {
    "inputs": [
        {
            "name": "IMAGE_URLS",
            "datatype": "BYTES",
            "shape": [1, len(IMAGE_URLS)],
            "data": IMAGE_URLS,
        },
        {"name": "TEXTS", "datatype": "BYTES", "shape": [1, len(TEXTS)], "data": TEXTS},
    ]
}

resp = requests.post(TRITON_URL, json=payload)
print("Status:", resp.status_code)
print("Response:", json.dumps(resp.json(), indent=2))
