import requests
import json
import numpy as np

TRITON_URL = "http://localhost:8000/v2/models/bert_ensemble/infer"

TEXTS = [
    "This is a test sentence.",
    "I love natural language processing.",
    "Transformers are powerful models.",
    "The cat sat on the mat.",
    "Deep learning enables many AI applications.",
]

payload = {
    "inputs": [
        {
            "name": "TEXTS",
            "datatype": "BYTES",
            "shape": [1, len(TEXTS)],
            "data": [TEXTS],
        }
    ]
}

print(f"Sending request to {TRITON_URL} ...")
resp = requests.post(TRITON_URL, json=payload)
print("Status:", resp.status_code)

data = resp.json()
print(json.dumps(data, indent=2)[:400] + " ...")

if "outputs" in data:
    output = data["outputs"][0]
    name = output["name"]
    shape = output.get("shape", [])
    dtype = output.get("datatype", "")
    emb = output.get("data", [])

    arr = np.array(emb, dtype=np.float32).reshape(shape)
    print(f"\n--- BERT output: {name}, shape={arr.shape}, dtype={dtype} ---")
    print(f"First embedding (first 10 dims): {arr[0][:10].tolist()}")
else:
    print("\nTriton returned an error:")
    print(data)
