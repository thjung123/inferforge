Before running this,
you need to **generate all TensorRT `.plan` files and `config.pbtxt` files** first.
These are required for Triton to load and serve your models.

Use the Builder API to build and deploy models:

```bash
curl -X POST http://localhost:8090/build -H 'Content-Type: application/json' -d '{"model_type": "bert"}'
```

The resulting folder structure under model_repository/ should look like this:

```
model_repository/
├── bert_preprocessing/
│   ├── 1/
│   │   └── model.py
│   └── config.pbtxt
├── clip_text_preprocessing/
│   ├── 1/
│   │   └── model.py
│   └── config.pbtxt
├── clip_image_preprocessing/
│   ├── 1/
│   │   └── model.py
│   └── config.pbtxt
├── bert_encoder/
│   ├── 1/
│   │   └── model.plan
│   └── config.pbtxt
├── clip_text_encoder/
│   ├── 1/
│   │   └── model.plan
│   └── config.pbtxt
├── clip_image_encoder/
│   ├── 1/
│   │   └── model.plan
│   └── config.pbtxt
└── multimodal_ensemble/
│   ├── 1/
│   │   └── .gitkeep
│   └── config.pbtxt

```