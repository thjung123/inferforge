Before running this,

you need to generate the plan file and config.pbtxt first inside the `model/scripts` directory.

folder architecture

```
model_repository/
├── clip_image_preprocessing/
│   ├── 1/
│   │   └── model.py
│   └── config.pbtxt
├── clip_text_preprocessing/
│   ├── 1/
│   │   └── model.py
│   └── config.pbtxt
├── bert_preprocessing/
│   ├── 1/
│   │   └── model.py
│   └── config.pbtxt
├── clip_image_encoder/
│   ├── 1/
│   │   └── model.plan
│   └── config.pbtxt
├── clip_text_encoder/
│   ├── 1/
│   │   └── model.plan
│   └── config.pbtxt
├── bert_encoder/
│   ├── 1/
│   │   └── model.plan
│   └── config.pbtxt
└── multimodal_ensemble/
│   ├── 1/
│   │   └── .gitkeep
│   └── config.pbtxt
```