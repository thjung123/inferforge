# 🏗️ Triton Inference Architecture


```
        [Client]
            ↓ (REST API)
    [ FastAPI Gateway ]
        ├─ Auth, Rate Limiter
        ├─ Request Validation
        ├─ Retry / Circuit Breaker
        └─ gRPC/HTTP → Triton Client
            ↓ (gRPC)
    [ Triton Server ]
        ├─ Preprocessor (Python backend)
        ├─ TensorRT Inference Engine
        └─ Postprocessor (Python backend)
            ↓
        [ Response ]
```