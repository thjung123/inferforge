#!/usr/bin/env bash
set -e

echo "[STEP 1] Converting models to ONNX..."

python model/scripts/convert_to_onnx.py --config model/model_configs/bert_config.yaml
python model/scripts/convert_to_onnx.py --config model/model_configs/clip_text_encoder.yaml
python model/scripts/convert_to_onnx.py --config model/model_configs/clip_image_encoder.yaml

echo "[STEP 1] ONNX export complete."


if command -v nvidia-smi &> /dev/null; then
  echo "[STEP 2] GPU detected — building TensorRT engines..."

  mkdir -p /workspace/model_repository/bert_encoder/1
  mkdir -p /workspace/model_repository/clip_text_encoder/1
  mkdir -p /workspace/model_repository/clip_image_encoder/1

  chmod -R 777 /workspace/model_repository

  echo "[BUILD] BERT Encoder"
  chmod +x ./model/scripts/build_trt_engine/bert.sh
  chmod +x ./model/scripts/build_trt_engine/clip_text.sh
  chmod +x ./model/scripts/build_trt_engine/clip_image.sh

  ./model/scripts/build_trt_engine/bert.sh \
  model_repository/bert_encoder/bert_encoder.onnx fp16

  echo "[BUILD] CLIP Text Encoder"
  ./model/scripts/build_trt_engine/clip_text.sh \
  model_repository/clip_text_encoder/clip_text_encoder.onnx fp16


  echo "[BUILD] CLIP Image Encoder"
  ./model/scripts/build_trt_engine/clip_image.sh \
  model_repository/clip_image_encoder/clip_image_encoder.onnx fp16


  echo "[STEP 2] TensorRT engines built successfully."
else
  echo "[STEP 2] GPU not found — skipping TensorRT engine build."
fi


echo "[STEP 3] Generating Triton config.pbtxt files..."

python model/scripts/generate_triton_config.py --config model/model_configs/bert_config.yaml
python model/scripts/generate_triton_config.py --config model/model_configs/clip_text_encoder.yaml
python model/scripts/generate_triton_config.py --config model/model_configs/clip_image_encoder.yaml

echo "[STEP 3] Triton config generation complete."
echo "All models have been successfully built and configured."
