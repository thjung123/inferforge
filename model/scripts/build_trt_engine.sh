#!/bin/bash
# build_trt_engine.sh
# Converts ONNX model to TensorRT engine (.plan)
# Usage: ./build_trt_engine.sh model_repository/bert_encoder/onnx/bert_encoder.onnx fp16

set -e

ONNX_PATH=$1
PRECISION=${2:-fp16}
ENGINE_DIR=$(dirname $(dirname "$ONNX_PATH"))/1
ENGINE_PATH="${ENGINE_DIR}/model.plan"

mkdir -p "$ENGINE_DIR"

echo "[INFO] Building TensorRT engine: $ENGINE_PATH"
echo "[INFO] Precision: $PRECISION"

if [ "$PRECISION" = "int8" ]; then
  echo "[WARN] INT8 mode requires calibration cache (not implemented yet)"
  trtexec --onnx="$ONNX_PATH" --saveEngine="$ENGINE_PATH" --int8 --verbose
elif [ "$PRECISION" = "fp16" ]; then
  trtexec --onnx="$ONNX_PATH" --saveEngine="$ENGINE_PATH" --fp16 --workspace=4096
else
  trtexec --onnx="$ONNX_PATH" --saveEngine="$ENGINE_PATH" --workspace=4096
fi

echo "[+] Engine built successfully: $ENGINE_PATH"
