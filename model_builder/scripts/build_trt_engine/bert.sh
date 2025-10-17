#!/bin/bash
set -e

ONNX_PATH=$1
PRECISION=${2:-fp16}

ENGINE_DIR=$(dirname "$ONNX_PATH")/1
ENGINE_PATH="${ENGINE_DIR}/model.plan"

mkdir -p "$ENGINE_DIR"

echo "[INFO] Building TensorRT engine"
echo "[INFO] ONNX: $ONNX_PATH"
echo "[INFO] Output: $ENGINE_PATH"
echo "[INFO] Precision: $PRECISION"

if [ ! -f "$ONNX_PATH" ]; then
  echo "[ERROR] ONNX model not found: $ONNX_PATH"
  exit 1
fi

if [ "$PRECISION" = "int8" ]; then
  echo "[WARN] INT8 calibration not implemented yet"
  trtexec --onnx="$ONNX_PATH" \
          --saveEngine="$ENGINE_PATH" \
          --int8 \
          --verbose
elif [ "$PRECISION" = "fp16" ]; then
  trtexec --onnx="$ONNX_PATH" \
          --saveEngine="$ENGINE_PATH" \
          --fp16 \
          --minShapes=input_ids:1x8,attention_mask:1x8,token_type_ids:1x8 \
          --optShapes=input_ids:32x64,attention_mask:32x64,token_type_ids:32x64 \
          --maxShapes=input_ids:256x128,attention_mask:256x128,token_type_ids:256x128
          --verbose
else
  trtexec --onnx="$ONNX_PATH" \
          --saveEngine="$ENGINE_PATH" \
          --verbose
fi

echo "[+] Engine built successfully: $ENGINE_PATH"
