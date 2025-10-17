#!/bin/bash
set -e

ONNX_PATH=$1
PRECISION=${2:-fp16}

ENGINE_DIR=$(dirname "$ONNX_PATH")/1
ENGINE_PATH="${ENGINE_DIR}/model.plan"

mkdir -p "$ENGINE_DIR"

echo "[INFO] Building TensorRT CLIP Text Encoder engine"
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
          --minShapes=text:1x77 \
          --optShapes=text:64x77 \
          --maxShapes=text:256x77 \
          --verbose
elif [ "$PRECISION" = "fp16" ]; then
  trtexec --onnx="$ONNX_PATH" \
          --saveEngine="$ENGINE_PATH" \
          --fp16 \
          --minShapes=text:1x77 \
          --optShapes=text:64x77 \
          --maxShapes=text:256x77 \
          --verbose
else
  trtexec --onnx="$ONNX_PATH" \
          --saveEngine="$ENGINE_PATH" \
          --minShapes=text:1x77 \
          --optShapes=text:64x77 \
          --maxShapes=text:256x77 \
          --verbose
fi

echo "[+] CLIP Text Encoder engine built successfully: $ENGINE_PATH"
