from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

orig   = Path("model/crnn.onnx")
target = Path("webapp/crnn_q.onnx")

# default quant_format = QOperator → emits QLinearConv, which ONNX-RT WASM supports
quantize_dynamic(
    model_input  = str(orig),
    model_output = str(target),
    weight_type  = QuantType.QInt8,  # 8-bit weights
)

size_mb = target.stat().st_size / 1024 / 1024
print(f"✅ Quantized ONNX → {target} ({size_mb:.1f} MiB)")
