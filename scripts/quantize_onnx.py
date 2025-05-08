from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

# point to your original
orig   = Path("model/crnn.onnx")
target = Path("webapp/crnn_q.onnx")

# dynamic quantize all weights to 8-bit integers
quantize_dynamic(
    model_input   = str(orig),
    model_output  = str(target),
    weight_type   = QuantType.QInt8
)

print(f"✅ Quantized ONNX written to {target}")
print(f"   size: {target.stat().st_size/1024/1024:.1f} MiB")
