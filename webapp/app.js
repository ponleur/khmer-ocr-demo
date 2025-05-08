import { CHAR_MAP } from "./vocab.js";

// DOM refs
const picker = document.getElementById("filepicker");
const status = document.getElementById("status");
const preview = document.getElementById("preview");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const result = document.getElementById("result");

let session = null;

// 1) Load ONNX model
status.innerText = "🔄 Loading ONNX model…";
ort.InferenceSession.create( "crnn_q.onnx")
    .then(s => {
        session = s;
        status.innerText = "✅ Model loaded! Select an image.";
        console.log("✅ ONNX session loaded");
    })
    .catch(err => {
        console.error("❌ Failed to load ONNX model:", err);
        status.innerText = "❌ Model load failed; check console.";
    });

// 2) Preprocess: grayscale, resize, normalize → tensor [1,1,32,128]
function preprocessForONNX(img) {
    // draw into hidden 128×32 canvas
    ctx.clearRect(0, 0, 128, 32);
    ctx.drawImage(img, 0, 0, 128, 32);

    // extract RGBA pixel data
    const { data: px } = ctx.getImageData(0, 0, 128, 32);
    const buffer = new Float32Array(1 * 1 * 32 * 128);

    // fill buffer with normalized grayscale
    for (let y = 0; y < 32; y++) {
        for (let x = 0; x < 128; x++) {
            const i = (y * 128 + x) * 4;
            // luma conversion
            const gray = (0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2]) / 255;
            // match transforms.Normalize((0.5,),(0.5,))
            buffer[y * 128 + x] = (gray - 0.5) / 0.5;
        }
    }

    // return the ONNX input tensor
    return new ort.Tensor("float32", buffer, [1, 1, 32, 128]);
}

// 3) Greedy CTC decode for output logits [T,B,C]
function ctcGreedyDecode(logits, dims) {
    const [T, B, C] = dims;   // B should be 1
    const blank = 0;
    let prev = blank;
    let text = "";

    for (let t = 0; t < T; t++) {
        // find argmax over classes
        let maxI = 0, maxV = logits[t * C];
        for (let c = 1; c < C; c++) {
            const v = logits[t * C + c];
            if (v > maxV) {
                maxV = v;
                maxI = c;
            }
        }
        // collapse repeats & skip blanks
        if (maxI !== blank && maxI !== prev) {
            text += CHAR_MAP[maxI - 1];
        }
        prev = maxI;
    }

    return text;
}

// NEW: core inference pipeline given an HTMLImageElement
async function runOnImage(img) {
    if (!session) {
        status.innerText = "⌛ Model still loading…";
        return;
    }
    // show preview
    preview.src = img.src;
    preview.style.display = "block";

    status.innerText = "🔄 Running inference…";
    result.innerText = "–";

    // preprocess → tensor → run
    const inputTensor = preprocessForONNX(img);
    const outputMap = await session.run({ input: inputTensor });
    const outTensor = outputMap.output;
    console.log("Output dims:", outTensor.dims);

    // decode & display
    const text = ctcGreedyDecode(outTensor.data, outTensor.dims);
    result.innerText = text || "–";
    status.innerText = "✅ Done!";
}

// wire up file picker
picker.addEventListener("change", () => {
    const file = picker.files[0];
    if (!file) return;
    const img = new Image();
    img.onload = () => runOnImage(img);
    img.src = URL.createObjectURL(file);
});

// wire up sample thumbnails
document.querySelectorAll(".sample").forEach(el => {
    el.addEventListener("click", () => {
        const img = new Image();
        img.onload = () => runOnImage(img);
        img.src = el.src;  // already a URL
    });
});