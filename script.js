import { pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers/dist/transformers.min.js";

let embedder;
let svmSession;

// Load models
async function loadModels() {
    console.log("Loading models...");

    // Load MiniLM feature-extraction pipeline
    embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
    console.log("Embedder loaded");

    // Load SVM ONNX model
    svmSession = await ort.InferenceSession.create("./agnews_svm.onnx");
    console.log("SVM ONNX loaded");
}

loadModels();

// Mean pooling for token embeddings to get 384-dim sentence vector
function meanPooling(flatEmbedding) {
    const dim = 384;
    const tokenCount = flatEmbedding.length / dim;
    const sentenceEmbedding = new Float32Array(dim);

    for (let i = 0; i < dim; i++) {
        let sum = 0;
        for (let t = 0; t < tokenCount; t++) {
            sum += flatEmbedding[i + t * dim];
        }
        sentenceEmbedding[i] = sum / tokenCount;
    }

    return sentenceEmbedding;
}

// Classify news text
async function classifyNews() {
    const text = document.getElementById("newsInput").value;
    if (!text) return alert("Enter some text!");

    // 1️⃣ Get token embeddings
    const embeddingResult = await embedder(text);
           console.log(embeddingResult.data)
    // 2️⃣ Convert to 384-dim sentence vector
    const sentenceEmbedding = meanPooling(embeddingResult.data);

    // 3️⃣ Create ONNX Tensor
    const tensor = new ort.Tensor("float32", sentenceEmbedding, [1, 384]);

    // 4️⃣ Run SVM ONNX inference
    const output = await svmSession.run({ input: tensor });
    console.log(output)
    const outputTensor = Object.values(output)[0];
    console.log(outputTensor)
    const predictedIndex = Number(outputTensor.data[0]);
    console.log(predictedIndex)
    // 5️⃣ Map prediction to label
    // const predictedIndex = output.output_label.data[0];
    const labels = ["World", "Sports", "Business", "Sci/Tech"];
    document.getElementById("result").innerText = labels[predictedIndex-1];
}

// Button event
document.getElementById("classifyBtn").addEventListener("click", classifyNews);
