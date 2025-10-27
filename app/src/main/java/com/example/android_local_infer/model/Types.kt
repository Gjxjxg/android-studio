package com.example.android_local_infer.model

data class InferRequest(
    val image_b64: String,
    val topk: Int? = 5,
    val delegate: String? = null // "cpu" | "nnapi" | "gpu"
)

data class InferResponse(
    val topk: List<Pair<String, Float>>, // [(label, prob), ...]
    val timing_ms: Map<String, Double>
)
