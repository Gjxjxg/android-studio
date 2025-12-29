package com.example.android_local_infer.offload

data class OffloadRequest(
    val client_id: String,
    val request_id: String,
    val payload_type: String = "image",
    val image_b64: String
)

data class OffloadResponse(
    val ok: Boolean,
    val error: String? = null,
    val model: String? = null,
    val precision: String? = null,
    val top1: Any? = null,
    val top5: Any? = null,
    val timing_ms: Any? = null,
    val request_id: String? = null,
    val client_id: String? = null
)