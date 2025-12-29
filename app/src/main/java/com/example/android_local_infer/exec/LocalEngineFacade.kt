package com.example.android_local_infer.exec

import org.json.JSONObject

interface LocalEngineFacade {
    suspend fun inferOnCpu(imageBytes: ByteArray): JSONObject
    suspend fun inferOnGpu(imageBytes: ByteArray): JSONObject
    suspend fun inferOnNpu(imageBytes: ByteArray): JSONObject
}