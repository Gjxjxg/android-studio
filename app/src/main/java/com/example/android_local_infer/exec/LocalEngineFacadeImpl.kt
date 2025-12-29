package com.example.android_local_infer.exec

import com.example.android_local_infer.infer.DelegateMode
import com.example.android_local_infer.infer.InferenceEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

class LocalEngineFacadeImpl(
    private val engine: InferenceEngine,
    private val topk: Int = 5
) : LocalEngineFacade {

    override suspend fun inferOnCpu(imageBytes: ByteArray): JSONObject =
        runLocal(imageBytes, DelegateMode.CPU)

    override suspend fun inferOnGpu(imageBytes: ByteArray): JSONObject =
        runLocal(imageBytes, DelegateMode.GPU)

    override suspend fun inferOnNpu(imageBytes: ByteArray): JSONObject =
        runLocal(imageBytes, DelegateMode.NNAPI)

    private suspend fun runLocal(imageBytes: ByteArray, mode: DelegateMode): JSONObject =
        withContext(Dispatchers.Default) {
            val (top, timing) = engine.run(imageBytes, topk, mode)

            val topJson = JSONArray().apply {
                for ((label, prob) in top) {
                    put(JSONObject().apply {
                        put("label", label)
                        put("prob", prob)
                    })
                }
            }

            val timingJson = JSONObject().apply {
                // timing: Map<String, Double> = {"preprocess":..., "infer":..., "total":...}
                for ((k, v) in timing) put(k, v)
            }

            JSONObject().apply {
                put("ok", true)
                put("mode", mode.name.lowercase())
                put("topk", topJson)
                put("timing_ms", timingJson)

                if (top.isNotEmpty()) {
                    put("top1_label", top[0].first)
                    put("top1_prob", top[0].second)
                }
            }
        }
}