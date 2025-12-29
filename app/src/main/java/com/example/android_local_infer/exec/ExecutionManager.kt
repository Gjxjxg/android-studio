package com.example.android_local_infer.exec

import com.example.android_local_infer.infer.ModelCfg
import com.example.android_local_infer.offload.MqttOffloadClient
import org.json.JSONObject

enum class ExecMode { CPU, GPU, NPU, OFFLOAD }

class ExecutionManager(
    private val offloadClient: MqttOffloadClient,
    private val localEngine: LocalEngineFacade
) {

    suspend fun runTask(
        mode: ExecMode,
        imageBytes: ByteArray,
        model: ModelCfg
    ): JSONObject {
        return when (mode) {
            ExecMode.OFFLOAD -> {
                // MQTT offload
                offloadClient.inferImage(imageBytes, model)
            }
            ExecMode.CPU -> localEngine.inferOnCpu(imageBytes)
            ExecMode.GPU -> localEngine.inferOnGpu(imageBytes)
            ExecMode.NPU -> localEngine.inferOnNpu(imageBytes)
        }
    }
}