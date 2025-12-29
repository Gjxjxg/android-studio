package com.example.android_local_infer.offload

import java.util.UUID
import kotlin.random.Random

object MqttTopics {
    const val DEVICE_REQUESTS_TOPIC = "device/requests"
    const val DEVICE_RESPONSES_BASE = "device/responses"

    fun responseTopic(clientId: String): String = "$DEVICE_RESPONSES_BASE/$clientId"
}

object ClientIdFactory {

    fun newDeviceClientId(): String {
        val suffix = UUID.randomUUID().toString().replace("-", "").take(3)
        return "device-$suffix"
    }
}