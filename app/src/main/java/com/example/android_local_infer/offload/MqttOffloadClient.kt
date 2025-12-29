package com.example.android_local_infer.offload

import android.util.Base64
import com.example.android_local_infer.infer.ModelCfg
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import org.eclipse.paho.client.mqttv3.*
import org.eclipse.paho.client.mqttv3.persist.MemoryPersistence
import org.json.JSONObject
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap

class MqttOffloadClient(
    brokerHost: String,
    brokerPort: Int = 1883,
    val clientId: String = "device-" + UUID.randomUUID().toString().replace("-", "").take(3),
) {
    private val brokerUri = "tcp://$brokerHost:$brokerPort"

    private val mqtt: MqttClient = MqttClient(brokerUri, clientId, MemoryPersistence())

    private val pending = ConcurrentHashMap<String, CompletableDeferred<JSONObject>>()

    @Volatile private var connected: Boolean = false

    companion object {
        private const val REQ_TOPIC = "device/requests"
        private fun respTopic(clientId: String) = "device/responses/$clientId"
    }

    suspend fun connect() = withContext(Dispatchers.IO) {
        if (connected && mqtt.isConnected) return@withContext

        val opts = MqttConnectOptions().apply {
            isCleanSession = true
            keepAliveInterval = 60
            connectionTimeout = 10
            isAutomaticReconnect = true
        }

        mqtt.setCallback(object : MqttCallback {
            override fun connectionLost(cause: Throwable?) {
                connected = false
            }

            override fun messageArrived(topic: String?, message: MqttMessage?) {
                if (message == null) return
                val json = runCatching { JSONObject(String(message.payload)) }.getOrNull() ?: return
                val reqId = json.optString("request_id", "")
                if (reqId.isNotBlank()) {
                    pending.remove(reqId)?.complete(json)
                }
            }

            override fun deliveryComplete(token: IMqttDeliveryToken?) {}
        })


        mqtt.connect(opts)

        mqtt.subscribe(respTopic(clientId), 1)

        connected = true
    }

    fun disconnect() {
        runCatching { if (mqtt.isConnected) mqtt.disconnect() }
        connected = false
        pending.forEach { (_, d) -> if (!d.isCompleted) d.cancel() }
        pending.clear()
    }

    suspend fun inferImage(
        imageBytes: ByteArray,
        model: ModelCfg,
        qos: Int = 1,
        timeoutMs: Long = 60_000
    ): JSONObject = withContext(Dispatchers.IO) {
        if (!connected || !mqtt.isConnected) connect()

        val requestId = UUID.randomUUID().toString().replace("-", "")

        val b64 = Base64.encodeToString(imageBytes, Base64.NO_WRAP)

        val reqJson = JSONObject().apply {
            put("client_id", clientId)
            put("request_id", requestId)
            put("payload_type", "image")
            put("image_b64", b64)
            put("model_name", model.name)
        }

        val deferred = CompletableDeferred<JSONObject>()
        pending[requestId] = deferred

        val msg = MqttMessage(reqJson.toString().toByteArray()).apply {
            this.qos = qos
            isRetained = false
        }

        mqtt.publish(REQ_TOPIC, msg)

        try {
            withTimeout(timeoutMs) { deferred.await() }
        } finally {
            pending.remove(requestId)
        }
    }
}
