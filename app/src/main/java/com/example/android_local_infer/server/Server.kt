package com.example.android_local_infer.server

import android.content.Context
import android.util.Base64
import android.util.Log
import com.example.android_local_infer.infer.DelegateMode
import com.example.android_local_infer.infer.InferenceEngine
import com.example.android_local_infer.infer.ModelCfg
import com.example.android_local_infer.model.InferRequest
import com.example.android_local_infer.model.InferResponse
import io.ktor.http.HttpStatusCode
import io.ktor.serialization.jackson.jackson
import io.ktor.server.application.*
import io.ktor.server.cio.CIO
import io.ktor.server.engine.EmbeddedServer
import io.ktor.server.engine.embeddedServer
import io.ktor.server.plugins.contentnegotiation.ContentNegotiation
import io.ktor.server.request.*
import io.ktor.server.response.*
import io.ktor.server.routing.*

private const val TAG = "LocalServer"

object LocalServer {
    private var engine: EmbeddedServer<*, *>? = null
    private lateinit var infer: InferenceEngine

    @Volatile var currentDelegate: DelegateMode = DelegateMode.CPU
    @Volatile var currentModel: ModelCfg = ModelCfg.MV3

    fun start(ctx: Context) {
        if (engine != null) {
            Log.i(TAG, "Server already running on 127.0.0.1:8080")
            return
        }
        infer = InferenceEngine(ctx, cfg = currentModel)

        engine = embeddedServer(CIO, port = 8080) {
            install(ContentNegotiation) { jackson() }

            routing {
                get("/health") { call.respond(mapOf("ok" to true)) }

                get("/set_delegate") {
                    val mode = (call.request.queryParameters["mode"] ?: "cpu").lowercase()
                    currentDelegate = when (mode) {
                        "gpu" -> DelegateMode.GPU
                        "nnapi", "npu" -> DelegateMode.NNAPI
                        else -> DelegateMode.CPU
                    }
                    call.respond(mapOf("ok" to true, "delegate" to currentDelegate.name))
                }

                get("/set_model") {
                    val m = call.request.queryParameters["m"] ?: "mv3"
                    val newCfg = ModelCfg.from(m)
                    currentModel = newCfg
                    infer.switchModel(newCfg)
                    call.respond(mapOf("ok" to true, "model" to newCfg.name, "input" to newCfg.inputSize))
                }

                get("/models") {
                    call.respond(ModelCfg.all().map { mapOf("name" to it.name, "file" to it.file, "input" to it.inputSize) })
                }

                post("/infer") {
                    try {
                        val req = call.receive<InferRequest>()
                        val bytes = Base64.decode(req.image_b64, Base64.DEFAULT)
                        val topk = req.topk ?: 5
                        val delegate = (req.delegate ?: currentDelegate.name).lowercase()
                        val mode = when (delegate) {
                            "gpu" -> DelegateMode.GPU
                            "nnapi", "npu" -> DelegateMode.NNAPI
                            else -> DelegateMode.CPU
                        }

                        val (result, timing) = infer.run(bytes, topk, mode)
                        call.respond(InferResponse(result, timing))
                    } catch (t: Throwable) {
                        Log.e(TAG, "infer failed", t)
                        call.respond(HttpStatusCode.BadRequest, mapOf("ok" to false, "error" to (t.message ?: "unknown")))
                    }
                }
            }
        }

        engine!!.start(false)
        Log.i(TAG, "Server started")
    }

    fun stop() {
        try { engine?.stop() } finally { engine = null }
    }
}
