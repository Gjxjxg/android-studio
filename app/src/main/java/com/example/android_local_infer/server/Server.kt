package com.example.android_local_infer.server

import android.content.Context
import android.util.Base64
import android.util.Log
import com.example.android_local_infer.infer.InferenceEngine
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

object LocalServer {
    private var engine: EmbeddedServer<*, *>? = null
    private lateinit var infer: InferenceEngine
    private const val TAG = "LocalServer"

    @Volatile
    var currentDelegate: InferenceEngine.DelegateMode = InferenceEngine.DelegateMode.CPU

    fun start(ctx: Context) {
        if (engine != null) {
            Log.i(TAG, "Server already running on 127.0.0.1:8080")
            return
        }

        infer = InferenceEngine(ctx)
        Log.i(TAG, "Starting Ktor server on 127.0.0.1:8080")

        engine = embeddedServer(CIO, port = 8080) {
            install(ContentNegotiation) { jackson() }

            routing {
                get("/health") {
                    Log.i(TAG, "Health check called")
                    call.respond(mapOf("ok" to true))
                }

                get("/set_delegate") {
                    val mode = call.request.queryParameters["mode"] ?: "cpu"
                    currentDelegate = InferenceEngine.DelegateMode.from(mode)
                    call.respond(mapOf("ok" to true, "delegate" to currentDelegate.name))
                }

                post("/infer") {
                    try {
                        val req = call.receive<InferRequest>()
                        val bytes = Base64.decode(req.image_b64, Base64.DEFAULT)
                        val topk = req.topk ?: 5
                        val delegate = InferenceEngine.DelegateMode.from(req.delegate ?: currentDelegate.name)

                        val (result, timing) = infer.run(bytes, topk, delegate)
                        call.respond(InferResponse(result, timing))
                    } catch (t: Throwable) {
                        Log.e(TAG, "infer failed", t)
                        call.respond(
                            HttpStatusCode.BadRequest,
                            mapOf("ok" to false, "error" to (t.message ?: "unknown"))
                        )
                    }
                }
            }
        }

        engine!!.start(false)
        Log.i(TAG, "Server started")
    }

    fun stop() {
        try {
            engine?.stop()
        } finally {
            engine = null
        }
    }
}
