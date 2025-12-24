package com.example.android_local_infer.infer

import android.content.Context
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.gpu.GpuDelegate
import java.nio.ByteBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.atomic.AtomicReference

private const val TAG = "InferenceEngine"

enum class DelegateMode { CPU, NNAPI, GPU }


sealed class ModelCfg(val name: String, val file: String, val inputSize: Int) {
    object MV3   : ModelCfg("mv3",  "mobilenet_v3_small_224_1.0_float.tflite", 224)
    object Eff0  : ModelCfg("eff0", "efficientnet_lite0_float.tflite",        224)
    object Eff1  : ModelCfg("eff1", "efficientnet_lite1_float.tflite",        240)
    object Eff2  : ModelCfg("eff2", "efficientnet_lite2_float.tflite",        260)
    object Eff3  : ModelCfg("eff3", "efficientnet_lite3_float.tflite",        280)
    object Eff4  : ModelCfg("eff4", "efficientnet_lite4_float.tflite",        300)

    companion object {
        fun from(s: String): ModelCfg = when (s.lowercase()) {
            "eff0" -> Eff0
            "eff1" -> Eff1
            "eff2" -> Eff2
            "eff3" -> Eff3
            "eff4" -> Eff4
            "mv3", "mobilenet" -> MV3
            else -> MV3
        }

        fun all(): List<ModelCfg> = listOf(MV3, Eff0, Eff1, Eff2, Eff3, Eff4)
    }
}

class InferenceEngine(
    private val ctx: Context,
    private var cfg: ModelCfg = ModelCfg.MV3
) {
    private val labels: List<String> =
        AssetLoader.loadLabels(ctx, "labels.txt")

    private val interpRef = AtomicReference<Interpreter?>()
    private var current: DelegateMode = DelegateMode.CPU
    private var nnapi: NnApiDelegate? = null
    private var gpu: GpuDelegate? = null


    @Volatile private var cachedInput: ByteBuffer? = null
    @Volatile private var cachedW: Int = -1
    @Volatile private var cachedH: Int = -1

    @Synchronized
    fun switchModel(newCfg: ModelCfg) {
        if (cfg.name == newCfg.name) return
        Log.i(TAG, "Switching model ${cfg.name} -> ${newCfg.name}")
        releaseInterpreter()
        cfg = newCfg

        cachedInput = null
        cachedW = -1
        cachedH = -1
    }

    private fun releaseInterpreter() {
        runCatching { interpRef.get()?.close() }
        runCatching { nnapi?.close() }
        runCatching { gpu?.close() }
        interpRef.set(null)
        nnapi = null
        gpu = null
        current = DelegateMode.CPU
    }

    private fun buildNnapiDelegateOrNull(): NnApiDelegate? {
        return try {
            try {
                val optCls = Class.forName("org.tensorflow.lite.nnapi.NnApiDelegate\$Options")
                val opts = optCls.getDeclaredConstructor().newInstance().apply {
                    runCatching { optCls.getMethod("setAllowFp16", Boolean::class.javaPrimitiveType).invoke(this, true) }
                    runCatching { optCls.getMethod("setUseNnapiCpu", Boolean::class.javaPrimitiveType).invoke(this, false) }
                }
                val ctor = Class.forName("org.tensorflow.lite.nnapi.NnApiDelegate").getDeclaredConstructor(optCls)
                ctor.newInstance(opts) as NnApiDelegate
            } catch (_: Throwable) {
                NnApiDelegate()
            }
        } catch (t: Throwable) {
            Log.w(TAG, "NNAPI delegate creation failed; fallback to CPU", t)
            null
        }
    }

    private fun buildGpuDelegateOrNull(): GpuDelegate? {
        return try {
            try {
                val factoryCls = Class.forName("org.tensorflow.lite.gpu.GpuDelegateFactory")
                val flavorCls  = Class.forName("org.tensorflow.lite.gpu.GpuDelegateFactory\$RuntimeFlavor")
                val best       = flavorCls.getField("BEST").get(null)
                val create     = factoryCls.getMethod("create", flavorCls)
                val factory    = factoryCls.getDeclaredConstructor().newInstance()
                create.invoke(factory, best) as GpuDelegate
            } catch (_: Throwable) {
                GpuDelegate()
            }
        } catch (t: Throwable) {
            Log.w(TAG, "GPU delegate creation failed; fallback to CPU", t)
            null
        }
    }

    private fun loadModelMapped(): MappedByteBuffer =
        ctx.assets.openFd(cfg.file).use { fd ->
            fd.createInputStream().channel.map(
                FileChannel.MapMode.READ_ONLY,
                fd.startOffset,
                fd.declaredLength
            )
        }

    @Synchronized
    private fun ensureInterpreter(mode: DelegateMode): Interpreter {
        interpRef.get()?.let { if (current == mode) return it }

        releaseInterpreter()

        val opts = Interpreter.Options().apply {

            runCatching {
                val m = this::class.java.getMethod("setUseXNNPACK", Boolean::class.javaPrimitiveType)
                m.invoke(this, true)
            }
            runCatching {
                val m = this::class.java.getMethod("setNumThreads", Int::class.javaPrimitiveType)
                m.invoke(this, Runtime.getRuntime().availableProcessors().coerceAtMost(4))
            }
        }

        when (mode) {
            DelegateMode.CPU -> Unit
            DelegateMode.NNAPI -> buildNnapiDelegateOrNull()?.let { nnapi = it; opts.addDelegate(it) }
            DelegateMode.GPU   -> buildGpuDelegateOrNull()?.let { gpu = it; opts.addDelegate(it) }
        }

        val interpreter = Interpreter(loadModelMapped(), opts)
        interpRef.set(interpreter)
        current = mode
        Log.i(TAG, "Interpreter created: model=${cfg.name}, backend=$current, input=${cfg.inputSize}")
        return interpreter
    }


    private fun getOrCreateInput(imageBytes: ByteArray, w: Int, h: Int): Pair<ByteBuffer, Double> {
        val buf = cachedInput
        if (buf != null && cachedW == w && cachedH == h) {
            buf.rewind()
            return buf to 0.0
        }
        val (newBuf, preMs) = ImageUtils.preprocess(imageBytes, w, h)
        cachedInput = newBuf
        cachedW = w
        cachedH = h
        return newBuf to preMs
    }


    fun run(imageBytes: ByteArray, topk: Int, mode: DelegateMode)
            : Pair<List<Pair<String, Float>>, Map<String, Double>> {

        val interpreter = ensureInterpreter(mode)

        val in0 = interpreter.getInputTensor(0)
        val shape = in0.shape()
        val h = shape[1]
        val w = shape[2]

        val (inputTensor, preMs) = getOrCreateInput(imageBytes, w, h)

        val out0 = interpreter.getOutputTensor(0)
        val outLen = out0.numElements()
        val output = Array(1) { FloatArray(outLen) }

        val t0 = SystemClock.elapsedRealtimeNanos()
        interpreter.run(inputTensor, output)
        val t1 = SystemClock.elapsedRealtimeNanos()

        val probs = output[0]
        val top = TopK.topK(probs, labels, topk)

        val inferMs = (t1 - t0) / 1e6
        val timing = mapOf("preprocess" to preMs, "infer" to inferMs, "total" to (preMs + inferMs))
        return top to timing
    }


    fun runLoop(imageBytes: ByteArray, loops: Int, mode: DelegateMode): Map<String, Any> {
        val interpreter = ensureInterpreter(mode)

        val in0 = interpreter.getInputTensor(0)
        val shape = in0.shape()
        val h = shape[1]
        val w = shape[2]

        val (inputTensor, preMs) = getOrCreateInput(imageBytes, w, h)

        val out0 = interpreter.getOutputTensor(0)
        val outLen = out0.numElements()
        val output = Array(1) { FloatArray(outLen) }

        repeat(5) {
            inputTensor.rewind()
            interpreter.run(inputTensor, output)
        }

        val t0 = SystemClock.elapsedRealtime()
        repeat(loops) {
            inputTensor.rewind()
            interpreter.run(inputTensor, output)
        }
        val t1 = SystemClock.elapsedRealtime()

        return mapOf(
            "preprocess_ms" to preMs,
            "loops" to loops,
            "elapsed_ms" to (t1 - t0),
            "avg_ms_per_infer" to ((t1 - t0).toDouble() / loops.toDouble())
        )
    }
}

private object AssetLoader {
    fun loadLabels(ctx: Context, filename: String): List<String> =
        ctx.assets.open(filename).bufferedReader().readLines().filter { it.isNotBlank() }
}

object TopK {
    fun topK(probs: FloatArray, labels: List<String>, k: Int): List<Pair<String, Float>> =
        probs.withIndex()
            .sortedByDescending { it.value }
            .take(k)
            .map { (i, p) -> (labels.getOrNull(i) ?: "class_$i") to p }
}
