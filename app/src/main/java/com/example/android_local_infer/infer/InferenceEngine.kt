package com.example.android_local_infer.infer

import android.content.Context
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.gpu.GpuDelegateFactory
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.nio.MappedByteBuffer
import java.util.concurrent.atomic.AtomicReference
import java.nio.channels.FileChannel

private const val TAG = "InferenceEngine"

class InferenceEngine(private val ctx: Context) {

    enum class DelegateMode { CPU, NNAPI, GPU;
        companion object {
            fun from(name: String): DelegateMode = when (name.lowercase()) {
                "nnapi" -> NNAPI
                "gpu" -> GPU
                else -> CPU
            }
        }
    }

    private val model: MappedByteBuffer =
        AssetLoader.loadModel(ctx, "mobilenet_v3_small_224_1.0_float.tflite")
    private val labels: List<String> =
        AssetLoader.loadLabels(ctx, "labels.txt")

    private val interpRef = AtomicReference<Interpreter>()
    private var current: DelegateMode = DelegateMode.CPU
    private var nnapi: NnApiDelegate? = null
    private var gpu: Delegate? = null  // It may be a GpuDelegate or a delegate created by Factory

    /** Create/reuse Interpreter; create delegates according to the pattern,
     * and fall back to CPU+XNNPACK if failure occurs */
    private fun buildInterpreter(mode: DelegateMode): Interpreter {
        interpRef.get()?.let { existing ->
            if (current == mode) return existing
            try { existing.close() } catch (_: Throwable) {}
        }
        // Clean up old delegates
        try { nnapi?.close() } catch (_: Throwable) {}
        try { (gpu as? AutoCloseable)?.close() } catch (_: Throwable) {}
        nnapi = null
        gpu = null

        val opts = Interpreter.Options().apply {
            // Use XNNPACK when running on the CPU; it can be enabled even if a delegate fails and falls back.
            setUseXNNPACK(true)
            setNumThreads(Runtime.getRuntime().availableProcessors().coerceAtMost(4))
        }

        when (mode) {
            DelegateMode.CPU -> {
                Log.i(TAG, "Using CPU (XNNPACK)")
            }
            DelegateMode.NNAPI -> {
                nnapi = buildNnapiDelegateOrNull()
                if (nnapi != null) {
                    opts.addDelegate(nnapi)
                    Log.i(TAG, "Using NNAPI delegate")
                } else {
                    Log.w(TAG, "NNAPI not available; fallback to CPU")
                }
            }
            DelegateMode.GPU -> {
                gpu = buildGpuDelegateOrNull()
                if (gpu != null) {
                    opts.addDelegate(gpu)
                    Log.i(TAG, "Using GPU delegate")
                } else {
                    Log.w(TAG, "GPU not available; fallback to CPU")
                }
            }
        }

        val intr = Interpreter(model, opts)
        interpRef.set(intr)
        current = mode
        return intr
    }

    /** GPU: Prioritize Factory API; if the class is missing or not supported, try legacy, and if that fails, return null */
    private fun buildGpuDelegateOrNull(): Delegate? {
        return try {
            val compat = CompatibilityList()
            if (!compat.isDelegateSupportedOnThisDevice) {
                Log.w(TAG, "GPU not supported by CompatibilityList; fallback to CPU")
                return null
            }

            // 1) Reflection loading GpuDelegateFactory
            val factoryClass = Class.forName("org.tensorflow.lite.gpu.GpuDelegateFactory")
            val factory = factoryClass.getDeclaredConstructor().newInstance()

            // 2) First try the signature create(RuntimeFlavor)
            try {
                val flavorClass = Class.forName("org.tensorflow.lite.gpu.GpuDelegateFactory\$RuntimeFlavor")
                val best = java.lang.Enum.valueOf(flavorClass as Class<out Enum<*>>, "BEST")
                val m = factoryClass.getMethod("create", flavorClass)
                @Suppress("UNCHECKED_CAST")
                val delegate = m.invoke(factory, best) as Delegate
                Log.i(TAG, "GPU delegate created via Factory(RuntimeFlavor)")
                return delegate
            } catch (_: Throwable) { /* Try again Options version */ }

            // 3) Try the create(Options) signature again
            try {
                val optsClass = Class.forName("org.tensorflow.lite.gpu.GpuDelegateFactory\$Options")
                val opts = optsClass.getDeclaredConstructor().newInstance()
                val m = factoryClass.getMethod("create", optsClass)
                @Suppress("UNCHECKED_CAST")
                val delegate = m.invoke(factory, opts) as Delegate
                Log.i(TAG, "GPU delegate created via Factory(Options)")
                return delegate
            } catch (_: Throwable) { /* Try legacy again */ }

            // 4) Finally try the old GpuDelegate()
            try {
                val legacy = GpuDelegate()
                Log.i(TAG, "GPU delegate created via legacy GpuDelegate()")
                legacy
            } catch (legacy: Throwable) {
                Log.w(TAG, "Legacy GpuDelegate() failed; fallback to CPU", legacy)
                null
            }
        } catch (t: Throwable) {
            Log.w(TAG, "GPU delegate creation failed; fallback to CPU", t)
            null
        }
    }
    /** NNAPI: Some models/ROMs may fail; if failed, return null */
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


    /**
     * Run inference
     * @return Pair(topk result (label, probability), timing in milliseconds)
     */
    fun run(imageBytes: ByteArray, topk: Int, mode: DelegateMode)
            : Pair<List<Pair<String, Float>>, Map<String, Double>> {

        val interpreter = buildInterpreter(mode)

        // preprocessing
        val (tensor, preMs) = ImageUtils.preprocess(imageBytes, 224, 224)

        // The output tensor length is read according to the model to avoid out-of-bounds
        val outTensor = interpreter.getOutputTensor(0)
        val outElems = outTensor.numElements()  // Usually 1001
        val output = Array(1) { FloatArray(outElems) }

        val t0 = SystemClock.elapsedRealtimeNanos()
        interpreter.run(tensor, output)
        val t1 = SystemClock.elapsedRealtimeNanos()

        val probs = output[0]
        val top = TopK.topK(probs, labels, topk)

        val timing = mapOf(
            "preprocess" to preMs,
            "infer" to ((t1 - t0) / 1e6),
            "total" to (preMs + (t1 - t0) / 1e6)
        )
        return top to timing
    }
}

/* ---------- helpers ---------- */

private object AssetLoader {
    fun loadModel(ctx: Context, filename: String): MappedByteBuffer =
        ctx.assets.openFd(filename).use { fd ->
            fd.createInputStream().channel.map(
                FileChannel.MapMode.READ_ONLY,
                fd.startOffset, fd.declaredLength
            )
        }

    fun loadLabels(ctx: Context, filename: String): List<String> =
        ctx.assets.open(filename).bufferedReader().readLines()
            .map { it.trim() }
            .filter { it.isNotEmpty() }
}

object TopK {
    fun topK(probs: FloatArray, labels: List<String>, k: Int): List<Pair<String, Float>> {
        return probs.withIndex()
            .sortedByDescending { it.value }
            .take(k)
            .map { idx ->
                val label = labels.getOrNull(idx.index) ?: "class_${idx.index}"
                label to idx.value
            }
    }
}
