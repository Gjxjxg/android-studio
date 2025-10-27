package com.example.android_local_infer.infer

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.SystemClock
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min

object ImageUtils {
    /**
     * Convert any image bytes to 224x224 float32 NHWC, [0,1] normalized ByteBuffer
     * @return Pair(ByteBuffer, Preprocessing takes time ms)
     */
    fun preprocess(bytes: ByteArray, dstW: Int, dstH: Int): Pair<ByteBuffer, Double> {
        val t0 = SystemClock.elapsedRealtimeNanos()

        val src = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        val cropped = centerCrop(src)
        val bmp = Bitmap.createScaledBitmap(cropped, dstW, dstH, true)

        val input = ByteBuffer.allocateDirect(4 * dstW * dstH * 3)
        input.order(ByteOrder.nativeOrder())

        val pixels = IntArray(dstW * dstH)
        bmp.getPixels(pixels, 0, dstW, 0, 0, dstW, dstH)
        var i = 0
        for (y in 0 until dstH) {
            for (x in 0 until dstW) {
                val c = pixels[i++]
                val r = ((c shr 16) and 0xFF) / 255.0f
                val g = ((c shr 8) and 0xFF) / 255.0f
                val b = (c and 0xFF) / 255.0f
                input.putFloat(r)
                input.putFloat(g)
                input.putFloat(b)
            }
        }
        input.rewind()

        val t1 = SystemClock.elapsedRealtimeNanos()
        return input to ((t1 - t0) / 1e6)
    }

    private fun centerCrop(src: Bitmap): Bitmap {
        val w = src.width; val h = src.height
        if (w == h) return src
        val size = min(w, h)
        val x = (w - size) / 2
        val y = (h - size) / 2
        return Bitmap.createBitmap(src, x, y, size, size)
    }
}
