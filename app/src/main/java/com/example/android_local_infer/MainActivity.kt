package com.example.android_local_infer

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.example.android_local_infer.server.LocalServer

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val tv = TextView(this).apply {
            text = "LocalInfer running on ${LocalServer.currentDelegate.toString()}\nhttp://127.0.0.1:8080"

            textSize = 18f
            setPadding(32, 48, 32, 48)
        }
        setContentView(tv)
    }
}
