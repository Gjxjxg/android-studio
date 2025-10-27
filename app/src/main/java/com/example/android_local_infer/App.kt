package com.example.android_local_infer

import android.app.Application
import com.example.android_local_infer.server.LocalServer
import android.util.Log


class App : Application() {
    override fun onCreate() {
        super.onCreate()
        // Start the local HTTP service (TensorFlow Lite inference interface)
        Log.i("App", "Application onCreate")
        LocalServer.start(this)
    }
}
