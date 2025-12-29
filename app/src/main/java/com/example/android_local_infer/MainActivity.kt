package com.example.android_local_infer

import android.os.Bundle
import android.util.Log
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.android_local_infer.infer.InferenceEngine
import com.example.android_local_infer.infer.ModelCfg
import com.example.android_local_infer.server.LocalServer
import com.example.android_local_infer.exec.ExecutionManager
import com.example.android_local_infer.exec.ExecMode
import com.example.android_local_infer.exec.LocalEngineFacadeImpl
import kotlinx.coroutines.launch
import com.example.android_local_infer.offload.MqttOffloadClient

class MainActivity : AppCompatActivity() {

    private lateinit var execMgr: ExecutionManager
    private lateinit var engine: InferenceEngine

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)


        val tv = TextView(this).apply {
            text = "LocalInfer running\nhttp://127.0.0.1:8080"
            textSize = 18f
            setPadding(32, 48, 32, 48)
        }
        setContentView(tv)


        engine = InferenceEngine(this)
        val localFacade = LocalEngineFacadeImpl(engine, topk = 5)


        val offloadClient = MqttOffloadClient(
            brokerHost = "192.168.2.220",
            brokerPort = 1883
        )



        execMgr = ExecutionManager(offloadClient, localFacade)


        lifecycleScope.launch {
            runDemoSchedule()
        }
    }

    private suspend fun runDemoSchedule() {


        val schedule = listOf(
            ExecMode.CPU,  // mv3
            ExecMode.CPU,
            ExecMode.CPU,
            ExecMode.CPU,
            ExecMode.CPU,
            ExecMode.CPU,
            ExecMode.CPU,
            ExecMode.CPU,
            ExecMode.CPU,
            ExecMode.CPU,
            ExecMode.OFFLOAD,  // mv3
            ExecMode.OFFLOAD,
            ExecMode.OFFLOAD,
            ExecMode.OFFLOAD,
            ExecMode.OFFLOAD,
            ExecMode.OFFLOAD,
            ExecMode.OFFLOAD,
            ExecMode.OFFLOAD,
            ExecMode.OFFLOAD


        )

        val models = listOf(
            ModelCfg.MV3,
            ModelCfg.Eff0,
            ModelCfg.MV3,
            ModelCfg.MV3,
            ModelCfg.MV3,
            ModelCfg.MV3,
            ModelCfg.MV3,
            ModelCfg.MV3,
            ModelCfg.MV3,
            ModelCfg.MV3,
            ModelCfg.MV3,
            ModelCfg.Eff0,
            ModelCfg.MV3,
            ModelCfg.MV3,
            ModelCfg.MV3,
            ModelCfg.MV3,
            ModelCfg.MV3,
            ModelCfg.MV3,
            ModelCfg.MV3,
            

        )

        for (i in schedule.indices) {
            val taskId = i + 1
            val mode = schedule[i]
            val model = models[i]

            Log.i("EXP", "TASK $taskId START mode=$mode model=${model.name}")


            engine.switchModel(model)

            val imageBytes = loadTestImageBytes(taskId)

            if (mode == ExecMode.OFFLOAD) {
                Log.i("EXP", "TASK $taskId OFFLOAD UPLOAD_START")
            }

            val result = execMgr.runTask(mode, imageBytes, model)

            if (mode == ExecMode.OFFLOAD) {
                Log.i("EXP", "TASK $taskId OFFLOAD RESULT_RECV")
            }

            Log.i("EXP", "TASK $taskId END result=$result")
        }

        Log.i("EXP", "ALL_TASKS_DONE")
    }

    private fun loadTestImageBytes(taskId: Int): ByteArray {

        val name = "test${taskId}.jpg"
        return assets.open(name).readBytes()
    }
}
