plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.android_local_infer"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.android_local_infer"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
        manifestPlaceholders["appName"] = "LocalInfer"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    // Uncompress .tflite
    androidResources { noCompress += listOf("tflite") }

    packaging {
        resources {
            excludes += setOf("META-INF/{AL2.0,LGPL2.1}")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
}

dependencies {
    // ===== TensorFlow Lite =====
    implementation("org.tensorflow:tensorflow-lite:2.14.0")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.14.0")
    implementation("org.tensorflow:tensorflow-lite-gpu-api:2.14.0")

    //===== Ktor Server 3.x (compatible with coroutines 1.9.x) =====
    val ktor = "3.0.1"
    implementation("io.ktor:ktor-server-core:$ktor")
    implementation("io.ktor:ktor-server-cio:$ktor")
    implementation("io.ktor:ktor-server-content-negotiation:$ktor")
    implementation("io.ktor:ktor-serialization-jackson:$ktor")

    implementation("com.fasterxml.jackson.module:jackson-module-kotlin:2.17.+")
    // Optional: Remove SLF4J prompt (can run without it)
    // implementation("org.slf4j:slf4j-android:1.7.36")

    // ===== Kotlin Coroutines =====
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")

    // ===== AndroidX =====
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.localbroadcastmanager:localbroadcastmanager:1.1.0")


    implementation("org.eclipse.paho:org.eclipse.paho.client.mqttv3:1.2.5")


    // implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")

}
