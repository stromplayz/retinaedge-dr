# RetinaEdge-DR release keep rules.
# The TFLite AAR ships consumer rules, but keep the JNI surface defensively.
-keep class org.tensorflow.lite.** { *; }
-dontwarn org.tensorflow.lite.gpu.**
-dontwarn org.tensorflow.lite.gpu.delegate.**
