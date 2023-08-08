load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_simple_subgraph",
)

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

mediapipe_simple_subgraph(
    name = "face_affect_cpu",
    graph = "face_affect_cpu.pbtxt",
    register_as = "FaceAffectCpu",
    deps = [
        "//mediapipe/calculators/tflite:tflite_custom_op_resolver_calculator",
        "//mediapipe/calculators/tflite:tflite_inference_calculator",
        "//mediapipe/calculators/tflite:tflite_tensors_to_classification_calculator",
    ],
)

exports_files(
    srcs = [
        "face_affect.tflite",
        "expressions_map.txt",
    ],
)

