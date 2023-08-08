load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_simple_subgraph",
)

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

mediapipe_simple_subgraph(
    name = "face_reid_cpu",
    graph = "face_reid_cpu.pbtxt",
    register_as = "FaceReidentificationCpu",
    deps = [
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/tflite:tflite_custom_op_resolver_calculator",
        "//mediapipe/calculators/image:warp_affine_calculator",
        "//mediapipe/calculators/tflite:tflite_converter_calculator",
        "//mediapipe/calculators/tflite:tflite_inference_calculator",
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mp_proctor/calculators/util:constant_image_size_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/tflite:tflite_tensors_to_floats_calculator",
    ],
)

exports_files(
    srcs = [
        "face_reid.tflite",
    ],
)

