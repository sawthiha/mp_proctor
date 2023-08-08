# Copyright 2022 by The Authors (https://github.com/sawthiha/mp_proctor/blob/master/AUTHORS).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

licenses(["notice"])

package(default_visibility = ["//visibility:private"])

cc_library(name = "landmark_standardization",
    srcs        = ["landmark_standardization.cc"],
    visibility  = ["//visibility:public"],
    deps        = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:detection_cc_proto",
    ],
    alwayslink = 1,
)

cc_library(name = "blank_image_calculator",
    srcs        = ["blank_image_calculator.cc"],
    visibility  = ["//visibility:public"],
    deps        = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/stream_handler:immediate_input_stream_handler",
        ":blank_image_calculator_cc_proto",
    ],
    alwayslink = 1,
)

mediapipe_proto_library(
    name = "blank_image_calculator_proto",
    srcs = ["blank_image_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/util:color_proto",
    ],
)

cc_library(name = "constant_image_size_calculator",
    srcs        = ["constant_image_size_calculator.cc"],
    visibility  = ["//visibility:public"],
    deps        = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/stream_handler:immediate_input_stream_handler",
        ":constant_image_size_calculator_cc_proto",
    ],
    alwayslink = 1,
)

mediapipe_proto_library(
    name = "constant_image_size_calculator_proto",
    srcs = ["constant_image_size_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

exports_files(
    srcs = [
        "proctor_result.h",
    ]
)

cc_library(name = "proctor_result",
    hdrs        = ["proctor_result.h"],
    include_prefix = ".",
    visibility  = ["//visibility:public"],
)

cc_library(name = "face_align",
    hdrs        = ["face_align.h"],
    include_prefix = ".",
    visibility  = ["//visibility:public"],
)

cc_library(name = "similarity_transform_calculator",
    srcs        = ["similarity_transform_calculator.cc"],
    visibility  = ["//visibility:public"],
    deps        = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "@eigen_archive//:eigen3",
        ":face_align",
    ],
    alwayslink = 1,
)

cc_library(name = "proctor_result_calculator",
    srcs        = ["proctor_result_calculator.cc"],
    visibility  = ["//visibility:public"],
    deps        = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework:timestamp",
        ":proctor_result",
        "//mediapipe/calculators/core:end_loop_calculator",
        "//mediapipe/calculators/core:begin_loop_calculator",
    ],
    alwayslink = 1,
)

cc_library(name = "constant_matrix_calculator",
    srcs        = ["constant_matrix_calculator.cc"],
    visibility  = ["//visibility:public"],
    deps        = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:status",
        ":constant_matrix_calculator_cc_proto",
    ],
    alwayslink = 1,
)

mediapipe_proto_library(
    name = "constant_matrix_calculator_proto",
    srcs = ["constant_matrix_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/util:color_proto",
    ],
)

cc_library(name = "proctor_result_to_render_data_calculator",
    srcs = ["proctor_result_to_render_data_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/util:render_data_cc_proto",
        ":proctor_result",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

cc_library(name = "split_gpu_tensor_vector_calculator",
    srcs        = ["split_gpu_tensor_vector_calculator.cc"],
    visibility  = ["//visibility:public"],
    copts = select({
        "//mediapipe:ios": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": [],
    }),
    deps        = [
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/util/tflite:config"
    ],
    alwayslink = 1,
)

cc_library(name = "gpu_tensors_to_floats_calculator",
    srcs        = ["gpu_tensors_to_floats_calculator.cc"],
    visibility  = ["//visibility:public"],
    copts = select({
        "//mediapipe:ios": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": [],
    }),
    deps        = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:status",
        "//mediapipe/util/tflite:config",
    ],
    alwayslink = 1,
)

