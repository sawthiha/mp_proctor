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
licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "custom_calculators",
    deps = [
        "//mp_proctor/calculators/util:landmark_standardization",
        "//mp_proctor/calculators/face_orientation:face_orientation_calculator",
        "//mp_proctor/calculators/eye_blink:eye_blink_calculator",
        "//mp_proctor/calculators/face_activity:face_movement_calculator",
        "//mp_proctor/calculators/face_activity:face_activity_calculator",
        "//mp_proctor/calculators/util:proctor_result_calculator",
        "//mp_proctor/calculators/util:proctor_result",
        "//mp_proctor/calculators/util:similarity_transform_calculator",
        "//mp_proctor/calculators/util:face_align",
    ],
)

cc_library(name = "proctor_deps",
    deps = [
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        ":custom_calculators",
        "//mp_proctor/modules/face_reid:face_reid_cpu",
        "//mp_proctor/modules/face_affect:face_affect_cpu",
    ] + select({
        "//mediapipe/gpu:disable_gpu": [
            "//mediapipe/modules/face_landmark:face_landmark_front_cpu",
        ],
        "//conditions:default": [
            "//mediapipe/modules/face_landmark:face_landmark_front_gpu",
        ],
    }),
)

exports_files(
    srcs = glob(
        ["*.pbtxt"]
    )
)

cc_library(
    name = "live_calculators",
    deps = [
        ":proctor_deps",
    ] + select({
        "//mediapipe/gpu:disable_gpu": [
            "//mp_proctor/graphs/subgraphs:face_renderer_cpu",
        ],
        "//conditions:default": [
            "//mp_proctor/graphs/subgraphs:face_renderer_gpu",
        ],
    }),
)

filegroup(name = "proctor_data",
    srcs = [
        "//mediapipe/modules/face_detection:face_detection_short_range.tflite",
        "//mediapipe/modules/face_landmark:face_landmark_with_attention.tflite",
        "//mp_proctor/modules/face_affect:face_affect.tflite",
        "//mp_proctor/modules/face_affect:expressions_map.txt",
        "//mp_proctor/modules/face_reid:face_reid.tflite",
    ],
)
