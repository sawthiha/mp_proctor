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
    ],
)

cc_library(name = "facemesh_deps",
    deps = [
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/core:packet_cloner_calculator",
        ":custom_calculators"
    ] + select({
        "//mediapipe/gpu:disable_gpu": [
            "//mediapipe/modules/face_landmark:face_landmark_front_cpu",
        ],
        "//conditions:default": [
            "//mediapipe/modules/face_landmark:face_landmark_front_gpu",
        ],
    }),
)

filegroup(name = "minimal_face_mesh_pbtxts",
    srcs = select({
        "//mediapipe/gpu:disable_gpu": [
            "//mp_proctor/graphs/face_mesh/minimal:face_mesh_cpu.pbtxt",
        ],
        "//conditions:default": [
            "//mp_proctor/graphs/face_mesh/minimal:face_mesh_gpu.pbtxt",
        ],
    }),
)

filegroup(name = "face_mesh_data",
    srcs = [
        "//mediapipe/modules/face_detection:face_detection_short_range.tflite",
        "//mediapipe/modules/face_landmark:face_landmark_with_attention.tflite",
    ],
)
