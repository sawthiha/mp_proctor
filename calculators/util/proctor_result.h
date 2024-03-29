// Copyright 2019 The Authors (https://github.com/sawthiha/mp_proctor/blob/master/AUTHORS).
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Proctor Result structure
#ifndef proctor_result_h
#define proctor_result_h 

enum FacialExpressionType
{
    neutral,
    happy,
    sad,
    surprise,
    fear,
    anger,
    disgust,
    contempt
};

struct FacialExpression
{
    enum FacialExpressionType type;
    float probability;
};

struct ProctorResult
{
    bool is_left_eye_blinking;
    bool is_right_eye_blinking;
    double horizontal_align;
    double vertical_align;
    double facial_activity;
    double face_movement;
    float face_reid_embeddings[128];
    struct FacialExpression expressions[8];
};

#endif