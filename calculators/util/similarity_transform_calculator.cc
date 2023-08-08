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

#include <Eigen/Core>
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/formats/landmark.pb.h"

#include "face_align.h"

namespace mediapipe {

constexpr char kInputImageSizeTag[] = "SIZE";
constexpr char kOutputImageSizeTag[] = "OUTPUT_SIZE";
constexpr char kInputLandmarkTag[] = "LANDMARKS";

constexpr char kOutputTag[] = "TRANSFORM";

/**
 * @brief Calculate the similarity transform to align and crop a face detected by face mesh
 * 
 * INPUTS:
 *      SIZE - Input Image Size
 *      OUTPUT_SIZE - Output Image Size
 *      LANDMARKS - Normalized Landmarks
 * OUTPUTS:
 *      TRANSFORM - Similarity Transform Matrix (std::array<float, 16>, for WarpAffineCalculator)
 * 
 * Example:
 * 
 * node {
 *   calculator: "SimilarityTransformCalculator"
 *   input_stream: "SIZE:image_size"
 *   input_stream: "OUTPUT_SIZE:output_image_size"
 *   input_stream: "LANDMARKS:landmarks"
 *   output_stream: "TRANSFORM:similarity_transform"
 * }
 * 
 */
class SimilarityTransformCalculator: public CalculatorBase
{
private:

public:
    SimilarityTransformCalculator() = default;
    ~SimilarityTransformCalculator() override = default;

    static absl::Status GetContract(CalculatorContract* cc);

    absl::Status Open(CalculatorContext* cc) override;
    absl::Status Process(CalculatorContext* cc) override;
    absl::Status Close(CalculatorContext* cc) override;
};

// Register the calculator to be used in the graph
REGISTER_CALCULATOR(SimilarityTransformCalculator);

absl::Status SimilarityTransformCalculator::GetContract(CalculatorContract* cc)
{
    cc->Inputs().Tag(kInputLandmarkTag).Set<NormalizedLandmarkList>();
    cc->Inputs().Tag(kInputImageSizeTag).Set<std::pair<int, int>>();
    cc->Inputs().Tag(kOutputImageSizeTag).Set<std::pair<int, int>>();
    cc->Outputs().Tag(kOutputTag).Set<std::array<float, 16>>();
    return absl::OkStatus();
}

absl::Status SimilarityTransformCalculator::Open(CalculatorContext* cc)
{
    
    return absl::OkStatus();
}

absl::Status SimilarityTransformCalculator::Process(CalculatorContext* cc)
{
    if(cc->Inputs().Tag(kInputLandmarkTag).IsEmpty())
    {
        return absl::FailedPreconditionError("SimilarityTransformCalculator: Landmark packet is empty!");
    }
    if(cc->Inputs().Tag(kInputImageSizeTag).IsEmpty())
    {
        return absl::FailedPreconditionError("SimilarityTransformCalculator: Image size packet is empty!");
    }
    const auto& landmarks = cc->Inputs().Tag(kInputLandmarkTag).Get<NormalizedLandmarkList>();
    const auto& input_frame_size = cc->Inputs().Tag(kInputImageSizeTag).Get<std::pair<int, int>>();
    const auto& output_frame_size = cc->Inputs().Tag(kOutputImageSizeTag).Get<std::pair<int, int>>();
    
    int width = input_frame_size.first, height = input_frame_size.second;
    
    float facial_points[5][2] = {
        {((landmarks.landmark(469).x() + landmarks.landmark(470).x() + landmarks.landmark(471).x() + landmarks.landmark(472).x()) * width) / 4, ((landmarks.landmark(469).y() + landmarks.landmark(470).y() + landmarks.landmark(471).y() + landmarks.landmark(472).y()) * height) / 4},
        {((landmarks.landmark(474).x() + landmarks.landmark(475).x() + landmarks.landmark(476).x() + landmarks.landmark(477).x()) * width) / 4, ((landmarks.landmark(474).y() + landmarks.landmark(475).y() + landmarks.landmark(476).y() + landmarks.landmark(477).y()) * height) / 4},
        {landmarks.landmark(1).x() * width, landmarks.landmark(1).y() * height},
        {landmarks.landmark(61).x() * width, landmarks.landmark(61).y() * height},
        {landmarks.landmark(291).x() * width, landmarks.landmark(291).y() * height}
    };
    
    cv::Mat facial_transform (5, 2, CV_32F, facial_points);
    float reference_points[5][2] = {
        {38.29459953, 51.69630051}, // left eye
        {73.53179932, 51.50139999}, // right eye
        {56.02519989, 71.73660278}, // nose
        {41.54930115, 92.3655014 }, // left mouth
        {70.72990036, 92.20410156} // right mouth
    };
    cv::Mat reference_transform(5, 2, CV_32F, reference_points);
    auto transform = FacePreprocess::similarTransform(facial_transform, reference_transform)(cv::Rect(0, 0, 3, 2));
    
    cv::Mat transform3D = cv::Mat::eye(4, 4, CV_32F);
    transform3D.at<float>(0, 0) = transform.at<float>(0, 0);
    transform3D.at<float>(0, 1) = transform.at<float>(0, 1);
    transform3D.at<float>(0, 3) = transform.at<float>(0, 2);
    transform3D.at<float>(1, 0) = transform.at<float>(1, 0);
    transform3D.at<float>(1, 1) = transform.at<float>(1, 1);
    transform3D.at<float>(1, 3) = transform.at<float>(1, 2);
    cv::invert(transform3D, transform3D);

    // Fix the output as transform to dst image for WrapAffineCalculator
    float adjust_dst_coordinate_floats[4][4] = {
        {1.0f / output_frame_size.first, 0.0f,               0.0f, 0.0f},
        {0.0f,              1.0f / output_frame_size.second, 0.0f, 0.0f},
        {0.0f,              0.0f,               1.0f, 0.0f},
        {0.0f,              0.0f,               0.0f, 1.0f}
    };
    cv::Mat adjust_dst_coordinate(4, 4, CV_32F, adjust_dst_coordinate_floats);
//    cv::Matx44f adjust_dst_coordinate({
//      1.0f / output_frame_size.first, 0.0f,               0.0f, 0.0f,
//      0.0f,              1.0f / output_frame_size.second, 0.0f, 0.0f,
//      0.0f,              0.0f,               1.0f, 0.0f,
//      0.0f,              0.0f,               0.0f, 1.0f});
    cv::invert(adjust_dst_coordinate, adjust_dst_coordinate);
    float adjust_src_coordinate_floats[4][4] = {
        {1.0f * width, 0.0f,                  0.0f, 0.0f},
        {0.0f,                 1.0f * height, 0.0f, 0.0f},
        {0.0f,                 0.0f,                  1.0f, 0.0f},
        {0.0f,                 0.0f,                  0.0f, 1.0f}
    };
    cv::Mat adjust_src_coordinate(4, 4, CV_32F, adjust_src_coordinate_floats);
//    cv::Matx44f adjust_src_coordinate({
//      1.0f * width, 0.0f,                  0.0f, 0.0f,
//      0.0f,                 1.0f * height, 0.0f, 0.0f,
//      0.0f,                 0.0f,                  1.0f, 0.0f,
//      0.0f,                 0.0f,                  0.0f, 1.0f});
    cv::invert(adjust_src_coordinate, adjust_src_coordinate);
    
    transform3D = adjust_src_coordinate * transform3D * adjust_dst_coordinate;
    
    std::array<float, 16> output;
    for(int row = 0; row < transform3D.rows; row++)
    {
        for (int col = 0; col < transform3D.cols; col++)
        {
            output[(row * 4) + col] = transform3D.at<float>(row, col);
        }
    }

    Packet packet = MakePacket<std::array<float, 16>>(output)
        .At(cc->InputTimestamp());
    cc->Outputs().Tag(kOutputTag).AddPacket(packet);

    return absl::OkStatus();
} // Process()

absl::Status SimilarityTransformCalculator::Close(CalculatorContext* cc)
{ return absl::OkStatus(); }

}
