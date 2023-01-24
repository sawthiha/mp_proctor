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
// Calculator to Generate Blank Image
#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/timestamp.h"
#include "mp_proctor/calculators/util/blank_image_calculator.pb.h"

namespace {
// Maps ImageFrame format to OpenCV Mat type.
// See mediapipe...image_format.proto and cv...opencv2/core/hal/interface.h
// for more details on respective formats.
int GetMatType(const mediapipe::ImageFormat::Format format) {
  int type = 0;
  switch (format) {
    case mediapipe::ImageFormat::UNKNOWN:
      // Invalid; Default to uchar.
      type = CV_8U;
      break;
    case mediapipe::ImageFormat::SRGB:
      type = CV_8U;
      break;
    case mediapipe::ImageFormat::SRGBA:
      type = CV_8U;
      break;
    case mediapipe::ImageFormat::GRAY8:
      type = CV_8U;
      break;
    case mediapipe::ImageFormat::GRAY16:
      type = CV_16U;
      break;
    case mediapipe::ImageFormat::YCBCR420P:
      // Invalid; Default to uchar.
      type = CV_8U;
      break;
    case mediapipe::ImageFormat::YCBCR420P10:
      // Invalid; Default to uint16.
      type = CV_16U;
      break;
    case mediapipe::ImageFormat::SRGB48:
      type = CV_16U;
      break;
    case mediapipe::ImageFormat::SRGBA64:
      type = CV_16U;
      break;
    case mediapipe::ImageFormat::VEC32F1:
      type = CV_32F;
      break;
    case mediapipe::ImageFormat::VEC32F2:
      type = CV_32FC2;
      break;
    case mediapipe::ImageFormat::LAB8:
      type = CV_8U;
      break;
    case mediapipe::ImageFormat::SBGRA:
      type = CV_8U;
      break;
    default:
      // Invalid or unknown; Default to uchar.
      type = CV_8U;
      break;
  }
  return type;
}
}

namespace mediapipe
{
    /**
     * @brief Create Blank Image Frame
     * 
     * OUTPUTS:
     *      IMAGE - blank_image ()
     * 
     * Example:
     * 
     * # Blank Image Source Calculator
     *  node  {
     *      calculator: "BlankImageCalculator"
     *      input_stream: "SYNC:sync_stream"
     *      output_stream: "IMAGE:blank_image"
     *      node_options: {
     *          [type.googleapis.com/mediapipe.BlankImageCalculatorOptions] {
     *              color { r: 255 g: 255 b: 255 }
     *              width: 500
     *              height: 500
     *          }
     *      }
     *  }
     * 
     */
    class BlankImageCalculator: public CalculatorBase
    {
    private:
        int64 m_timestamp = 0;
        BlankImageCalculatorOptions m_options;
    public:
        BlankImageCalculator() = default;
        ~BlankImageCalculator() override = default;

        static absl::Status GetContract(CalculatorContract* cc);

        absl::Status Open(CalculatorContext* cc) override;
        absl::Status Process(CalculatorContext* cc) override;
        absl::Status Close(CalculatorContext* cc) override;
    };

    // Register the calculator to be used in the graph
    REGISTER_CALCULATOR(BlankImageCalculator);

    absl::Status BlankImageCalculator::GetContract(CalculatorContract* cc)
    {
        cc->Inputs().Tag("SYNC").SetAny();
        cc->Outputs().Tag("IMAGE").Set<ImageFrame>();
        return absl::OkStatus();
    }

    absl::Status BlankImageCalculator::Open(CalculatorContext* cc)
    {
        m_options = cc->Options<BlankImageCalculatorOptions>();
        return absl::OkStatus();
    }

    absl::Status BlankImageCalculator::Process(CalculatorContext* cc)
    {
        cv::Vec3b color(
            m_options.color().r(),
            m_options.color().g(),
            m_options.color().b()
        );
        int sizes[] = {m_options.width(), m_options.height()};
        const int type =
            CV_MAKETYPE(GetMatType(ImageFormat::SRGB), 3);
        cv::Mat color_mat(2, sizes, type, color);

        auto frame = std::unique_ptr<ImageFrame>(
            new ImageFrame(
                ImageFormat::SRGB,
                m_options.width(), m_options.height()
            )
        );

        auto img_mat = mediapipe::formats::MatView(frame.get());
        color_mat.copyTo(img_mat);

        cc->Outputs().Tag("IMAGE").Add(frame.release(), cc->InputTimestamp());

        return absl::OkStatus();
    } // Process()

    absl::Status BlankImageCalculator::Close(CalculatorContext* cc)
    { return absl::OkStatus(); }

} // namespace mediapipe

