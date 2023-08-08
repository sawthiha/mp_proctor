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
#include "mp_proctor/calculators/util/constant_image_size_calculator.pb.h"

namespace mediapipe
{
    /**
     * @brief Create image size std::pair<int, int> from constant value
     * 
     * OUTPUTS:
     *      SIZE - Image size (std::pait<int, int>)
     * 
     * Example:
     * 
     * # Constant Image Size Calculator
     *  node  {
     *      calculator: "ConstantImageSizeCalculator"
     *      input_stream: "SYNC:sync_stream"
     *      output_stream: "SIZE:image_size"
     *      node_options: {
     *          [type.googleapis.com/mediapipe.ConstantImageSizeCalculatorOptions] {
     *              width: 112
     *              height: 112
     *          }
     *      }
     *  }
     * 
     */
    class ConstantImageSizeCalculator: public CalculatorBase
    {
    private:
        int64 m_timestamp = 0;
        ConstantImageSizeCalculatorOptions m_options;
    public:
        ConstantImageSizeCalculator() = default;
        ~ConstantImageSizeCalculator() override = default;

        static absl::Status GetContract(CalculatorContract* cc);

        absl::Status Open(CalculatorContext* cc) override;
        absl::Status Process(CalculatorContext* cc) override;
        absl::Status Close(CalculatorContext* cc) override;
    };

    // Register the calculator to be used in the graph
    REGISTER_CALCULATOR(ConstantImageSizeCalculator);

    absl::Status ConstantImageSizeCalculator::GetContract(CalculatorContract* cc)
    {
        cc->Inputs().Tag("TICK").SetAny();
        cc->Outputs().Tag("SIZE").Set<std::pair<int, int>>();
        return absl::OkStatus();
    }

    absl::Status ConstantImageSizeCalculator::Open(CalculatorContext* cc)
    {
        m_options = cc->Options<ConstantImageSizeCalculatorOptions>();
        return absl::OkStatus();
    }

    absl::Status ConstantImageSizeCalculator::Process(CalculatorContext* cc)
    {
      auto size = std::make_pair<int, int>(m_options.width(), m_options.height());
      cc->Outputs().Tag("SIZE").AddPacket(
        MakePacket<decltype(size)>(size).At(cc->InputTimestamp())
      );

      return absl::OkStatus();
    } // Process()

    absl::Status ConstantImageSizeCalculator::Close(CalculatorContext* cc)
    { return absl::OkStatus(); }

} // namespace mediapipe

