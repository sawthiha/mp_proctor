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
// Calculator to aggregate proctoring results
#include <vector>
#include <algorithm>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/calculators/core/end_loop_calculator.h"
#include "mediapipe/calculators/core/begin_loop_calculator.h"
#include "proctor_result.h"
// #include "mediapipe/framework/port/opencv_core_inc.h"
// #include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/classification.pb.h"

namespace mediapipe
{
    /**
     * @brief Proctor Result Calculator
     * 
     * OUTPUTS:
     *      RESULT - Proctoring Result <ProctorResult>
     * 
     * Example:
     * 
     * # Proctor Result Calculator
     *  node  {
     *      calculator: "ProctorResultCalculator"
     *      input_stream: "ORIENT:face_orientations"
     *      input_stream: "BLINK:face_blinks"
     *      input_stream: "ACTIVE:face_activity"
     *      input_stream: "MOVE:face_movement"
     *      output_stream: "RESULT:result"
     * 
     *  }
     * 
     */
    class ProctorResultCalculator: public CalculatorBase
    {

    public:
        ProctorResultCalculator() = default;
        ~ProctorResultCalculator() override = default;

        static absl::Status GetContract(CalculatorContract* cc);

        absl::Status Open(CalculatorContext* cc) override;
        absl::Status Process(CalculatorContext* cc) override;
        absl::Status Close(CalculatorContext* cc) override;
    };

    // Register the calculator to be used in the graph
    REGISTER_CALCULATOR(ProctorResultCalculator);

    absl::Status ProctorResultCalculator::GetContract(CalculatorContract* cc)
    {
        cc->Inputs().Tag("ORIENT").Set<std::map<std::string, double>>();
        cc->Inputs().Tag("BLINK").Set<std::map<std::string, double>>();
        cc->Inputs().Tag("ACTIVE").Set<double>();
        cc->Inputs().Tag("MOVE").Set<double>();
        // cc->Inputs().Tag("ALIGNED").Set<ImageFrame>();
        cc->Inputs().Tag("EMBED").Set<std::vector<float>>();
        cc->Inputs().Tag("EXP").Set<ClassificationList>();
        // cc->Inputs().Tag("EXP").Set<std::vector<float>>();
        cc->Outputs().Tag("RESULT").Set<ProctorResult>();

        return absl::OkStatus();
    }

    absl::Status ProctorResultCalculator::Open(CalculatorContext* cc)
    {
        return absl::OkStatus();
    }

    absl::Status ProctorResultCalculator::Process(CalculatorContext* cc)
    {
        ProctorResult result;
        auto blink = cc->Inputs().Tag("BLINK").Get<std::map<std::string, double>>();
        auto threshold = blink.at("threshold");
        result.is_left_eye_blinking = blink.at("left") < threshold;
        result.is_right_eye_blinking = blink.at("right") < threshold;
        
        auto orientation = cc->Inputs().Tag("ORIENT").Get<std::map<std::string, double>>();
        result.horizontal_align = orientation.at("horizontal_align");
        result.vertical_align   = orientation.at("vertical_align");
            
        result.facial_activity = cc->Inputs().Tag("ACTIVE").Get<double>();
        result.face_movement = cc->Inputs().Tag("MOVE").Get<double>();

        result.face_reid_embeddings = std::move(cc->Inputs().Tag("EMBED").Get<std::vector<float>>());

        auto expressions = cc->Inputs().Tag("EXP").Get<ClassificationList>();
        auto raw_expressions = expressions.mutable_classification();
        std::sort(raw_expressions->begin(), raw_expressions->end(),
              [](const Classification a, const Classification b) {
                return a.score() > b.score();
              });

        for (int i = 0; i < raw_expressions->size(); i++)
        {
            result.expressions.emplace_back(
                raw_expressions->at(i).label(),
                raw_expressions->at(i).score()
            );
        }

        // std::vector<std::pair<std::string, float>>::iterator it;

        // for (it = result.expressions.begin(); it != result.expressions.end(); it++)
        // {
        //     std::cout << it->first    // string (key)
        //             << ':'
        //             << it->second   // string's value 
        //             << std::endl;
        // }
        
        Packet packet = MakePacket<decltype(result)>(result).At(cc->InputTimestamp()); 
        cc->Outputs().Tag("RESULT").AddPacket(packet);

        return absl::OkStatus();
    } // Process()

    absl::Status ProctorResultCalculator::Close(CalculatorContext* cc)
    { return absl::OkStatus(); }

    typedef BeginLoopCalculator<std::vector<ProctorResult>> BeginLoopProctorResultVectorCalculator;
    REGISTER_CALCULATOR(BeginLoopProctorResultVectorCalculator);

    typedef EndLoopCalculator<std::vector<ProctorResult>> EndLoopProctorResultVectorCalculator;
    REGISTER_CALCULATOR(EndLoopProctorResultVectorCalculator);

} // namespace mediapipe

