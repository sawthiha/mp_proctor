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
// Calculator to annotate proctoring results
#include <vector>
#include <map>
#include <sstream>
#include <string>
#include <iomanip> // For setting precision

#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/color.pb.h"
#include "mediapipe/util/render_data.pb.h"
#include "mp_proctor/calculators/util/proctor_result.h"

namespace mediapipe
{

    namespace
    {
        constexpr char kResultStreamTag[]  = "RESULT";
        constexpr char kRenderDataStreamTag[] = "RENDER";
    } // namespace

    /**
     * @brief Annotate Detected Eye Blink
     * 
     * INPUTS:
     *      RESULT - Proctor Result (ProctorResult)
     * OUTPUTS:
     *      RENDER - Render Data to be render by OverlayRenderer (RenderData)
     * 
     * Example:
     * 
     * node {
     *   calculator: "ProctorResultToRenderDataCalculator"
     *   input_stream: "RESULT:proctor_result"
     *   output_stream: "RENDER:result_render_data"
     * }
     * 
     */
    class ProctorResultToRenderDataCalculator: public CalculatorBase
    {
    private:
        void AnnotateBlink(RenderData& render_data, bool is_blinking, double left_pos);
        void AnnotateOrientation(RenderData& render_data, std::string orientation, double left_pos);
        void AnnotateExpressions(RenderData& render_data, FacialExpression expressions[8]);

    public:
        ProctorResultToRenderDataCalculator() = default;
        ~ProctorResultToRenderDataCalculator() override = default;

        static absl::Status GetContract(CalculatorContract* cc);

        absl::Status Open(CalculatorContext* cc) override;
        absl::Status Process(CalculatorContext* cc) override;
        absl::Status Close(CalculatorContext* cc) override;

    };
    // Register the calculator to be used in the graph
    REGISTER_CALCULATOR(ProctorResultToRenderDataCalculator);

    absl::Status ProctorResultToRenderDataCalculator::GetContract(CalculatorContract* cc)
    {
        cc->Inputs().Tag(kResultStreamTag).Set<ProctorResult>();
        cc->Outputs().Tag(kRenderDataStreamTag).Set<RenderData>();
        return absl::OkStatus();
    }

    absl::Status ProctorResultToRenderDataCalculator::Open(CalculatorContext* cc)
    { return absl::OkStatus(); }


    void ProctorResultToRenderDataCalculator::AnnotateBlink(RenderData& render_data, bool is_blinking, double left_pos)
    {
        std::string blink = is_blinking ? "Blink": "";
        auto annotation = render_data.add_render_annotations();
        if(blink == "")
        {
            annotation->mutable_color()->set_r(0);
            annotation->mutable_color()->set_g(255);
            annotation->mutable_color()->set_b(0);
        }else
        {
            annotation->mutable_color()->set_r(255);
            annotation->mutable_color()->set_g(0);
            annotation->mutable_color()->set_b(0);
        }
        annotation->set_thickness(3);
        auto text = annotation->mutable_text();
        text->set_font_height(0.03);
        text->set_font_face(0);
        text->set_display_text(blink);
        text->set_normalized(true);
        text->set_left(left_pos);
        text->set_baseline(0.25);
    } // AnnotateBlink

    void ProctorResultToRenderDataCalculator::AnnotateOrientation(RenderData& render_data, std::string orientation, double left_pos)
    {
        auto annotation = render_data.add_render_annotations();
        if(orientation == "Neutral")
        {
            annotation->mutable_color()->set_r(0);
            annotation->mutable_color()->set_g(255);
            annotation->mutable_color()->set_b(0);
        }else
        {
            annotation->mutable_color()->set_r(255);
            annotation->mutable_color()->set_g(0);
            annotation->mutable_color()->set_b(0);
        }
        annotation->set_thickness(4);
        auto text = annotation->mutable_text();
        text->set_font_height(0.04);
        text->set_font_face(0);
        text->set_display_text(orientation);
        text->set_normalized(true);
        text->set_left(left_pos);
        // Normalized coordinates must be between 0.0 and 1.0, if they are used.
        text->set_baseline(0.2);
    }

    void ProctorResultToRenderDataCalculator::AnnotateExpressions(
        RenderData& render_data,
        FacialExpression expressions[8]
        // const std::vector<std::pair<std::string,float>>& expressions
    )
    {
        for (size_t i = 0; i < 3; i++)
        {
            std::stringstream ss;
            switch (expressions[i].type)
            {
            case FacialExpressionType::neutral:
                ss << "Neutral";
                break;
            case FacialExpressionType::happy:
                ss << "Happy";
                break;
            case FacialExpressionType::sad:
                ss << "Sad";
                break;
            case FacialExpressionType::surprise:
                ss << "Surprise";
                break;
            case FacialExpressionType::fear:
                ss << "Fear";
                break;
            case FacialExpressionType::anger:
                ss << "Anger";
                break;
            case FacialExpressionType::disgust:
                ss << "Disgust";
                break;
            case FacialExpressionType::contempt:
                ss << "Contempt";
                break;
            
            default:
                break;
            }
            ss << ": " << std::fixed << std::setprecision(2) << (expressions[i].probability * 100.0);
            auto annotation = render_data.add_render_annotations();
            if(i == 0)
            {
                annotation->mutable_color()->set_r(0);
                annotation->mutable_color()->set_g(255);
                annotation->mutable_color()->set_b(0);
            }
            
            annotation->set_thickness(4);
            auto text = annotation->mutable_text();
            text->set_font_height(0.02);
            text->set_font_face(0);
            text->set_display_text(ss.str());
            text->set_normalized(true);
            text->set_left(0.05);
            // Normalized coordinates must be between 0.0 and 1.0, if they are used.
            text->set_baseline(0.3 + (0.1 * i) );
        }
    }

    absl::Status ProctorResultToRenderDataCalculator::Process(CalculatorContext* cc)
    {
        RenderData render_data;
        if (cc->Inputs().Tag(kResultStreamTag).IsEmpty()) { return absl::OkStatus(); }

        auto result = cc->Inputs().Tag(kResultStreamTag).Get<ProctorResult>();
        
        this->AnnotateBlink(render_data, result.is_left_eye_blinking, 0.08);
        this->AnnotateBlink(render_data, result.is_right_eye_blinking, 0.64);

        std::string hor_align = result.horizontal_align >= 0.3 ? "Right":
                                result.horizontal_align <= -0.3 ? "Left":
                                "Neutral";
        std::string ver_align = result.vertical_align >= 0.6 ? "Down":
                                result.vertical_align <= -0.05 ? "Up":
                                "Neutral";
        this->AnnotateOrientation(render_data, hor_align, 0.05);
        this->AnnotateOrientation(render_data, ver_align, 0.6);

        this->AnnotateExpressions(render_data, result.expressions);
        
        Packet packet = MakePacket<decltype(render_data)>(render_data).At(cc->InputTimestamp());
        cc->Outputs().Tag(kRenderDataStreamTag).AddPacket(packet);

        return absl::OkStatus();
    } // Process()

    absl::Status ProctorResultToRenderDataCalculator::Close(CalculatorContext* cc)
    { return absl::OkStatus(); }

} // namespace mediapipe
