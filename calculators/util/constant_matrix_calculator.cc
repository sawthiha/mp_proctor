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
// Calculator to generate a matrix
#include <array>

#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#include "mp_proctor/calculators/util/constant_matrix_calculator.pb.h"

namespace mediapipe
{
    /**
     * @brief Constant Matrix Stream Calculator
     * 
     * OUTPUTS:
     *      TICK - For synchronization purpose
     * 
     * Example:
     * 
     * # Constant Matrix Calculator
     *  node  {
     *      calculator: "ConstantMatrixCalculator"
     *      input_stream: "TICK:sync_stream"
     *      output_stream: "MATRIX:matrix"
     *      node_options: {
     *          [type.googleapis.com/mediapipe.ConstantMatrixCalculatorOptions] {
     *              # In row-major format
     *              1.0, 0.0, 0.0, 0.0,
     *              0.0, 1.0, 0.0, 0.0,
     *              0.0, 0.0, 1.0, 0.0,
     *              0.0, 0.0, 0.0, 1.0
     *          }
     *      }
     *  }
     * 
     */
    class ConstantMatrixCalculator: public CalculatorBase
    {
    private:
        ConstantMatrixCalculatorOptions m_options;

    public:
        ConstantMatrixCalculator() = default;
        ~ConstantMatrixCalculator() override = default;

        static absl::Status GetContract(CalculatorContract* cc);

        absl::Status Open(CalculatorContext* cc) override;
        absl::Status Process(CalculatorContext* cc) override;
        absl::Status Close(CalculatorContext* cc) override;
    };

    // Register the calculator to be used in the graph
    REGISTER_CALCULATOR(ConstantMatrixCalculator);

    absl::Status ConstantMatrixCalculator::GetContract(CalculatorContract* cc)
    {
        cc->Inputs().Tag("TICK").SetAny();
        cc->Outputs().Tag("MATRIX").Set<std::array<float, 16>>();
        return absl::OkStatus();
    }

    absl::Status ConstantMatrixCalculator::Open(CalculatorContext* cc)
    {
        m_options = cc->Options<ConstantMatrixCalculatorOptions>();
        assert(m_options.values_size() == 16);
        return absl::OkStatus();
    }

    absl::Status ConstantMatrixCalculator::Process(CalculatorContext* cc)
    {
        auto values = m_options.values().data();
        auto matrix = std::make_unique<std::array<float, 16>>();
        std::copy(values, values + 16, matrix->data());

        cc->Outputs().Tag("MATRIX").Add(matrix.release(), cc->InputTimestamp());

        return absl::OkStatus();
    } // Process()

    absl::Status ConstantMatrixCalculator::Close(CalculatorContext* cc)
    { return absl::OkStatus(); }

} // namespace mediapipe

