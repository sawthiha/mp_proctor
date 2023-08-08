#include <vector>

#include "mediapipe/util/tflite/config.h"

namespace mediapipe
{

/**
 * @brief Calculator to concert GpuTensor to floats
 * 
 * INPUTS:
 *      TENSORS - GpuTensor
 * OUTPUTS:
 *      FLOATS - std::vector<float>
 * 
 * Example:
 * 
 * node {
 *   calculator: "GpuTensorsToFloatsCalculator"
 *   input_stream: "TENSORS:input_tensor"
 *   output_stream: "FLOATS:output_floats"
 * }
 * 
 */

constexpr char kInputTag[] = "TENSORS";

constexpr char kOutputTag[] = "FLOATS";

class GpuTensorsToFloatsCalculator: public CalculatorBase
{
private:

public:
    GpuTensorsToFloatsCalculator() = default;
    ~GpuTensorsToFloatsCalculator() override = default;

    static absl::Status GetContract(CalculatorContract* cc);

    absl::Status Open(CalculatorContext* cc) override;
    absl::Status Process(CalculatorContext* cc) override;
    absl::Status Close(CalculatorContext* cc) override;
};

// Register the calculator to be used in the graph
REGISTER_CALCULATOR(GpuTensorsToFloatsCalculator);

absl::Status GpuTensorsToFloatsCalculator::GetContract(CalculatorContract* cc)
{
    cc->Inputs().Tag(kInputTag).Set<std::vector<GpuTensor>>();
    cc->Outputs().Tag(kOutputTag).Set<std::vector<float>>();

    #if MEDIAPIPE_TFLITE_METAL_INFERENCE
      return absl::OkStatus();
    #else
      return absl::FailedPreconditionError("Only Metal Inference supported for now!");
    #endif
}

absl::Status GpuTensorsToFloatsCalculator::Open(CalculatorContext* cc)
{
  return absl::OkStatus();
}

absl::Status GpuTensorsToFloatsCalculator::Process(CalculatorContext* cc)
{
    if(cc->Inputs().Tag(kInputTag).IsEmpty())
    {
      return absl::FailedPreconditionError("GpuTensorsToFloatsCalculator: Tensors packet is empty!");
    }
    
    const auto& tensors = cc->Inputs().Tag(kInputTag).Get<std::vector<GpuTensor>>();
    if(tensors.empty())
    {
      return absl::FailedPreconditionError("GpuTensorsToFloatsCalculator: No Tensors to be converted!");
    }
    auto tensor = tensors[0];

    #if MEDIAPIPE_TFLITE_METAL_INFERENCE

      float* floatPtr = static_cast<float*>([tensor contents]);

      // Calculate the number of elements in the float array
      size_t numFloats = [tensor length] / sizeof(float);

      // Create the float vector using the float* pointer and length
      std::vector<float> floatVector(floatPtr, floatPtr + numFloats);
      
      Packet packet = MakePacket<std::vector<float>>(floatVector)
          .At(cc->InputTimestamp());
      cc->Outputs().Tag(kOutputTag).AddPacket(packet);

      return absl::OkStatus();
    #else
      return absl::FailedPreconditionError("Only Metal Inference supported for now!");
    #endif
} // Process()

absl::Status GpuTensorsToFloatsCalculator::Close(CalculatorContext* cc)
{ return absl::OkStatus(); }

} // namespace mediapipe

