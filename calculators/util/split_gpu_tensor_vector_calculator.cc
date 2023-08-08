#include "mediapipe/calculators/core/split_vector_calculator.h"

#include "mediapipe/util/tflite/config.h"

namespace mediapipe
{
    
typedef SplitVectorCalculator<GpuTensor, false>
    SplitGpuTensorVectorCalculator;
REGISTER_CALCULATOR(SplitGpuTensorVectorCalculator);

} // namespace mediapipe

