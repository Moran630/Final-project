#include <torch/torch.h>
#include <torch/script.h>
#include <c10/cuda/CUDAGuard.h>

#include "itkFunc.h"
#include <iostream>
#include <vector>
#include <string>
#include <typeinfo>
#include <chrono>


torch::Tensor transferItkToTensor(ThreeDFloatImageType::Pointer InputImage, std::vector<int> size={144, 144, 144});
std::map<std::string, std::vector<float>> model_infer(const std::string pt_path, torch::Tensor& input_tensor);


