#include "torchFunc.h"

torch::Tensor transferItkToTensor(ThreeDFloatImageType::Pointer InputImage, std::vector<int> size) {
    // std::shared_ptr<float> ptr(new float[size[0] * size[1] * size[2]], std::default_delete<float[]>());
    // float* ptrData = ptr.get();
    float* ptrData = new float[size[0] * size[1] * size[2]];
    memset(ptrData, -1.0, sizeof(float) * size[0] * size[1] * size[2]);

    using IteratorType = itk::ImageRegionConstIterator<ImageType>;
    IteratorType it(InputImage, InputImage->GetLargestPossibleRegion());
    size_t ptrData_index(0);
    it.GoToBegin();
    while (!it.IsAtEnd()) {
        ptrData[ptrData_index] = static_cast<float>(norm(it.Get(), -150, 250));
        ++it;
        ++ptrData_index;
    }
    torch::Tensor dataTensor = torch::from_blob(ptrData, {int(size[2]), int(size[1]), int(size[0])}).toType(torch::kFloat32).to(torch::Device(torch::kCUDA));
    dataTensor = dataTensor.unsqueeze(0).unsqueeze(0);
    delete[] ptrData;
    return dataTensor;
}

std::map<std::string, std::vector<float>> model_infer(const std::string pt_path, torch::Tensor& input_tensor) {
    torch::jit::script::Module model = torch::jit::load(pt_path);
    auto result = model.forward({input_tensor}).toTensor();

    c10::IntArrayRef tsize = result.sizes();
    int n = tsize[0]; // n = 1
    int c = tsize[1]; // c = 13
    // std::cout << "n: " << n << ", c: " << c << std::endl;
    std::vector<float> m_vecResult;
    for (int i = 0; i < c; i++) {
        float val = result[0][i].item<float>();
        // std::cout << val << std::endl;
        m_vecResult.push_back(val);
    }
    
    std::map<std::string, std::vector<float>> results;

    std::vector<float> bowel_probs(m_vecResult.begin(), m_vecResult.begin() + 2);
    std::vector<float> extra_probs(m_vecResult.begin() + 2, m_vecResult.begin() + 4);
    std::vector<float> kidney_probs(m_vecResult.begin() + 4, m_vecResult.begin() + 7);
    std::vector<float> liver_probs(m_vecResult.begin() + 7, m_vecResult.begin() + 10); 
    std::vector<float> spleen_probs(m_vecResult.begin() + 10, m_vecResult.begin() + 13);

    results["bowel"] = bowel_probs;
    results["extravasation"] = extra_probs;
    results["kidney"] = kidney_probs;
    results["liver"] = liver_probs;
    results["spleen"] = spleen_probs;
    return results;
}

