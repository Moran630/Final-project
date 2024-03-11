#include "torchFunc.h"
#include "itkFunc.h"


int main(int argc, char **argv) {
    const std::string pt_file = argv[1];
    const std::string file_path = argv[2];

    std::cout << "Loading file: "<< file_path << std::endl;
    
    // load nii.gz file
    ThreeDFloatImageType::Pointer rawImage = read_nii_image(file_path);

    // using ImageType = itk::Image<float, 3>;
	ImageType::SpacingType rawSpacing = rawImage->GetSpacing();
    ImageType::SizeType rawSize = rawImage->GetLargestPossibleRegion().GetSize();
	std::cout << "rawSpacing:" << rawSpacing << std::endl;
    std::cout << "rawSize:" << rawSize << std::endl;

    ThreeDFloatImageType::Pointer cropImage = ThreeDFloatImageType::New();
    std::vector<int> start_point;
    std::vector<int> end_point;
    std::vector<float> ratio_start = {0.15, 0.2, 0.15};
    std::vector<float> ratio_end = {0.85, 0.7, 0.9};

    for (int i = 0; i < 3; i++) {
        int size = rawSize[i];
        float s = size * ratio_start[i];
        float e = size * ratio_end[i];
        start_point.push_back(static_cast<int>(s));
        end_point.push_back(static_cast<int>(e));
    }
    std::cout << "start point: ";
    for (auto ss : start_point)
        std::cout << ss << ", ";

    std::cout << "\n";
    std::cout << "end point: ";
    for (auto ss : end_point)
        std::cout << ss << ", ";
    std::cout << "\n";
    crop_image_by_sp_ep(rawImage.GetPointer(), cropImage.GetPointer(), start_point, end_point);
    ImageType::SpacingType cropSpacing = cropImage->GetSpacing();
    ImageType::SizeType cropSize = cropImage->GetLargestPossibleRegion().GetSize();
	std::cout << "cropSpacing:" << cropSpacing << std::endl;
    std::cout << "cropSize:" << cropSize << std::endl;

    ThreeDFloatImageType::Pointer resizeImage = ThreeDFloatImageType::New();
    std::vector<double> newSpacing;

    int max_resize = 144;
    int max_size = std::max({int(cropSize[0]), int(cropSize[1]), int(cropSize[2])});
    double scale = float(max_resize) / max_size;

    ThreeDFloatImageType::SizeType newSize;
     for (int i = 0; i < 3; i++){
        newSize[i] = static_cast<int>(cropSize[i] * scale);
    }

    for (int i = 0; i < 3; i++){
        newSpacing.push_back(cropSize[i] * cropSpacing[i] / newSize[i]);
    }
    std::cout << "newSpacing: ";
    for (auto ss : newSpacing)
        std::cout << ss << ", "; 
    std::cout << "\n";
    ResampleImageToFixedSize(cropImage.GetPointer(), resizeImage.GetPointer(), newSize, newSpacing);
    ImageType::SpacingType resizeSpacing = resizeImage->GetSpacing();
    ImageType::SizeType resizeSize = resizeImage->GetLargestPossibleRegion().GetSize();
    std::cout << "resizeSpacing:" << resizeSpacing << std::endl;
    std::cout << "resizeSize:" << resizeSize << std::endl;

    ThreeDFloatImageType::Pointer resizePadImage = ThreeDFloatImageType::New();
    std::vector<int> target_shape = {144, 144, 144};
    InitImage(resizePadImage.GetPointer(), target_shape);

    ThreeDFloatImageType::IndexType patch_index;
    for (int x_index = 0; x_index < resizeSize[0]; x_index++) {
        for (int y_index = 0; y_index < resizeSize[1]; y_index++) {
            for (int z_index = 0; z_index < resizeSize[2]; z_index++) {
                patch_index[0] = x_index, patch_index[1] = y_index, patch_index[2] = z_index;
                double pixel_val = resizeImage->GetPixel(patch_index);
                resizePadImage->SetPixel(patch_index, pixel_val);
            }
        }
    }

    ImageType::SpacingType resizepadSpacing = resizePadImage->GetSpacing();
    ImageType::SizeType resizePadSize = resizePadImage->GetLargestPossibleRegion().GetSize();
    std::cout << "resizepadSpacing:" << resizepadSpacing << std::endl;
    std::cout << "resizePadSize:" << resizePadSize << std::endl;

    torch::Tensor inputs = transferItkToTensor(resizePadImage.GetPointer());
    std::map<std::string, std::vector<float>> outputs_map = model_infer(pt_file, inputs);

    // for (auto it = outputs_map.begin(); it != outputs_map.end(); ++it) {
    //     std::cout << "label: " << it->first << std::endl;
    //     std::vector probs = it->second;
    //     for (auto prob : probs)
    //         std::cout << prob << ", ";
    //     std::cout << "\n";
    // }

    std::map<std::string, std::map<int, std::string>> pred_label_to_string = {
        {"bowel", {{0, "healthy"}, {1, "injury"}}},
        {"extravasation", {{0, "healthy"}, {1, "injury"}}},
        {"kidney", {{0, "healthy"}, {1, "low injury"}, {2, "high injury"}}},
        {"liver", {{0, "healthy"}, {1, "low injury"}, {2, "high injury"}}},
        {"spleen", {{0, "healthy"}, {1, "low injury"}, {2, "high injury"}}},
    };

    for (auto it = outputs_map.begin(); it != outputs_map.end(); ++it) {
        // std::cout << "label: " << it->first << std::endl;
        std::vector<float> probs = it->second;
        float max_prob = *std::max_element(probs.begin(), probs.end());
        int pred_label = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
        std::string type = pred_label_to_string[it->first][pred_label];
        std::cout << it->first << ": " << type << ", prob: " << max_prob << std::endl;
    }
    return 0;

}