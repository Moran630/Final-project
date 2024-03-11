#ifndef _ITKFUNC_H_
#define _ITKFUNC_H_
#include<itkNiftiImageIO.h>
#include<itkImageFileReader.h>
#include<itkRegionOfInterestImageFilter.h>
#include<itkIdentityTransform.h>
#include<itkResampleImageFilter.h>
#include<itkLinearInterpolateImageFunction.h>

#include <vector>


typedef itk::Image<float, 3> ThreeDFloatImageType;
using ImageType = itk::Image<float, 3>;
using ImageIOType = itk::NiftiImageIO;
using ReaderType = itk::ImageFileReader<ImageType>;

ThreeDFloatImageType::Pointer read_nii_image(const std::string file_path);
void crop_image_by_sp_ep(ThreeDFloatImageType::Pointer intputImage, ThreeDFloatImageType::Pointer outputImage, std::vector<int>& start_point, std::vector<int>& end_point);
void ResampleImageToFixedSize(ThreeDFloatImageType::Pointer inputImage,ThreeDFloatImageType::Pointer outputImage, ThreeDFloatImageType::SizeType new_size, std::vector<double>& new_spacing);

template<typename TInput>
float norm(TInput value, float dbWinMin, float dbWinMax) {
    float tempValue = static_cast<float>((value - dbWinMin) / (dbWinMax - dbWinMin));
    if (std::isless(tempValue, 0))
        return -1.0;
    else if (std::isgreater(tempValue, 1))
        return 1.0;
    else {
        return (tempValue - 0.5) * 2;
    }
}

template<typename TImage>
static void InitImage(TImage* image, std::vector<int> target_shape, double fillval = -150.0) {
    typename TImage::IndexType startRH;
    startRH.Fill(0);
    typename TImage::SizeType sizeRH;
    for (int temp_idx = 0; temp_idx < target_shape.size(); temp_idx++) {
        sizeRH[temp_idx] = target_shape[temp_idx];
    }
    typename TImage::RegionType regionRH;
    regionRH.SetSize(sizeRH);
    regionRH.SetIndex(startRH);

    image->SetRegions(regionRH);
    image->Allocate();
    image->FillBuffer(fillval);
}

#endif