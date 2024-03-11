#include"itkFunc.h"


ThreeDFloatImageType::Pointer read_nii_image(const std::string file_path) {
	using ImageType = itk::Image<float, 3>;
	using ImageIOType = itk::NiftiImageIO;
	using ReaderType = itk::ImageFileReader<ImageType>;

	ReaderType::Pointer reader = ReaderType::New(); // 定义reader
	ImageIOType::Pointer niftiIO = ImageIOType::New(); // 定义文件IO
	reader->SetImageIO(niftiIO); // 设置reader的文件IO

	reader->SetFileName(file_path);
	reader->Update(); // 读取文件结束

	return reader->GetOutput();
}


void crop_image_by_sp_ep(ThreeDFloatImageType::Pointer intputImage, ThreeDFloatImageType::Pointer outputImage, std::vector<int>& start_point, std::vector<int>& end_point) {
    typename ThreeDFloatImageType::RegionType desireRegion;
    typename ThreeDFloatImageType::IndexType imageStart;
    typename ThreeDFloatImageType::SizeType imageSize;
    imageStart[0] = start_point[0];
    imageStart[1] = start_point[1];
    imageStart[2] = start_point[2];
    imageSize[0] = end_point[0] - start_point[0];
    imageSize[1] = end_point[1] - start_point[1];
    imageSize[2] = end_point[2] - start_point[2];

    desireRegion.SetIndex(imageStart);
    desireRegion.SetSize(imageSize);
    typedef itk::RegionOfInterestImageFilter<ThreeDFloatImageType, ThreeDFloatImageType> ROIFilterType;
    typename ROIFilterType::Pointer roi_filter = ROIFilterType::New();
    roi_filter->SetInput(intputImage);
    roi_filter->SetRegionOfInterest(desireRegion);
    roi_filter->Update();
    outputImage->Graft(roi_filter->GetOutput());
    outputImage->SetSpacing(intputImage->GetSpacing());
}

void ResampleImageToFixedSize(ThreeDFloatImageType::Pointer inputImage,ThreeDFloatImageType::Pointer outputImage, ThreeDFloatImageType::SizeType new_size, std::vector<double>& new_spacing) {
    ThreeDFloatImageType::SpacingType spacing;

    for (int i = 0; i < 3; ++i) {
        spacing[i] = new_spacing[i];
    }


    typedef itk::IdentityTransform<double, 3> TransformType;
    typedef itk::ResampleImageFilter<ThreeDFloatImageType, ThreeDFloatImageType> ResampleImageFilterType;
    ResampleImageFilterType::Pointer resample = ResampleImageFilterType::New();
    typedef itk::LinearInterpolateImageFunction<ThreeDFloatImageType, double > InterpolatorType;
    InterpolatorType::Pointer interpolator = InterpolatorType::New();

    resample->SetInterpolator(interpolator);
    resample->SetInput(inputImage);
    resample->SetOutputDirection(inputImage->GetDirection());
    resample->SetOutputOrigin(inputImage->GetOrigin());
    resample->SetSize(new_size);
    resample->SetOutputSpacing(spacing);
    resample->SetTransform(TransformType::New());

    resample->UpdateLargestPossibleRegion();
    ThreeDFloatImageType::Pointer resampleImgData;
    resampleImgData = resample->GetOutput();

    outputImage->Graft(resampleImgData);
}