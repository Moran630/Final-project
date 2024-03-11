cmake_minimum_required(VERSION 3.8)
project(FinalProject)

# 设置ITK的路径
set(ITK_DIR "InsightToolkit-5.4rc01/build")

# 查找ITK包
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
message("ITK_USE_FILE: "${ITK_USE_FILE})

message("ITK_LIBRARIES: "${ITK_LIBRARIES})

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message("TORCH_LIBRARIES: "${TORCH_LIBRARIES})

set(Test_INCLUDES 
    ${CMAKE_CURRENT_SOURCE_DIR}/inc  
)
include_directories(${Test_INCLUDES})

set(ITK_SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/itkFunc.cpp
)
add_library(ItkFunc SHARED ${ITK_SOURCE})
target_link_libraries(ItkFunc ${ITK_LIBRARIES})

set(TORCH_SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/torchFunc.cpp
)
add_library(torchFunc SHARED ${TORCH_SOURCE})
target_link_libraries(torchFunc ${TORCH_LIBRARIES})

# 添加您的源文件
add_executable(Test test.cpp)

# 链接ITK库
target_link_libraries(Test optimized torchFunc ItkFunc)

# target_link_libraries(demo "${TORCH_LIBRARIES}")
set_property(TARGET Test PROPERTY CXX_STANDARD 17)
