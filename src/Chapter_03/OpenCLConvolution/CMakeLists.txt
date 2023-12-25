add_executable(convolution Convolution.cpp)
target_link_libraries(convolution PRIVATE OpenCL::OpenCL)
configure_file(Convolution.cl ${CMAKE_CURRENT_BINARY_DIR}/Convolution.cl COPYONLY)
