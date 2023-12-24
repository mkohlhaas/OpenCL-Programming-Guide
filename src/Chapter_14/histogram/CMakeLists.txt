add_executable(histogram histogram.cpp)
target_link_libraries(histogram PRIVATE OpenCL::OpenCL)

configure_file(histogram_image.cl ${CMAKE_CURRENT_BINARY_DIR}/histogram_image.cl COPYONLY)
