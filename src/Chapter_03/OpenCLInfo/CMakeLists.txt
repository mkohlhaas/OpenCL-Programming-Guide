add_executable(info OpenCLInfo.cpp)
target_link_libraries(info PRIVATE OpenCL::OpenCL)
