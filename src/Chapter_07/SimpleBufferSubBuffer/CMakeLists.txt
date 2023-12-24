add_executable(simpleBufferSubBuffer simple.cpp)
target_link_libraries(simpleBufferSubBuffer PRIVATE OpenCL::OpenCL)

configure_file(simple.cl ${CMAKE_CURRENT_BINARY_DIR}/simple.cl COPYONLY)
