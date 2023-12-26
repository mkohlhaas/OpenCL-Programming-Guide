add_executable(helloBinaryWorld HelloBinaryWorld.cpp)
target_link_libraries(helloBinaryWorld PRIVATE OpenCL::OpenCL)
configure_file(HelloWorld.cl ${CMAKE_CURRENT_BINARY_DIR}/HelloWorld.cl COPYONLY)
