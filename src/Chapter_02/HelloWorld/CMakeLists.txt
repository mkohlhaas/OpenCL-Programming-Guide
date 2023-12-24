add_executable(helloWorld HelloWorld.cpp)
target_link_libraries(helloWorld PRIVATE OpenCL::OpenCL)

configure_file(HelloWorld.cl ${CMAKE_CURRENT_BINARY_DIR}/HelloWorld.cl COPYONLY)
