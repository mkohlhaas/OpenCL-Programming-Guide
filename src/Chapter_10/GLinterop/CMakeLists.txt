# NV GLUT paths
set(CMAKE_INCLUDE_PATH $ENV{NVSDKCOMPUTE_ROOT}/shared/inc)

find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

message("${GLEW_INCLUDE_PATH}")
configure_file(GLinterop.cl ${CMAKE_CURRENT_BINARY_DIR}/GLinterop.cl COPYONLY)
link_directories(${CMAKE_LIBRARY_PATH} ${GLEW_LIBRARY_PATH})
add_executable( GLinterop GLinterop.cpp )
include_directories( ${GLUT_INCLUDE_DIR} ${OPENGL_INCLUDE_DIR})
target_link_libraries( GLinterop ${OPENCL_LIBRARIES} ${GLUT_LIBRARIES} ${GLEW_LIBRARY} )
