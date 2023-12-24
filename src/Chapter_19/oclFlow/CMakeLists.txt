# NV GLUT paths
set(CMAKE_INCLUDE_PATH $ENV{NVSDKCOMPUTE_ROOT}/shared/inc)

find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GLEW REQUIRED)

# Detect 32/64 bit build environment and set glut  and glew libs correctly
if(WIN32)
	STRING(REGEX MATCH "Win64" ISWIN64 ${CMAKE_GENERATOR})
	if("${ISWIN64}" STREQUAL "Win64")
		SET( GLUT_LIBRARIES "freeglut.lib glew64.lib")
		FIND_PATH( GLEW_LIBRARY_PATH glew64.lib PATHS $ENV{NVSDKCOMPUTE_ROOT}/shared/lib/x64)
	else("${ISWIN64}" STREQUAL "Win64")
		SET( GLUT_LIBRARIES "freeglut.lib glew32.lib")
		FIND_PATH( GLEW_LIBRARY_PATH glew32.lib PATHS $ENV{NVSDKCOMPUTE_ROOT}/shared/lib/Win32 )
	endif("${ISWIN64}" STREQUAL "Win64")
endif(WIN32)

# find glew include
find_path( GLEW_INCLUDE_PATH GL/glew.h PATHS $ENV{NVSDKCOMPUTE_ROOT}/shared/inc )
find_path( OCLUTILS_INCLUDE_PATH oclUtils.h PATHS $ENV{NVSDKCOMPUTE_ROOT}/OpenCL/common/inc )
find_path( SHRUTILS_INCLUDE_PATH shrUtils.h PATHS $ENV{NVSDKCOMPUTE_ROOT}/shared/inc )
if(WIN32)
    find_library( OCLUTILS_LIBRARIES oclUtils32.lib PATHS $ENV{NVSDKCOMPUTE_ROOT}/OpenCL/common/lib/Win32 )
else(WIN32)
    #todo : this needs to be arch specific 32/64 on Linux
    find_library( OCLUTILS_LIBRARIES liboclUtil_x86_64.a PATHS $ENV{NVSDKCOMPUTE_ROOT}/OpenCL/common/lib/ )
    find_library( SHRUTILS_LIBRARIES libshrutil_x86_64.a PATHS $ENV{NVSDKCOMPUTE_ROOT}/shared/lib/ )
endif(WIN32)


message("${GLEW_INCLUDE_PATH}")

#We statically link because OpenCV does this by default so to link there we use the same
# comment this out if all other libraries use dynamic linking
if(MSVC)
foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
    endif(${flag_var} MATCHES "/MD")
    if(${flag_var} MATCHES "/MDd")
        string(REGEX REPLACE "/MDd" "/MTd" ${flag_var} "${${flag_var}}")
    endif(${flag_var} MATCHES "/MDd")
endforeach(flag_var)
endif(MSVC)

# copy the .cl kernel files and the data files
configure_file(filters.cl ${CMAKE_CURRENT_BINARY_DIR}/filters.cl COPYONLY)
configure_file(motion.cl ${CMAKE_CURRENT_BINARY_DIR}/motion.cl COPYONLY)
configure_file(lkflow.cl ${CMAKE_CURRENT_BINARY_DIR}/lkflow.cl COPYONLY)
configure_file(data/minicooper/frame10.pgm ${CMAKE_CURRENT_BINARY_DIR}/data/minicooper/frame10.pgm COPYONLY)
configure_file(data/minicooper/frame11.pgm ${CMAKE_CURRENT_BINARY_DIR}/data/minicooper/frame11.pgm COPYONLY)
configure_file(readme.txt ${CMAKE_CURRENT_BINARY_DIR}/readme.txt COPYONLY)


# todo: Linux opencv ubuntu distro req's cudart at compile time, add library "cudart" and path to toolkit below to compile on Linux

link_directories(${CMAKE_LIBRARY_PATH} )
add_executable( oclFlow oclFlow.cpp flowGL.cpp)
include_directories( ${GLUT_INCLUDE_DIR} ${OPENGL_INCLUDE_DIR} ${OCLUTILS_INCLUDE_PATH} ${SHRUTILS_INCLUDE_PATH} )
target_link_libraries( oclFlow ${OPENCL_LIBRARIES} ${GLUT_LIBRARIES} ${OpenCV_LIBS} ${OCLUTILS_LIBRARIES} ${SHRUTILS_LIBRARIES} ${GLEW_LIBRARY})
