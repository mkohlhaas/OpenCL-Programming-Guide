# Find package module for FreeImage library.
#
# The following variables are set by this module:
#
# FREEIMAGE_FOUND: TRUE if FreeImage is found.
# freeimage::FreeImage: Imported target to link against.
#
# The following variables control the behavior of this module:
#
# FREEIMAGE_INCLUDE_DIR_HINTS: List of additional directories in which to
#                              search for FreeImage includes.
# FREEIMAGE_LIBRARY_DIR_HINTS: List of additional directories in which to
#                              search for FreeImage libraries.

set(FREEIMAGE_INCLUDE_DIR_HINTS "" CACHE PATH "FreeImage include directory")
set(FREEIMAGE_LIBRARY_DIR_HINTS "" CACHE PATH "FreeImage library directory")

unset(FREEIMAGE_FOUND)

find_package(FreeImage CONFIG QUIET)
if(FreeImage_FOUND)
    if(TARGET freeimage::FreeImage)
        set(FREEIMAGE_FOUND TRUE)
        message(STATUS "Found FreeImage")
        message(STATUS "  Target : freeimage::FreeImage")
    endif()
else()
    list(APPEND FREEIMAGE_CHECK_INCLUDE_DIRS
        ${FREEIMAGE_INCLUDE_DIR_HINTS}
        /usr/include
        /usr/local/include
        /opt/include
        /opt/local/include
    )

    list(APPEND FREEIMAGE_CHECK_LIBRARY_DIRS
        ${FREEIMAGE_LIBRARY_DIR_HINTS}
        /usr/lib
        /usr/local/lib
        /opt/lib
        /opt/local/lib
    )

    find_path(FREEIMAGE_INCLUDE_DIRS
        NAMES
        FreeImage.h
        PATHS
        ${FREEIMAGE_CHECK_INCLUDE_DIRS})
    find_library(FREEIMAGE_LIBRARIES
        NAMES
        freeimage
        PATHS
        ${FREEIMAGE_CHECK_LIBRARY_DIRS})

    if(FREEIMAGE_INCLUDE_DIRS AND FREEIMAGE_LIBRARIES)
        set(FREEIMAGE_FOUND TRUE)
    endif()

    if(FREEIMAGE_FOUND)
        message(STATUS "Found FreeImage")
        message(STATUS "  Includes : ${FREEIMAGE_INCLUDE_DIRS}")
        message(STATUS "  Libraries : ${FREEIMAGE_LIBRARIES}")
    endif()

    add_library(freeimage::FreeImage INTERFACE IMPORTED)
    target_include_directories(
        freeimage::FreeImage INTERFACE ${FREEIMAGE_INCLUDE_DIRS})
    target_link_libraries(
        freeimage::FreeImage INTERFACE ${FREEIMAGE_LIBRARIES})
endif()

if(NOT FREEIMAGE_FOUND AND FREEIMAGE_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find FreeImage")
endif()

if (FREEIMAGE_FOUND)
    include_directories(${FREEIMAGE_INCLUDE_PATH})

    add_executable(ImageFilter2D ImageFilter2D.cpp)
    target_link_libraries(ImageFilter2D ${OPENCL_LIBRARIES} ${FREEIMAGE_LIBRARIES})

    configure_file(ImageFilter2D.cl ${CMAKE_CURRENT_BINARY_DIR}/ImageFilter2D.cl COPYONLY)
endif (FREEIMAGE_INCLUDE_PATH AND FREEIMAGE_LIBRARY)
