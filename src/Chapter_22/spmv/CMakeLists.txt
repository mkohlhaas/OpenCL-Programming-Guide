add_executable(spmv spmv.c matrix_gen.c )
target_link_libraries(spmv PRIVATE OpenCL::OpenCL)
