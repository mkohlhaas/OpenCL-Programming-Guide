#include <CL/cl.h>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

const int ARRAY_SIZE = 1024;

//  Create an OpenCL context on the first available platform using either a GPU or CPU depending on what is available.
cl_context CreateContext() {
  // First, select an OpenCL platform to run on.  For this example, we simply choose the first available platform.
  // Normally, you would query for all available platforms and select the most appropriate one.
  cl_uint numPlatforms;
  cl_platform_id firstPlatformId;
  cl_int errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
  if (errNum || numPlatforms == 0) {
    std::cerr << "Failed to find any OpenCL platforms." << std::endl;
    return EXIT_SUCCESS;
  }
  // Attempt to create a GPU-based context. If that fails, try to create a CPU-based context.
  cl_context_properties contextProperties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)firstPlatformId, 0};
  cl_context context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);
  if (errNum) {
    std::cout << "Could not create GPU context, trying CPU..." << std::endl;
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU, NULL, NULL, &errNum);
    if (errNum) {
      std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
      return EXIT_SUCCESS;
    }
  }
  return context;
}

// Create a command-queue on the first device available on the created context.
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device) {
  // Get the size of the devices buffer.
  size_t deviceBufferSize;
  cl_int errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
  if (errNum) {
    std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
    return EXIT_SUCCESS;
  }

  if (!deviceBufferSize) {
    std::cerr << "No devices available.";
    return EXIT_SUCCESS;
  }

  cl_device_id *devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
  errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
  if (errNum) {
    delete[] devices;
    std::cerr << "Failed to get device IDs";
    return EXIT_SUCCESS;
  }

  // In this example, we just choose the first available device.
  // In a real program, you would likely use all available devices or choose the highest performance device based on
  // OpenCL device queries
  cl_command_queue commandQueue = clCreateCommandQueueWithProperties(context, devices[0], 0, NULL);
  if (!commandQueue) {
    delete[] devices;
    std::cerr << "Failed to create commandQueue for device 0";
    return EXIT_SUCCESS;
  }

  *device = devices[0];
  delete[] devices;
  return commandQueue;
}

cl_program CreateProgram(cl_context context, cl_device_id device, const char *fileName) {

  std::ifstream kernelFile(fileName, std::ios::in);
  if (!kernelFile.is_open()) {
    std::cerr << "Failed to open file for reading: " << fileName << std::endl;
    return EXIT_SUCCESS;
  }

  std::ostringstream oss;
  oss << kernelFile.rdbuf();

  std::string srcStdStr = oss.str();
  const char *srcStr = srcStdStr.c_str();
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&srcStr, NULL, NULL);
  if (!program) {
    std::cerr << "Failed to create CL program from source." << std::endl;
    return EXIT_SUCCESS;
  }

  cl_int errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (errNum) {
    char buildLog[16384];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);

    std::cerr << "Error in kernel: " << std::endl;
    std::cerr << buildLog;
    clReleaseProgram(program);
    return EXIT_SUCCESS;
  }

  return program;
}

bool CreateMemObjects(cl_context context, cl_mem memObjects[3], float *a, float *b) {
  // Note usage of 'CL_MEM_COPY_HOST_PTR'.
  memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * ARRAY_SIZE, a, NULL);
  memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * ARRAY_SIZE, b, NULL);
  memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * ARRAY_SIZE, NULL, NULL);

  if (!memObjects[0] || !memObjects[1] || !memObjects[2]) {
    std::cerr << "Error creating memory objects." << std::endl;
    return false;
  }

  return true;
}

void Cleanup(cl_context context, cl_command_queue commandQueue, cl_program program, cl_kernel kernel, cl_mem memObjects[3],
             char const *message, int returnStatus) {
  for (int i = 0; i < 3; i++) {
    if (memObjects[i])
      clReleaseMemObject(memObjects[i]);
  }
  if (commandQueue)
    clReleaseCommandQueue(commandQueue);
  if (kernel)
    clReleaseKernel(kernel);
  if (program)
    clReleaseProgram(program);
  if (context)
    clReleaseContext(context);
  exit(returnStatus);
}

int main(int argc, char **argv) {

  // Create an OpenCL context on first available platform
  cl_context context = CreateContext();
  if (!context) {
    std::cerr << "Failed to create OpenCL context." << std::endl;
    return EXIT_FAILURE;
  }

  cl_device_id device;
  cl_command_queue commandQueue = CreateCommandQueue(context, &device);
  if (!commandQueue) {
    Cleanup(context, NULL, NULL, NULL, NULL, "Failed to create a command queue.", EXIT_FAILURE);
  }

  cl_program program = CreateProgram(context, device, "HelloWorld.cl");
  if (!program) {
    Cleanup(context, commandQueue, NULL, NULL, NULL, "Failed to create a program.", EXIT_FAILURE);
  }

  // Create OpenCL kernel
  cl_kernel kernel = clCreateKernel(program, "hello_kernel", NULL);
  if (!kernel) {
    Cleanup(context, commandQueue, program, NULL, NULL, "Failed to create kernel", EXIT_FAILURE);
  }

  // Create memory objects that will be used as arguments to kernel.
  // First create host memory arrays that will be used to store the arguments to the kernel.
  float a[ARRAY_SIZE];
  float b[ARRAY_SIZE];
  float result[ARRAY_SIZE];
  for (int i = 0; i < ARRAY_SIZE; i++) {
    a[i] = (float)i;
    b[i] = (float)(i * 2);
  }

  cl_mem memObjects[3] = {0, 0, 0};
  if (!CreateMemObjects(context, memObjects, a, b)) {
    Cleanup(context, commandQueue, program, kernel, memObjects, "Failed to create memory objects.", EXIT_FAILURE);
  }

  cl_int errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
  errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
  errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);
  if (errNum) {
    Cleanup(context, commandQueue, program, kernel, memObjects, "Error setting kernel arguments.", EXIT_FAILURE);
  }

  // Queue the kernel up for execution across the array.
  size_t globalWorkSize[1] = {ARRAY_SIZE};
  size_t localWorkSize[1] = {1};
  errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
  if (errNum) {
    Cleanup(context, commandQueue, program, kernel, memObjects, "Error queuing kernel for execution.", EXIT_FAILURE);
  }

  // Read the output buffer back to the Host.
  errNum = clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE, 0, ARRAY_SIZE * sizeof(float), result, 0, NULL, NULL);
  if (errNum) {
    Cleanup(context, commandQueue, program, kernel, memObjects, "Error reading result buffer.", EXIT_FAILURE);
  }

  // Output the result buffer.
  for (int i = 0; i < ARRAY_SIZE; i++) {
    std::cout << result[i] << " ";
  }
  Cleanup(context, commandQueue, program, kernel, memObjects, "Executed program succesfully.", EXIT_SUCCESS);
}
