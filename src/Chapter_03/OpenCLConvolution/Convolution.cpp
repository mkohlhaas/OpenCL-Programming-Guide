#include <CL/cl.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

const unsigned int inputSignalWidth = 8;
const unsigned int inputSignalHeight = 8;

cl_uint inputSignal[inputSignalWidth][inputSignalHeight] = {
    {3, 1, 1, 4, 8, 2, 1, 3}, {4, 2, 1, 1, 2, 1, 2, 3}, {4, 4, 4, 4, 3, 2, 2, 2}, {9, 8, 3, 8, 9, 0, 0, 0},
    {9, 3, 3, 9, 0, 0, 0, 0}, {0, 9, 0, 8, 0, 0, 0, 0}, {3, 0, 8, 8, 9, 4, 4, 4}, {5, 9, 8, 1, 8, 1, 1, 1}};

const unsigned int outputSignalWidth = 6;
const unsigned int outputSignalHeight = 6;

cl_uint outputSignal[outputSignalWidth][outputSignalHeight];

const unsigned int maskWidth = 3;
const unsigned int maskHeight = 3;

cl_uint mask[maskWidth][maskHeight] = {
    {1, 1, 1},
    {1, 0, 1},
    {1, 1, 1},
};

inline void checkErr(cl_int err, const char *name) {
  if (err) {
    std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
    exit(EXIT_FAILURE);
  }
}

void CL_CALLBACK contextCallback(const char *errInfo, const void *private_info, size_t cb, void *user_data) {
  std::cout << "Error occured during context use: " << errInfo << std::endl;
  // Should really perform any clean-up on at this point but for simplicitly just exit.
  exit(EXIT_FAILURE);
}

int main(void) {
  // Select an OpenCL platform to run on.
  cl_uint numPlatforms;
  checkErr(clGetPlatformIDs(0, NULL, &numPlatforms), "clGetPlatformIDs");

  cl_platform_id *platformIDs = (cl_platform_id *)alloca(sizeof(cl_platform_id) * numPlatforms);
  checkErr(clGetPlatformIDs(numPlatforms, platformIDs, NULL), "clGetPlatformIDs");

  // Iterate through the list of platforms until we find one that supports a CPU device, otherwise fail with an error.
  cl_device_id *deviceIDs;
  cl_uint numDevices;
  cl_int errNum;
  cl_uint i;
  for (i = 0; i < numPlatforms; i++) {
    errNum = clGetDeviceIDs(platformIDs[i], CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
    if (errNum)
      continue; // try next platform
    deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
    errNum = clGetDeviceIDs(platformIDs[i], CL_DEVICE_TYPE_CPU, numDevices, &deviceIDs[0], NULL);
    if (!errNum)
      break; // we found a CPU device
  }
  if (!deviceIDs) {
    checkErr(-1, "clGetDeviceIDs");
  }

  // Create an OpenCL context on the selected platform.
  cl_context_properties contextProperties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platformIDs[i], 0};
  cl_context context = clCreateContext(contextProperties, numDevices, deviceIDs, &contextCallback, NULL, &errNum);
  checkErr(errNum, "clCreateContext");

  std::ifstream srcFile("Convolution.cl");
  checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading Convolution.cl");

  std::string srcProg(std::istreambuf_iterator<char>(srcFile), (std::istreambuf_iterator<char>()));

  const char *src = srcProg.c_str();
  size_t length = srcProg.length();

  // Create program from source.
  cl_program program = clCreateProgramWithSource(context, 1, &src, &length, &errNum);
  checkErr(errNum, "clCreateProgramWithSource");

  // Build program.
  errNum = clBuildProgram(program, numDevices, deviceIDs, NULL, NULL, NULL);
  if (errNum) {
    char buildLog[16384];
    clGetProgramBuildInfo(program, deviceIDs[0], CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
    std::cerr << "Error in kernel: " << std::endl << buildLog;
    checkErr(errNum, "clBuildProgram");
  }

  // Create kernel object.
  cl_kernel kernel = clCreateKernel(program, "convolve", &errNum);
  checkErr(errNum, "clCreateKernel");

  // Allocate buffers.
  cl_mem_flags flags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
  size_t bufferSize = sizeof(cl_uint) * inputSignalHeight * inputSignalWidth;
  cl_mem inputSignalBuffer = clCreateBuffer(context, flags, bufferSize, static_cast<void *>(inputSignal), &errNum);
  checkErr(errNum, "clCreateBuffer(inputSignal)");

  flags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
  bufferSize = sizeof(cl_uint) * maskHeight * maskWidth;
  cl_mem maskBuffer = clCreateBuffer(context, flags, bufferSize, static_cast<void *>(mask), &errNum);
  checkErr(errNum, "clCreateBuffer(mask)");

  flags = CL_MEM_WRITE_ONLY;
  bufferSize = sizeof(cl_uint) * outputSignalHeight * outputSignalWidth;
  cl_mem outputSignalBuffer = clCreateBuffer(context, flags, bufferSize, NULL, &errNum);
  checkErr(errNum, "clCreateBuffer(outputSignal)");

  // Pick the first device and create command queue.
  cl_command_queue queue = clCreateCommandQueueWithProperties(context, deviceIDs[0], 0, &errNum);
  checkErr(errNum, "clCreateCommandQueue");

  errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputSignalBuffer);
  errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &maskBuffer);
  errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputSignalBuffer);
  errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &inputSignalWidth);
  errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &maskWidth);
  checkErr(errNum, "clSetKernelArg");

  // Queue the kernel up for execution across the array.
  const size_t globalWorkSize[1] = {outputSignalWidth * outputSignalHeight};
  const size_t localWorkSize[1] = {1};
  checkErr(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL),
           "clEnqueueNDRangeKernel");

  bufferSize = sizeof(cl_uint) * outputSignalHeight * outputSignalHeight;
  checkErr(clEnqueueReadBuffer(queue, outputSignalBuffer, CL_TRUE, 0, bufferSize, outputSignal, 0, NULL, NULL),
           "clEnqueueReadBuffer");

  // Output the result buffer.
  for (int y = 0; y < outputSignalHeight; y++) {
    for (int x = 0; x < outputSignalWidth; x++) {
      std::cout << outputSignal[x][y] << " ";
    }
    std::cout << std::endl;
  }

  std::cout << std::endl << "Executed program succesfully." << std::endl;
}
