#include <CL/cl.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string.h>

#include "FreeImage.h"

cl_int errNum;

cl_context CreateContext() {

  cl_uint numPlatforms;
  cl_platform_id firstPlatformId;
  errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
  if (errNum != CL_SUCCESS || numPlatforms <= 0) {
    std::cerr << "Failed to find any OpenCL platforms." << std::endl;
    return NULL;
  }

  cl_context_properties contextProperties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)firstPlatformId, 0};
  cl_context context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);
  if (errNum != CL_SUCCESS) {
    std::cout << "Could not create GPU context, trying CPU..." << std::endl;
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU, NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS) {
      std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
      return NULL;
    }
  }

  return context;
}

cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device) {

  size_t deviceBufferSize;
  errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
    return NULL;
  }

  if (deviceBufferSize <= 0) {
    std::cerr << "No devices available.";
    return NULL;
  }

  cl_device_id *devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
  errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Failed to get device IDs";
    return NULL;
  }

  cl_command_queue commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
  if (commandQueue == NULL) {
    std::cerr << "Failed to create commandQueue for device 0";
    return NULL;
  }

  *device = devices[0];
  delete[] devices;
  return commandQueue;
}

cl_program CreateProgram(cl_context context, cl_device_id device, const char *fileName) {

  std::ifstream kernelFile(fileName, std::ios::in);
  if (!kernelFile.is_open()) {
    std::cerr << "Failed to open file for reading: " << fileName << std::endl;
    return NULL;
  }

  std::ostringstream oss;
  oss << kernelFile.rdbuf();

  std::string srcStdStr = oss.str();
  const char *srcStr = srcStdStr.c_str();
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&srcStr, NULL, NULL);
  if (program == NULL) {
    std::cerr << "Failed to create CL program from source." << std::endl;
    return NULL;
  }

  errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (errNum != CL_SUCCESS) {
    // Determine the reason for the error
    char buildLog[16384];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
    std::cerr << "Error in kernel: " << std::endl << buildLog;
    clReleaseProgram(program);
    return NULL;
  }

  return program;
}

void Cleanup(cl_context context, cl_command_queue commandQueue, cl_program program, cl_kernel kernel, cl_mem imageObjects[2], cl_sampler sampler, char *buffer, const char *message, int returnValue) {

  for (int i = 0; i < 2; i++) {
    if (imageObjects[i] != 0)
      clReleaseMemObject(imageObjects[i]);
  }

  if (!commandQueue)
    clReleaseCommandQueue(commandQueue);
  if (!kernel)
    clReleaseKernel(kernel);
  if (!program)
    clReleaseProgram(program);
  if (!sampler)
    clReleaseSampler(sampler);
  if (!context)
    clReleaseContext(context);
  if (!buffer)
    delete[] buffer;

  std::cerr << message << std::endl;
  exit(returnValue);
}

// Load an image using FreeImage and create an OpenCL image out of it.
cl_mem LoadImage(cl_context context, char *fileName, size_t &width, size_t &height) {
  FREE_IMAGE_FORMAT format = FreeImage_GetFileType(fileName, 0);
  FIBITMAP *image = FreeImage_Load(format, fileName);

  FIBITMAP *temp = image;
  image = FreeImage_ConvertTo32Bits(image);
  FreeImage_Unload(temp);

  width = FreeImage_GetWidth(image);
  height = FreeImage_GetHeight(image);

  char *buffer = new char[width * height * 4];
  memcpy(buffer, FreeImage_GetBits(image), width * height * 4);

  FreeImage_Unload(image);

  cl_image_format clImageFormat;
  clImageFormat.image_channel_order = CL_RGBA;
  clImageFormat.image_channel_data_type = CL_UNORM_INT8;

  cl_int errNum;
  cl_mem clImage;
  clImage = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &clImageFormat, width, height, 0, buffer, &errNum);

  if (errNum != CL_SUCCESS) {
    std::cerr << "Error creating CL image object" << std::endl;
    return NULL;
  }

  return clImage;
}

bool SaveImage(char *fileName, char *buffer, int width, int height) {
  FREE_IMAGE_FORMAT format = FreeImage_GetFIFFromFilename(fileName);
  FIBITMAP *image = FreeImage_ConvertFromRawBits((BYTE *)buffer, width, height, width * 4, 32, 0xFF000000, 0x00FF0000, 0x0000FF00);
  return (FreeImage_Save(format, image, fileName)) ? true : false;
}

// Round up to the nearest multiple of the group size.
size_t RoundUp(int groupSize, int globalSize) {
  int r = globalSize % groupSize;
  if (r) {
    return globalSize + groupSize - r;
  } else {
    return globalSize;
  }
}

int main(int argc, char **argv) {

  if (argc != 3) {
    std::cerr << "USAGE: " << argv[0] << " <inputImageFile> <outputImageFiles>" << std::endl;
  }

  cl_context context = CreateContext();
  if (!context) {
    std::cerr << "Failed to create OpenCL context." << std::endl;
  }

  cl_device_id device;
  cl_command_queue commandQueue = CreateCommandQueue(context, &device);
  if (!commandQueue) {
    Cleanup(context, commandQueue, NULL, NULL, NULL, NULL, NULL, "Could not create command queue.", EXIT_FAILURE);
  }

  // Device supports images ?
  cl_bool imageSupport = CL_FALSE;
  clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), &imageSupport, NULL);
  if (!imageSupport) {
    Cleanup(context, commandQueue, NULL, NULL, NULL, NULL, NULL, "OpenCL device does not support images.", EXIT_FAILURE);
  }

  // Load input image from file and load it into an OpenCL image object.
  size_t width, height;
  cl_mem imageObjects[2] = {0, 0};
  imageObjects[0] = LoadImage(context, argv[1], width, height);
  if (!imageObjects[0]) {
    std::cerr << "Error loading: " << std::string(argv[1]) << std::endl;
    std::string str1 = "Error loading: ";
    auto errorMessage = str1.append(std::string(argv[1])).c_str();
    Cleanup(context, commandQueue, NULL, NULL, imageObjects, NULL, NULL, errorMessage, EXIT_FAILURE);
  }

  // Create ouput image object.
  cl_image_format clImageFormat;
  clImageFormat.image_channel_order = CL_RGBA;
  clImageFormat.image_channel_data_type = CL_UNORM_INT8;
  imageObjects[1] = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &clImageFormat, width, height, 0, NULL, &errNum);

  if (errNum) {
    Cleanup(context, commandQueue, NULL, NULL, imageObjects, NULL, NULL, "Error creating CL output image object.", EXIT_FAILURE);
  }

  // Create sampler for sampling image object.
  cl_sampler sampler = clCreateSampler(context,
                                       CL_FALSE, // Non-normalized coordinates
                                       CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &errNum);

  if (errNum) {
    Cleanup(context, commandQueue, NULL, NULL, imageObjects, sampler, NULL, "Error creating CL sampler object.", EXIT_FAILURE);
  }

  // Create program.
  cl_program program = CreateProgram(context, device, "ImageFilter2D.cl");
  if (!program) {
    Cleanup(context, commandQueue, program, NULL, imageObjects, sampler, NULL, "Error creating OpenCL program.", EXIT_FAILURE);
  }

  // Create kernel.
  cl_kernel kernel = clCreateKernel(program, "gaussian_filter", NULL);
  if (!kernel) {
    Cleanup(context, commandQueue, program, kernel, imageObjects, sampler, NULL, "Failed to create kernel", EXIT_FAILURE);
  }

  errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &imageObjects[0]);
  errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &imageObjects[1]);
  errNum |= clSetKernelArg(kernel, 2, sizeof(cl_sampler), &sampler);
  errNum |= clSetKernelArg(kernel, 3, sizeof(cl_int), &width);
  errNum |= clSetKernelArg(kernel, 4, sizeof(cl_int), &height);
  if (errNum) {
    Cleanup(context, commandQueue, program, kernel, imageObjects, sampler, NULL, "Error setting kernel arguments.", EXIT_FAILURE);
  }

  size_t localWorkSize[2] = {16, 16};
  size_t globalWorkSize[2] = {RoundUp(localWorkSize[0], width), RoundUp(localWorkSize[1], height)};

  // Queue the kernel up for execution.
  errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
  if (errNum != CL_SUCCESS) {
    Cleanup(context, commandQueue, program, kernel, imageObjects, sampler, NULL, "Error queueing kernel for execution.", EXIT_FAILURE);
  }

  // Read the output buffer back to the Host.
  char *buffer = new char[width * height * 4];
  size_t origin[3] = {0, 0, 0};
  size_t region[3] = {width, height, 1};
  errNum = clEnqueueReadImage(commandQueue, imageObjects[1], CL_TRUE, origin, region, 0, 0, buffer, 0, NULL, NULL);
  if (errNum != CL_SUCCESS) {
    Cleanup(context, commandQueue, program, kernel, imageObjects, sampler, NULL, "Error reading result buffer.", EXIT_FAILURE);
  }

  // memset(buffer, 0xff, width * height * 4);
  // Save the image out to disk.
  if (!SaveImage(argv[2], buffer, width, height)) {
    Cleanup(context, commandQueue, program, kernel, imageObjects, sampler, buffer, "Error writing output image: ", EXIT_FAILURE);
  }

  Cleanup(context, commandQueue, program, kernel, imageObjects, sampler, buffer, "Everything worked fine.", EXIT_SUCCESS);
}
