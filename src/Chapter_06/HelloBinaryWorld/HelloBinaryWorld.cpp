#include <CL/cl.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

const int ARRAY_SIZE = 1000;

cl_context CreateContext() {

  cl_platform_id firstPlatformId;
  cl_uint numPlatforms;
  cl_int errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
  if (errNum || numPlatforms == 0) {
    std::cerr << "Failed to find any OpenCL platforms." << std::endl;
    return NULL;
  }

  cl_context_properties contextProperties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)firstPlatformId, 0};
  cl_context context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);
  if (errNum) {
    std::cout << "Could not create GPU context, trying CPU..." << std::endl;
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU, NULL, NULL, &errNum);
    if (errNum) {
      std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
      return NULL;
    }
  }

  return context;
}

cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device) {
  // Get the size of the devices buffer.
  size_t deviceBufferSize;
  cl_int errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
  if (errNum) {
    std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
    return NULL;
  }

  if (!deviceBufferSize) {
    std::cerr << "No devices available.";
    return NULL;
  }

  // Allocate memory for the devices buffer.
  cl_device_id *devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
  errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
  if (errNum) {
    delete[] devices;
    std::cerr << "Failed to get device IDs";
    return NULL;
  }

  cl_command_queue commandQueue = clCreateCommandQueueWithProperties(context, devices[0], 0, NULL);
  if (!commandQueue) {
    delete[] devices;
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
  if (!program) {
    std::cerr << "Failed to create CL program from source." << std::endl;
    return NULL;
  }

  cl_int errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (errNum) {
    char buildLog[16384];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
    std::cerr << "Error in kernel: " << std::endl;
    std::cerr << buildLog;
    clReleaseProgram(program);
    return NULL;
  }

  return program;
}

// Attempt to create the program object from a cached binary.
// Note that on first run this will fail because the binary has not yet been created.
cl_program CreateProgramFromBinary(cl_context context, cl_device_id device, const char *fileName) {
  FILE *fp = fopen(fileName, "rb");
  if (!fp) {
    return NULL;
  }

  size_t binarySize;
  fseek(fp, 0, SEEK_END);
  binarySize = ftell(fp);
  rewind(fp);

  unsigned char *programBinary = new unsigned char[binarySize];
  fread(programBinary, 1, binarySize, fp);
  fclose(fp);

  cl_int errNum;
  cl_int binaryStatus;
  cl_program program = clCreateProgramWithBinary(context, 1, &device, &binarySize,
                                                 (const unsigned char **)&programBinary, &binaryStatus, &errNum);
  delete[] programBinary;
  if (errNum) {
    std::cerr << "Error loading program binary." << std::endl;
    return NULL;
  }

  if (binaryStatus) {
    std::cerr << "Invalid binary for device" << std::endl;
    return NULL;
  }

  errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (errNum) {
    char buildLog[16384];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
    std::cerr << "Error in program: " << std::endl;
    std::cerr << buildLog << std::endl;
    clReleaseProgram(program);
    return NULL;
  }

  return program;
}

// Retreive program binary for all of the devices attached to the program an and store the one for the device passed in.
bool SaveProgramBinary(cl_program program, cl_device_id device, const char *fileName) {
  // 1 - Query for number of devices attached to program
  cl_uint numDevices = 0;
  cl_int errNum = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &numDevices, NULL);
  if (errNum) {
    std::cerr << "Error querying for number of devices." << std::endl;
    return false;
  }

  // 2 - Get all of the Device IDs
  cl_device_id *devices = new cl_device_id[numDevices];
  errNum = clGetProgramInfo(program, CL_PROGRAM_DEVICES, sizeof(cl_device_id) * numDevices, devices, NULL);
  if (errNum) {
    std::cerr << "Error querying for devices." << std::endl;
    delete[] devices;
    return false;
  }

  // 3 - Determine the size of each program binary
  size_t *programBinarySizes = new size_t[numDevices];
  errNum = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t) * numDevices, programBinarySizes, NULL);
  if (errNum) {
    std::cerr << "Error querying for program binary sizes." << std::endl;
    delete[] devices;
    delete[] programBinarySizes;
    return false;
  }

  unsigned char **programBinaries = new unsigned char *[numDevices];
  for (cl_uint i = 0; i < numDevices; i++) {
    programBinaries[i] = new unsigned char[programBinarySizes[i]];
  }

  // 4 - Get all of the program binaries
  errNum = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char *) * numDevices, programBinaries, NULL);
  if (errNum) {
    std::cerr << "Error querying for program binaries" << std::endl;

    delete[] devices;
    delete[] programBinarySizes;
    for (cl_uint i = 0; i < numDevices; i++) {
      delete[] programBinaries[i];
    }
    delete[] programBinaries;
    return false;
  }

  // 5 - Finally store the binaries for the device requested out to disk for future reading.
  for (cl_uint i = 0; i < numDevices; i++) {
    // Store the binary just for the device requested.  In a scenario where
    // multiple devices were being used you would save all of the binaries out here.
    if (!devices[i]) {
      FILE *fp = fopen(fileName, "wb");
      fwrite(programBinaries[i], 1, programBinarySizes[i], fp);
      fclose(fp);
      break;
    }
  }

  // Cleanup
  delete[] devices;
  delete[] programBinarySizes;
  for (cl_uint i = 0; i < numDevices; i++) {
    delete[] programBinaries[i];
  }
  delete[] programBinaries;
  return true;
}

// Create memory objects used as the arguments to the kernel The kernel takes three arguments:
// result (output), a (input), and b (input).
bool CreateMemObjects(cl_context context, cl_mem memObjects[3], float *a, float *b) {
  memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * ARRAY_SIZE, a, NULL);
  memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * ARRAY_SIZE, b, NULL);
  memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * ARRAY_SIZE, NULL, NULL);

  if (!memObjects[0] || !memObjects[1] || !memObjects[2]) {
    std::cerr << "Error creating memory objects." << std::endl;
    return false;
  }

  return true;
}

// Cleanup OpenCL resources.
void Cleanup(cl_context context, cl_command_queue commandQueue, cl_program program, cl_kernel kernel,
             cl_mem memObjects[3]) {
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
}

int main(int argc, char **argv) {

  // Create an OpenCL context on first available platform.
  cl_context context = CreateContext();
  if (!context) {
    std::cerr << "Failed to create OpenCL context." << std::endl;
    return EXIT_FAILURE;
  }

  // Create a command-queue on the first device available on the created context.
  cl_mem memObjects[3] = {0, 0, 0};
  cl_device_id device;
  cl_program program;
  cl_kernel kernel;
  cl_command_queue commandQueue = CreateCommandQueue(context, &device);
  if (!commandQueue) {
    Cleanup(context, commandQueue, program, kernel, memObjects);
    return EXIT_FAILURE;
  }

  // Create OpenCL program - first attempt to load cached binary.
  // If that is not available, then create the program from source and store the binary for future use.
  std::cout << "Attempting to create program from binary..." << std::endl;
  program = CreateProgramFromBinary(context, device, "HelloWorld.cl.bin");
  if (!program) {
    std::cout << "Binary not loaded, create from source..." << std::endl;
    program = CreateProgram(context, device, "HelloWorld.cl");
    if (!program) {
      Cleanup(context, commandQueue, program, kernel, memObjects);
      return EXIT_FAILURE;
    }

    std::cout << "Save program binary for future run..." << std::endl;
    if (!SaveProgramBinary(program, device, "HelloWorld.cl.bin")) {
      std::cerr << "Failed to write program binary" << std::endl;
      Cleanup(context, commandQueue, program, kernel, memObjects);
      return EXIT_FAILURE;
    }
  } else {
    std::cout << "Read program from binary." << std::endl;
  }

  // Create OpenCL kernel
  kernel = clCreateKernel(program, "hello_kernel", NULL);
  if (!kernel) {
    std::cerr << "Failed to create kernel" << std::endl;
    Cleanup(context, commandQueue, program, kernel, memObjects);
    return EXIT_FAILURE;
  }

  // Create memory objects that will be used as arguments to kernel.
  // First create host memory arrays that will be used to store the arguments to the kernel
  float result[ARRAY_SIZE];
  float a[ARRAY_SIZE];
  float b[ARRAY_SIZE];
  for (int i = 0; i < ARRAY_SIZE; i++) {
    a[i] = (float)i;
    b[i] = (float)(i * 2);
  }

  if (!CreateMemObjects(context, memObjects, a, b)) {
    Cleanup(context, commandQueue, program, kernel, memObjects);
    return EXIT_FAILURE;
  }

  // Set the kernel arguments (result, a, b)
  cl_int errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
  errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
  errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);
  if (errNum) {
    std::cerr << "Error setting kernel arguments." << std::endl;
    Cleanup(context, commandQueue, program, kernel, memObjects);
    return EXIT_FAILURE;
  }

  size_t globalWorkSize[1] = {ARRAY_SIZE};
  size_t localWorkSize[1] = {1};

  // Queue the kernel up for execution across the array
  errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
  if (errNum) {
    std::cerr << "Error queuing kernel for execution." << std::endl;
    Cleanup(context, commandQueue, program, kernel, memObjects);
    return EXIT_FAILURE;
  }

  // Read the output buffer back to the Host
  errNum =
      clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE, 0, ARRAY_SIZE * sizeof(float), result, 0, NULL, NULL);
  if (errNum) {
    std::cerr << "Error reading result buffer." << std::endl;
    Cleanup(context, commandQueue, program, kernel, memObjects);
    return EXIT_FAILURE;
  }

  // Output the result buffer
  for (int i = 0; i < ARRAY_SIZE; i++) {
    std::cout << result[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "Executed program succesfully." << std::endl;
  Cleanup(context, commandQueue, program, kernel, memObjects);

  return EXIT_SUCCESS;
}
