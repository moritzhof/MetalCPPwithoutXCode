# Introduction to Metal-C++ with Vector Addition outside X-Code IDE


 This project demonstrates how to use the Metal-C++ API to perform vector addition on the M-generation GPUs (but this also worked on Intel CPU and AMD GPU Macs) using a compute kernel. One could think about this as a way to use Metel-C++ to do scientific computing. The code sets up a Metal compute pipeline to add two arrays of floats (A and B) and stores the result in a third array (C). This example processes 1024 elements using the add_vector kernel defined in operations.metal. I avoid the use of auto so that readers can learn as much as possible from this tutorial, however I do not go into alor of detail of what ever line of code means. A lot more code in provided then necessary in order to have a working example but I wanted to demonstrate as much as possible so that you can get started on setting up your own project. 

## Table of Contents
* [Prerequisites](#prerequisites)
* [Project Structure](#project-structure)
* [Code Explanation](#Code-explanation)
* [1. Include Headers and Define Macros](#1-include-headers-and-define-macros)
* [2. Main Function Overview](#2-main-function-overview)
* [3. Initialize Metal](#3-initialize-metal)
* [4. Load the Compute Function](#4-load-the-compute-function)
* [5. Set Up the Compute Pipeline](#5-set-up-the-compute-pipeline)
* [6. Prepare Data and Buffers](#6-prepare-data-and-buffers)
* [7. Encode Commands](#7-encode-commands)
* [8. Execute the Command Buffer](#8-execute-the-command-buffer)
* [9. Retrieve and Verify Results](#9-retrieve-and-verify-results)
* [10. Clean Up Resources](#9-clean-up-resources)
* [Building and Running the Program](#bulding-and-running-the-program)

## Prerequisites

    -    A Mac with an M3 Pro GPU (or any Metal-compatible GPU).
    -    Xcode installed (latest version recommended).
    -    Basic knowledge of C++ and GPU programming concepts.

## Project Structure

    *    main.cpp: The main C++ source file containing the Metal-C++ code.
    *    operations.metal: The Metal shader file containing the add_vector compute kernel.

# Code Explanation

## 1. Include Headers and Define Macros

At the beginning of the main.cpp file, include the necessary headers and define macros required for the Metal-C++ API:

```cpp
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <iostream>
#include <vector>
```

*    The macros NS_PRIVATE_IMPLEMENTATION, CA_PRIVATE_IMPLEMENTATION, and MTL_PRIVATE_IMPLEMENTATION   are defined to include the private implementations of the Metal and Foundation classes.

*    Headers for Foundation, Metal, and standard C++ libraries are included.

## 2. Main Function Overview

The main function demonstartes the entire process in order to run the GPU on macOS:
```cpp
int main() {
    // Initialization and setup code
    // Data preparation
    // Command encoding
    // Execution and result verification
    // Resource cleanup
    return 0;
}
```

## 3. Initialize Metal

Start by initializing the Metal device and creating a command queue:

```cpp
MTL::Device* device = MTL::CreateSystemDefaultDevice();
if (!device) {
    std::cerr << "Failed to find a compatible Metal device." << std::endl;
    return -1;
}

MTL::CommandQueue* commandQueue = device->newCommandQueue();
if (!commandQueue) {
    std::cerr << "Failed to create a command queue." << std::endl;
    device->release();
    return -1;
}
```
•    MTL::CreateSystemDefaultDevice() obtains the default Metal-compatible GPU.
•    device->newCommandQueue() creates a command queue for submitting commands to the GPU.

## 4. Load the Compute Function

Load the Metal shader library and retrieve the compute function:
This is where things get different since we are not in the X-Code environment. We cannot use the default library. We have to create our own <file>.metallib file. Steps on how to create this will follow in the Building and Running Section:

```cpp
    NS::Error* error = nullptr;
    NS::String* filePath = NS::String::string("/Path/to/metalCpp/Project/<kernel>.metallib", NS::UTF8StringEncoding);
    
    auto lib = device->newLibrary(filePath, &error);
    if(!lib){
        std::cerr << "Failed to Library\n";
        std::exit(-1);
    }
    
    
    NS::String* functionName = NS::String::string("add_vector", NS::UTF8StringEncoding);
    MTL::Function* computeFunction = lib->newFunction(functionName);
    if (!computeFunction) {
        std::cerr << "Failed to find the compute function 'add_vector'." << std::endl;
        lib->release();
        commandQueue->release();
        device->release();
        return -1;
    }
```
* device->newLibrary loads the metal library, which we create later on from <kernel>.metal. In this case we will create operations.metallib from operations.metal.
* lib->newFunction will retrieve the kernel located into <kernel>.metal file. 
* The functiond names have to match. In this tutorial the operations.metal contains the kernel 'add_vector'. 
* You will also have to update the Path in filePath to where your files are held. 

## 5. Set Up the Compute Pipeline

Initialize the input data and create buffers to store it on the GPU:
```cpp
    MTL::ComputePipelineState* computePipelineState = device->newComputePipelineState(computeFunction, &error);
    if (!computePipelineState) {
        std::cerr << "Failed to create compute pipeline state: "
                  << (error ? error->localizedDescription()->utf8String() : "Unknown error") << std::endl;
        computeFunction->release();
        lib->release();
        commandQueue->release();
        device->release();
        return -1;
    }
```

## 6. Prepare Data and Buffers

Initialize the input data and create buffers to store it on the GPU:
  ```cpp  const uint32_t arrayLength = 1024;
    const size_t bufferSize = arrayLength * sizeof(float);

    // Initialize input data
    std::vector<float> a(arrayLength);
    std::vector<float> b(arrayLength);
    for (uint32_t i = 0; i < arrayLength; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
    }

    // Create buffers for the input and output data
    MTL::Buffer* aBuffer = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);
    MTL::Buffer* bBuffer = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);
    MTL::Buffer* cBuffer = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);

    // Copy data into the Metal buffers
    memcpy(aBuffer->contents(), a.data(), bufferSize);
    memcpy(bBuffer->contents(), b.data(), bufferSize);

    // Notify Metal that the buffers have been modified
    aBuffer->didModifyRange(NS::Range::Make(0, aBuffer->length()));
    bBuffer->didModifyRange(NS::Range::Make(0, bBuffer->length()));
```

*    Define the length of the arrays and compute the buffer size.
*    Input vectors a and b are initialized with sample data.
*    Metal buffers aBuffer, bBuffer, and cBuffer are created to store the data on the GPU.
*    Data is copied into the GPU buffers, and Metal is notified of the changes.

This can also be done in different ways. One way is that you could generate random numbers directly to the device buffer. Here is how you could do it that way: 

```cpp
    MTL::Buffer* _A = _device->newBuffer(buffer_size, MTL::ResourceStorageModeShared);
    MTL::Buffer* _B = _device->newBuffer(buffer_size, MTL::ResourceStorageModeShared);
    MTL::Buffer* _C = _device->newBuffer(buffer_size, MTL::ResourceStorageModeShared);
    
    random_number_generator(_A);
    random_number_generator(_B); 
```

where random_number:generator is given by: 

```cpp
void random_number_generator(MTL::Buffer *buffer){
    float* data_ptr = (float*)buffer->contents();
        for (unsigned long index = 0; index < vector_length; ++index){
            data_ptr[index] = (float)rand() / (float)(RAND_MAX);
        }
}
```
Then you could transfer from how to device to host array: 
```cpp
    auto a = (float*)_A->contents();
    auto b = (float*)_B->contents();
    auto c = (float*)_C->contents();
```

## 7. Encode Commands
This ia rather lengthy step but this demonstrates how you encode the commands to be sent to the GPU.

```cpp
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    if (!commandBuffer) {
        std::cerr << "Failed to create a command buffer." << std::endl;
        // Release resources
        aBuffer->release();
        bBuffer->release();
        cBuffer->release();
        computePipelineState->release();
        computeFunction->release();
        lib->release();
        commandQueue->release();
        device->release();
        return -1;
    }

    // Create a compute command encoder
    MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();
    if (!computeEncoder) {
        std::cerr << "Failed to create a compute command encoder." << std::endl;
        // Release resources
        commandBuffer->release();
        aBuffer->release();
        bBuffer->release();
        cBuffer->release();
        computePipelineState->release();
        computeFunction->release();
        lib->release();
        commandQueue->release();
        device->release();
        return -1;
    }

    // Set the compute pipeline state and buffers
    computeEncoder->setComputePipelineState(computePipelineState);
    computeEncoder->setBuffer(aBuffer, 0, 0);
    computeEncoder->setBuffer(bBuffer, 0, 1);
    computeEncoder->setBuffer(cBuffer, 0, 2);

    // Determine the grid and threadgroup sizes
    MTL::Size gridSize = MTL::Size(arrayLength, 1, 1);
    
    // Ensure the threadgroup size does not exceed the maximum threads per threadgroup
    NS::UInteger threadgroup_Size = computePipelineState->maxTotalThreadsPerThreadgroup();
    if (threadgroup_Size> arrayLength) {
        threadgroup_Size = arrayLength;
    }
    
    
    MTL::Size threadgroupSize = MTL::Size(threadgroup_Size, 1, 1); // Adjust based on the device's capabilities

    // Dispatch the compute kernel
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);

    // End encoding
    computeEncoder->endEncoding();
```
    •    A command buffer and compute command encoder are created to encode the compute commands.
    •    The compute pipeline state and buffers are set for the encoder.
    •    The grid size and threadgroup size are defined to determine how the compute threads are dispatched.
    •    The compute kernel is dispatched with dispatchThreads.

## 8. Execute the Command Buffer
Commit the command buffer to execute the encoded commands on the GPU:
```cpp
commandBuffer->commit();
commandBuffer->waitUntilCompleted();
```

## 9. Retrieve and Verify Results

Access the output data from the GPU and verify the results:
```cpp
    float* cData = static_cast<float*>(cBuffer->contents());

    // Verify the results
    bool isCorrect = true;
    for (uint32_t i = 0; i < arrayLength; ++i) {
        float expected = a[i] + b[i];
        if (cData[i] != expected) {
            std::cerr << "Mismatch at index " << i << ": expected " << expected << ", got " << cData[i] << std::endl;
            isCorrect = false;
            break;
        }
    }

    if (isCorrect) {
        std::cout << "Computation successful! All results are correct." << std::endl;
    }
```

• Cast the contents of cBuffer to a float pointer to access the results.
• A loop checks each element to verify that the GPU computation matches the expected results.

## 10. Clean Up Resources
'''cpp
computeEncoder->release();
commandBuffer->release();
aBuffer->release();
bBuffer->release();
cBuffer->release();
computePipelineState->release();
computeFunction->release();
defaultLibrary->release();
functionName->release();
commandQueue->release();
device->release();
'''

## Building and Running the Program

Since we are not in X-Code IDE, we have to build a .metallib file containing our kernel. For reference, it is explained here: https://developer.apple.com/documentation/metal/shader_libraries/metal_libraries/building_a_shader_library_by_precompiling_source_files .
It is rather straightforward, though. We have an operations.metel file containing the kernel add_vector. 
We first have to compiler the operations.metal into a operations.ir file: In the terminal we execute the following command: 

```cpp
xcrun -sdk macosx metal -o operations.ir -c operations.metal
```

and then from that .ir file we can create the .metallib, as required: 

```cpp
xcrun -sdk macosx metallib -o operations.metallib operations.ir
```
All these files should be in a folder where we you have your main.cpp. After preforming that above commands, we should have the files: 
 main.cpp            metal-cpp           operations.ir       operations.metal    operations.metallib
 
 Note: the metal-cpp is a folder containing the Metal-C++ API file. it can be downloaded here only with a tutorial on how to use it with X-Code: https://developer.apple.com/metal/cpp/. For completeness, I have included the metal-cpp folder in this repository. You may also look at this repository https://github.com/moritzhof/metal-cpp-examples that is also a vector add example using metal-cpp but in X-Code.  However, it is technically not the same example. It is directly translated from Objective-C++ code found from: https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu?language=objc

 
 
 Finally you can compile the code: 
 
```cpp
clang++ -I/Path/to/metal-cpp  main.cpp -o main -std=c++20 -framework Foundation -framework Metal
```

If everything goes well, you should get an executable main

```cpp
./main
```

Hopefully you found this tutorial insightfull and learned something new :) 
