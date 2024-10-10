#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <iostream>
#include <vector>

int main() {
    // Initialize the Metal device
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Failed to find a compatible Metal device." << std::endl;
        return -1;
    }

    // Create a command queue
    MTL::CommandQueue* commandQueue = device->newCommandQueue();
    if (!commandQueue) {
        std::cerr << "Failed to create a command queue." << std::endl;
        device->release();
        return -1;
    }

    // Load the default library (assumes operations.metal is part of the project)
    // extra steps here are necessary since we are not in X-Code. We have to use
    // xcrun -sdk to create a .metallib file. see tutorial for instructions
    NS::Error* error = nullptr;
    NS::String* filePath = NS::String::string("/Users/moritzhof/Documents/Programming/MetalCpp/VectorOperations/operations.metallib", NS::UTF8StringEncoding);
    
    auto lib = device->newLibrary(filePath, &error);
    if(!lib){
        std::cerr << "Failed to Library\n";
        std::exit(-1);
    }

    // Retrieve the compute function from the library
    NS::String* functionName = NS::String::string("sum_vectors", NS::UTF8StringEncoding);
    MTL::Function* computeFunction = lib->newFunction(functionName);
    if (!computeFunction) {
        std::cerr << "Failed to find the compute function 'sum_vectors'." << std::endl;
        lib->release();
        commandQueue->release();
        device->release();
        return -1;
    }

    // Create a compute pipeline state
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

    // Define the length of the vectors and buffer size
    const uint32_t arrayLength = 1024;
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

    // Create a command buffer to encode commands
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

    // Commit the command buffer and wait for it to complete
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    // Read the output data from the GPU
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

    // Release all allocated resources
    computeEncoder->release();
    commandBuffer->release();
    aBuffer->release();
    bBuffer->release();
    cBuffer->release();
    computePipelineState->release();
    computeFunction->release();
    lib->release();
    functionName->release();
    commandQueue->release();
    device->release();

    return 0;
}

