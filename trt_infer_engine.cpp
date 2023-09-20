#include <algorithm>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <random>
#include <iterator>
#include <opencv2/cudaimgproc.hpp>
#include "engine.h"
#include "NvOnnxParser.h"

using namespace nvinfer1;

void Logger::log(Severity severity, const char *msg) noexcept
{
    // Would advise using a proper logging utility such as https://github.com/gabime/spdlog
    // For the sake of this tutorial, will just log to the console.

    // Only log Warnings or more important.
    if (severity <= Severity::kWARNING)
    {
        std::cout << msg << std::endl;
    }
}

Engine::Engine(const Options &options, std::string engine_name, const std::array<float, 3> &subVals, const std::array<float, 3> &divVals,
               bool normalize)
    : m_options(options),
      m_subVals(subVals),
      m_divVals(divVals),
      m_normalize(normalize),
      m_engineName(engine_name)
       {}

Engine::~Engine()
{
    // Free the GPU memory
    for (auto &buffer : m_buffers)
    {
        checkCudaErrorCode(cudaFree(buffer));
    }

    m_buffers.clear();
}

bool Engine::loadNetwork()
{
    // Read the serialized model from disk
    std::ifstream file(m_engineName, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size))
    {
        throw std::runtime_error("Unable to read engine file");
    }

    // Create a runtime to deserialize the engine file.
    m_runtime = std::unique_ptr<IRuntime>{createInferRuntime(m_logger)};
    if (!m_runtime)
    {
        return false;
    }

    // Set the device index
    auto ret = cudaSetDevice(m_options.deviceIndex);
    if (ret != 0)
    {
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);
        auto errMsg = "Unable to set GPU device index to: " + std::to_string(m_options.deviceIndex) +
                      ". Note, your device has " + std::to_string(numGPUs) + " CUDA-capable GPU(s).";
        throw std::runtime_error(errMsg);
    }

    // Create an engine, a representation of the optimized model.
    m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!m_engine)
    {
        return false;
    }

    // The execution context contains all of the state associated with a particular invocation
    m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if (!m_context)
    {
        return false;
    }

    // Storage for holding the input and output buffers
    // This will be passed to TensorRT for inference
    m_buffers.resize(m_engine->getNbIOTensors());

    // Create a cuda stream
    cudaStream_t stream;
    checkCudaErrorCode(cudaStreamCreate(&stream));

    // Allocate GPU memory for input and output buffers
    m_outputLengthsFloat.clear();
    for (int i = 0; i < m_engine->getNbIOTensors(); ++i)
    {
        const auto tensorName = m_engine->getIOTensorName(i);
        m_IOTensorNames.emplace_back(tensorName);
        const auto tensorType = m_engine->getTensorIOMode(tensorName);
        const auto tensorShape = m_engine->getTensorShape(tensorName);
        if (tensorType == TensorIOMode::kINPUT)
        {
            // Allocate memory for the input
            // Allocate enough to fit the max batch size (we could end up using less later)
            checkCudaErrorCode(cudaMallocAsync(&m_buffers[i], m_options.maxBatchSize * tensorShape.d[1] * tensorShape.d[2] * tensorShape.d[3] * sizeof(float), stream));

            // Store the input dims for later use
            m_inputDims.emplace_back(tensorShape.d[1], tensorShape.d[2], tensorShape.d[3]);
        }
        else if (tensorType == TensorIOMode::kOUTPUT)
        {
            // The binding is an output
            uint32_t outputLenFloat = 1;
            m_outputDims.push_back(tensorShape);

            for (int j = 1; j < tensorShape.nbDims; ++j)
            {
                // We ignore j = 0 because that is the batch size, and we will take that into account when sizing the buffer
                outputLenFloat *= tensorShape.d[j];
            }

            m_outputLengthsFloat.push_back(outputLenFloat);
            // Now size the output buffer appropriately, taking into account the max possible batch size (although we could actually end up using less memory)
            checkCudaErrorCode(cudaMallocAsync(&m_buffers[i], outputLenFloat * m_options.maxBatchSize * sizeof(float), stream));
        }
        else
        {
            throw std::runtime_error("Error, IO Tensor is neither an input or output!");
        }
    }

    // Synchronize and destroy the cuda stream
    checkCudaErrorCode(cudaStreamSynchronize(stream));
    checkCudaErrorCode(cudaStreamDestroy(stream));

    return true;
}

bool Engine::runInference(const std::vector<std::vector<cv::cuda::GpuMat>> &inputs, std::vector<std::vector<std::vector<float>>> &featureVectors)
{
    // First we do some error checking
    if (inputs.empty() || inputs[0].empty())
    {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "Provided input vector is empty!" << std::endl;
        return false;
    }

    const auto numInputs = m_inputDims.size();
    if (inputs.size() != numInputs)
    {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "Incorrect number of inputs provided!" << std::endl;
        return false;
    }

    // Ensure the batch size does not exceed the max
    if (inputs[0].size() > static_cast<size_t>(m_options.maxBatchSize))
    {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "The batch size is larger than the model expects!" << std::endl;
        std::cout << "Model max batch size: " << m_options.maxBatchSize << std::endl;
        std::cout << "Batch size provided to call to runInference: " << inputs[0].size() << std::endl;
        return false;
    }

    const auto batchSize = static_cast<int32_t>(inputs[0].size());
    // Make sure the same batch size was provided for all inputs
    for (size_t i = 1; i < inputs.size(); ++i)
    {
        if (inputs[i].size() != static_cast<size_t>(batchSize))
        {
            std::cout << "===== Error =====" << std::endl;
            std::cout << "The batch size needs to be constant for all inputs!" << std::endl;
            return false;
        }
    }

    // Create the cuda stream that will be used for inference
    cudaStream_t inferenceCudaStream;
    checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));

    // Preprocess all the inputs
    for (size_t i = 0; i < numInputs; ++i)
    {
        const auto &batchInput = inputs[i];
        const auto &dims = m_inputDims[i];

        auto &input = batchInput[0];
        if (input.channels() != dims.d[0] ||
            input.rows != dims.d[1] ||
            input.cols != dims.d[2])
        {
            std::cout << "===== Error =====" << std::endl;
            std::cout << "Input does not have correct size!" << std::endl;
            std::cout << "Expected: (" << dims.d[0] << ", " << dims.d[1] << ", "
                      << dims.d[2] << ")" << std::endl;
            std::cout << "Got: (" << input.channels() << ", " << input.rows << ", " << input.cols << ")" << std::endl;
            std::cout << "Ensure you resize your input image to the correct size" << std::endl;
            return false;
        }

        nvinfer1::Dims4 inputDims = {batchSize, dims.d[0], dims.d[1], dims.d[2]};
        m_context->setInputShape(m_IOTensorNames[i].c_str(), inputDims); // Define the batch size

        // OpenCV reads images into memory in NHWC format, while TensorRT expects images in NCHW format.
        // The following method converts NHWC to NCHW.
        // Even though TensorRT expects NCHW at IO, during optimization, it can internally use NHWC to optimize cuda kernels
        // See: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#data-layout
        // Copy over the input data and perform the preprocessing
        auto mfloat = blobFromGpuMats(batchInput, m_subVals, m_divVals, m_normalize);
        auto *dataPointer = mfloat.ptr<void>();

        checkCudaErrorCode(cudaMemcpyAsync(m_buffers[i], dataPointer,
                                           mfloat.cols * mfloat.rows * mfloat.channels() * sizeof(float),
                                           cudaMemcpyDeviceToDevice, inferenceCudaStream));
    }

    // Ensure all dynamic bindings have been defined.
    if (!m_context->allInputDimensionsSpecified())
    {
        throw std::runtime_error("Error, not all required dimensions specified.");
    }

    // Set the address of the input and output buffers
    for (size_t i = 0; i < m_buffers.size(); ++i)
    {
        bool status = m_context->setTensorAddress(m_IOTensorNames[i].c_str(), m_buffers[i]);
        if (!status)
        {
            return false;
        }
    }

    // Run inference.
    bool status = m_context->enqueueV3(inferenceCudaStream);
    if (!status)
    {
        return false;
    }

    // Copy the outputs back to CPU
    featureVectors.clear();

    for (int batch = 0; batch < batchSize; ++batch)
    {
        // Batch
        std::vector<std::vector<float>> batchOutputs{};
        for (int32_t outputBinding = numInputs; outputBinding < m_engine->getNbBindings(); ++outputBinding)
        {
            // We start at index m_inputDims.size() to account for the inputs in our m_buffers
            std::vector<float> output;
            auto outputLenFloat = m_outputLengthsFloat[outputBinding - numInputs];
            output.resize(outputLenFloat);
            // Copy the output
            checkCudaErrorCode(cudaMemcpyAsync(output.data(), static_cast<char *>(m_buffers[outputBinding]) + (batch * sizeof(float) * outputLenFloat), outputLenFloat * sizeof(float), cudaMemcpyDeviceToHost, inferenceCudaStream));
            batchOutputs.emplace_back(std::move(output));
        }
        featureVectors.emplace_back(std::move(batchOutputs));
    }

    // Synchronize the cuda stream
    checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
    checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));
    return true;
}

cv::cuda::GpuMat Engine::blobFromGpuMats(const std::vector<cv::cuda::GpuMat> &batchInput, const std::array<float, 3> &subVals, const std::array<float, 3> &divVals, bool normalize)
{
    cv::cuda::GpuMat gpu_dst(1, batchInput[0].rows * batchInput[0].cols * batchInput.size(), CV_8UC3);

    size_t width = batchInput[0].cols * batchInput[0].rows;
    for (size_t img = 0; img < batchInput.size(); img++)
    {
        std::vector<cv::cuda::GpuMat> input_channels{
            cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[0 + width * 3 * img])),
            cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[width + width * 3 * img])),
            cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U,
                             &(gpu_dst.ptr()[width * 2 + width * 3 * img]))};
        cv::cuda::split(batchInput[img], input_channels); // HWC -> CHW
    }

    cv::cuda::GpuMat mfloat;
    if (normalize)
    {
        // [0.f, 1.f]
        gpu_dst.convertTo(mfloat, CV_32FC3, 1.f / 255.f);
    }
    else
    {
        // [0.f, 255.f]
        gpu_dst.convertTo(mfloat, CV_32FC3);
    }

    // Apply scaling and mean subtraction
    cv::cuda::subtract(mfloat, cv::Scalar(subVals[0], subVals[1], subVals[2]), mfloat, cv::noArray(), -1);
    cv::cuda::divide(mfloat, cv::Scalar(divVals[0], divVals[1], divVals[2]), mfloat, 1, -1);

    return mfloat;
}

void Engine::transformOutput(std::vector<std::vector<std::vector<float>>> &input, std::vector<std::vector<float>> &output)
{
    if (input.size() != 1)
    {
        throw std::logic_error("The feature vector has incorrect dimensions!");
    }

    output = std::move(input[0]);
}

void Engine::transformOutput(std::vector<std::vector<std::vector<float>>> &input, std::vector<float> &output)
{
    if (input.size() != 1 || input[0].size() != 1)
    {
        throw std::logic_error("The feature vector has incorrect dimensions!");
    }

    output = std::move(input[0][0]);
}

Int8EntropyCalibrator2::Int8EntropyCalibrator2(int32_t batchSize, int32_t inputW, int32_t inputH,
                                               const std::string &calibDataDirPath,
                                               const std::string &calibTableName,
                                               const std::string &inputBlobName,
                                               const std::array<float, 3> &subVals,
                                               const std::array<float, 3> &divVals,
                                               bool normalize,
                                               bool readCache)
    : m_batchSize(batchSize), m_inputW(inputW), m_inputH(inputH), m_imgIdx(0), m_calibTableName(calibTableName), m_inputBlobName(inputBlobName), m_subVals(subVals), m_divVals(divVals), m_normalize(normalize), m_readCache(readCache)
{

    // Allocate GPU memory to hold the entire batch
    m_inputCount = 3 * inputW * inputH * batchSize;
    checkCudaErrorCode(cudaMalloc(&m_deviceInput, m_inputCount * sizeof(float)));

    // Read the name of all the files in the specified directory.
    if (!doesFileExist(calibDataDirPath))
    {
        throw std::runtime_error("Error, directory at provided path does not exist: " + calibDataDirPath);
    }

    m_imgPaths = getFilesInDirectory(calibDataDirPath);
    if (m_imgPaths.size() < static_cast<size_t>(batchSize))
    {
        throw std::runtime_error("There are fewer calibration images than the specified batch size!");
    }

    // Randomize the calibration data
    auto rd = std::random_device{};
    auto rng = std::default_random_engine{rd()};
    std::shuffle(std::begin(m_imgPaths), std::end(m_imgPaths), rng);
}

int32_t Int8EntropyCalibrator2::getBatchSize() const noexcept
{
    // Return the batch size
    return m_batchSize;
}

bool Int8EntropyCalibrator2::getBatch(void **bindings, const char **names, int32_t nbBindings) noexcept
{
    // This method will read a batch of images into GPU memory, and place the pointer to the GPU memory in the bindings variable.

    if (m_imgIdx + m_batchSize > static_cast<int>(m_imgPaths.size()))
    {
        // There are not enough images left to satisfy an entire batch
        return false;
    }

    // Read the calibration images into memory for the current batch
    std::vector<cv::cuda::GpuMat> inputImgs;
    for (int i = m_imgIdx; i < m_imgIdx + m_batchSize; i++)
    {
        std::cout << "Reading image " << i << ": " << m_imgPaths[i] << std::endl;
        auto cpuImg = cv::imread(m_imgPaths[i]);
        if (cpuImg.empty())
        {
            std::cout << "Fatal error: Unable to read image at path: " << m_imgPaths[i] << std::endl;
            return false;
        }

        cv::cuda::GpuMat gpuImg;
        gpuImg.upload(cpuImg);
        cv::cuda::cvtColor(gpuImg, gpuImg, cv::COLOR_BGR2RGB);

        // TODO: Define any preprocessing code here, such as resizing
        auto resized = Engine::resizeKeepAspectRatioPadRightBottom(gpuImg, m_inputH, m_inputW);

        inputImgs.emplace_back(std::move(resized));
    }

    // Convert the batch from NHWC to NCHW
    // ALso apply normalization, scaling, and mean subtraction
    auto mfloat = Engine::blobFromGpuMats(inputImgs, m_subVals, m_divVals, m_normalize);
    auto *dataPointer = mfloat.ptr<void>();

    // Copy the GPU buffer to member variable so that it persists
    checkCudaErrorCode(cudaMemcpyAsync(m_deviceInput, dataPointer, m_inputCount * sizeof(float), cudaMemcpyDeviceToDevice));

    m_imgIdx += m_batchSize;
    if (std::string(names[0]) != m_inputBlobName)
    {
        std::cout << "Error: Incorrect input name provided!" << std::endl;
        return false;
    }
    bindings[0] = m_deviceInput;
    return true;
}

void const *Int8EntropyCalibrator2::readCalibrationCache(size_t &length) noexcept
{
    std::cout << "Searching for calibration cache: " << m_calibTableName << std::endl;
    m_calibCache.clear();
    std::ifstream input(m_calibTableName, std::ios::binary);
    input >> std::noskipws;
    if (m_readCache && input.good())
    {
        std::cout << "Reading calibration cache: " << m_calibTableName << std::endl;
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(m_calibCache));
    }
    length = m_calibCache.size();
    return length ? m_calibCache.data() : nullptr;
}

void Int8EntropyCalibrator2::writeCalibrationCache(const void *ptr, std::size_t length) noexcept
{
    std::cout << "Writing calib cache: " << m_calibTableName << " Size: " << length << " bytes" << std::endl;
    std::ofstream output(m_calibTableName, std::ios::binary);
    output.write(reinterpret_cast<const char *>(ptr), length);
}

Int8EntropyCalibrator2::~Int8EntropyCalibrator2()
{
    checkCudaErrorCode(cudaFree(m_deviceInput));
};
