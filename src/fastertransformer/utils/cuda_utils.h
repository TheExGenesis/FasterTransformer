/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "3rdparty/INIReader.h"
#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include "src/fastertransformer/utils/logger.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#ifdef SPARSITY_ENABLED
#include <cusparseLt.h>
#endif

namespace fastertransformer {

#define MAX_CONFIG_NUM 20
#define COL32_ 32
// workspace for cublas gemm : 32MB
#define CUBLAS_WORKSPACE_SIZE 33554432

typedef struct __align__(4)
{
    half x, y, z, w;
}
half4;

/* **************************** type definition ***************************** */

enum CublasDataType {
    FLOAT_DATATYPE    = 0,
    HALF_DATATYPE     = 1,
    BFLOAT16_DATATYPE = 2,
    INT8_DATATYPE     = 3
};

enum FtCudaDataType {
    FP32 = 0,
    FP16 = 1,
    BF16 = 2,
    INT8 = 3
};

enum class OperationType {
    FP32,
    FP16,
    BF16
};

/* **************************** debug tools ********************************* */
static const char* _cudaGetErrorEnum(cudaError_t error)
{
    return cudaGetErrorString(error);
}

static const char* _cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "<unknown>";
}

template<typename T>
void check(T result, char const* const func, const char* const file, int const line)
{
    if (result) {
        throw std::runtime_error(std::string("[FT][ERROR] CUDA runtime error: ") + (_cudaGetErrorEnum(result)) + " "
                                 + file + ":" + std::to_string(line) + " \n");
    }
}

#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)
#define check_cuda_error_2(val, file, line) check((val), #val, file, line)

inline void syncAndCheck(const char* const file, int const line)
{
    // When FT_DEBUG_LEVEL=DEBUG, must check error
    static char* level_name = std::getenv("FT_DEBUG_LEVEL");
    if (level_name != nullptr) {
        static std::string level = std::string(level_name);
        if (level == "DEBUG") {
            cudaDeviceSynchronize();
            cudaError_t result = cudaGetLastError();
            if (result) {
                throw std::runtime_error(std::string("[FT][ERROR] CUDA runtime error: ") + (_cudaGetErrorEnum(result))
                                         + " " + file + ":" + std::to_string(line) + " \n");
            }
            FT_LOG_DEBUG("run syncAndCheck");
        }
    }

#ifndef NDEBUG
    cudaDeviceSynchronize();
    cudaError_t result = cudaGetLastError();
    if (result) {
        throw std::runtime_error(std::string("[FT][ERROR] CUDA runtime error: ") + (_cudaGetErrorEnum(result)) + " "
                                 + file + ":" + std::to_string(line) + " \n");
    }
#endif
}

#define sync_check_cuda_error() syncAndCheck(__FILE__, __LINE__)

#define checkCUDNN(expression)                                                                                         \
    {                                                                                                                  \
        cudnnStatus_t status = (expression);                                                                           \
        if (status != CUDNN_STATUS_SUCCESS) {                                                                          \
            std::cerr << "Error on file " << __FILE__ << " line " << __LINE__ << ": " << cudnnGetErrorString(status)   \
                      << std::endl;                                                                                    \
            std::exit(EXIT_FAILURE);                                                                                   \
        }                                                                                                              \
    }

template<typename T>
void print_to_file(const T*           result,
                   const int          size,
                   const char*        file,
                   cudaStream_t       stream    = 0,
                   std::ios::openmode open_mode = std::ios::out);

template<typename T>
void print_abs_mean(const T* buf, uint size, cudaStream_t stream, std::string name = "");

template<typename T>
void print_to_screen(const T* result, const int size);

template<typename T>
void printMatrix(T* ptr, int m, int k, int stride, bool is_device_ptr);

void printMatrix(unsigned long long* ptr, int m, int k, int stride, bool is_device_ptr);
void printMatrix(int* ptr, int m, int k, int stride, bool is_device_ptr);
void printMatrix(size_t* ptr, int m, int k, int stride, bool is_device_ptr);

template<typename T>
void check_max_val(const T* result, const int size);

template<typename T>
void check_abs_mean_val(const T* result, const int size);

#define PRINT_FUNC_NAME_()                                                                                             \
    do {                                                                                                               \
        std::cout << "[FT][CALL] " << __FUNCTION__ << " " << std::endl;                                                \
    } while (0)

inline void myAssert(bool result, const char* const file, int const line, std::string info = "")
{
    if (result != true) {
        throw std::runtime_error(std::string("[FT][ERROR] ") + info + std::string(" Assertion fail: ") + file + ":"
                                 + std::to_string(line) + " \n");
    }
}

#define FT_CHECK(val) myAssert(val, __FILE__, __LINE__)
#define FT_CHECK_WITH_INFO(val, info) myAssert(val, __FILE__, __LINE__, info)

#ifdef SPARSITY_ENABLED
#define CHECK_CUSPARSE(func)                                                                                           \
    {                                                                                                                  \
        cusparseStatus_t status = (func);                                                                              \
        if (status != CUSPARSE_STATUS_SUCCESS) {                                                                       \
            throw std::runtime_error(std::string("[FT][ERROR] CUSPARSE API failed at line ")                           \
                                     + std::to_string(__LINE__) + " in file " + __FILE__ + ": "                        \
                                     + cusparseGetErrorString(status) + " " + std::to_string(status));                 \
        }                                                                                                              \
    }
#endif

/*************Time Handling**************/
class CudaTimer {
private:
    cudaEvent_t  event_start_;
    cudaEvent_t  event_stop_;
    cudaStream_t stream_;

public:
    explicit CudaTimer(cudaStream_t stream = 0)
    {
        stream_ = stream;
    }
    void start()
    {
        check_cuda_error(cudaEventCreate(&event_start_));
        check_cuda_error(cudaEventCreate(&event_stop_));
        check_cuda_error(cudaEventRecord(event_start_, stream_));
    }
    float stop()
    {
        float time;
        check_cuda_error(cudaEventRecord(event_stop_, stream_));
        check_cuda_error(cudaEventSynchronize(event_stop_));
        check_cuda_error(cudaEventElapsedTime(&time, event_start_, event_stop_));
        check_cuda_error(cudaEventDestroy(event_start_));
        check_cuda_error(cudaEventDestroy(event_stop_));
        return time;
    }
    ~CudaTimer() {}
};

static double diffTime(timeval start, timeval end)
{
    return (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
}

/* ***************************** common utils ****************************** */

inline void print_mem_usage(std::string time = "after allocation")
{
    size_t free_bytes, total_bytes;
    check_cuda_error(cudaMemGetInfo(&free_bytes, &total_bytes));
    float free  = static_cast<float>(free_bytes) / 1024.0 / 1024.0 / 1024.0;
    float total = static_cast<float>(total_bytes) / 1024.0 / 1024.0 / 1024.0;
    float used  = total - free;
    printf("%-20s: free: %5.2f GB, total: %5.2f GB, used: %5.2f GB\n", time.c_str(), free, total, used);
}

inline int getSMVersion()
{
    int device{-1};
    check_cuda_error(cudaGetDevice(&device));
    int sm_major = 0;
    int sm_minor = 0;
    check_cuda_error(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
    check_cuda_error(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device));
    return sm_major * 10 + sm_minor;
}

inline std::string getDeviceName()
{
    int device{-1};
    check_cuda_error(cudaGetDevice(&device));
    cudaDeviceProp props;
    check_cuda_error(cudaGetDeviceProperties(&props, device));
    return std::string(props.name);
}

inline int div_up(int a, int n)
{
    return (a + n - 1) / n;
}

cudaError_t getSetDevice(int i_device, int* o_device = NULL);

inline int getDevice()
{
    int current_dev_id = 0;
    check_cuda_error(cudaGetDevice(&current_dev_id));
    return current_dev_id;
}

inline int getDeviceCount()
{
    int count = 0;
    check_cuda_error(cudaGetDeviceCount(&count));
    return count;
}

template<typename T>
CublasDataType getCublasDataType()
{
    if (std::is_same<T, half>::value) {
        return HALF_DATATYPE;
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        return BFLOAT16_DATATYPE;
    }
#endif
    else if (std::is_same<T, float>::value) {
        return FLOAT_DATATYPE;
    }
    else {
        FT_CHECK(false);
        return FLOAT_DATATYPE;
    }
}

template<typename T>
cudaDataType_t getCudaDataType()
{
    if (std::is_same<T, half>::value) {
        return CUDA_R_16F;
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        return CUDA_R_16BF;
    }
#endif
    else if (std::is_same<T, float>::value) {
        return CUDA_R_32F;
    }
    else {
        FT_CHECK(false);
        return CUDA_R_32F;
    }
}

template<CublasDataType T>
struct getTypeFromCudaDataType {
    using Type = float;
};

template<>
struct getTypeFromCudaDataType<HALF_DATATYPE> {
    using Type = half;
};

#ifdef ENABLE_BF16
template<>
struct getTypeFromCudaDataType<BFLOAT16_DATATYPE> {
    using Type = __nv_bfloat16;
};
#endif

FtCudaDataType getModelFileType(std::string ini_file, std::string section_name);

// clang-format off
template<typename T> struct packed_type;
template <>          struct packed_type<float>         { using type = float; }; // we don't need to pack float by default
template <>          struct packed_type<half>          { using type = half2; };

#ifdef ENABLE_BF16
template<>
struct packed_type<__nv_bfloat16> {
    using type = __nv_bfloat162;
};
#endif

template<typename T> struct num_elems;
template <>          struct num_elems<float>           { static constexpr int value = 1; };
template <>          struct num_elems<float2>          { static constexpr int value = 2; };
template <>          struct num_elems<float4>          { static constexpr int value = 4; };
template <>          struct num_elems<half>            { static constexpr int value = 1; };
template <>          struct num_elems<half2>           { static constexpr int value = 2; };
#ifdef ENABLE_BF16
template <>          struct num_elems<__nv_bfloat16>   { static constexpr int value = 1; };
template <>          struct num_elems<__nv_bfloat162>  { static constexpr int value = 2; };
#endif

template<typename T, int num> struct packed_as;
template<typename T>          struct packed_as<T, 1>              { using type = T; };
template<>                    struct packed_as<half,  2>          { using type = half2; };
template<>                    struct packed_as<float,  2>         { using type = float2; };
template<>                    struct packed_as<int8_t, 2>         { using type = int16_t; };
template<>                    struct packed_as<int32_t, 2>        { using type = int2; };
#ifdef ENABLE_BF16
template<>
struct packed_as<__nv_bfloat16, 2> {
    using type = __nv_bfloat162;
};
#endif

inline __device__ float2 operator*(float2 a, float2 b) { return make_float2(a.x * b.x, a.y * b.y); }
inline __device__ float2 operator*(float2 a, float  b) { return make_float2(a.x * b, a.y * b); }
// clang-format on

/* ************************** end of common utils ************************** */
}  // namespace fastertransformer
