/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
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

// Ignore CUTLASS warnings about type punning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"

#include "cutlass_extensions/compute_occupancy.h"
#include "cutlass_extensions/epilogue_helpers.h"
#include "cutlass_extensions/gemm/kernel/default_fpA_intB_traits.h"
#include "cutlass_extensions/gemm/kernel/moe_cutlass_kernel.h"
#include "cutlass_extensions/gemm/threadblock/default_mma.h"

#pragma GCC diagnostic pop

#include "src/fastertransformer/kernels/cutlass_kernels/cutlass_heuristic.h"
#include "src/fastertransformer/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <sstream>

namespace fastertransformer {

// ============================= Variable batched Gemm things ===========================
template<typename T,
         typename WeightType,
         typename arch,
         typename EpilogueTag,
         typename ThreadblockShape,
         typename WarpShape,
         int Stages>
void generic_moe_gemm_kernelLauncher(const T*          A,
                                     const WeightType* B,
                                     const T*          weight_scales,
                                     const T*          biases,
                                     T*                C,
                                     int64_t*          total_rows_before_expert,
                                     int64_t           gemm_n,
                                     int64_t           gemm_k,
                                     int               num_experts,
                                     CutlassGemmConfig gemm_config,
                                     const int         multi_processor_count,
                                     cudaStream_t      stream,
                                     int*              kernel_occupancy = nullptr)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
#ifdef BUILD_CUTLASS_MOE

    if (gemm_config.split_k_style != SplitKStyle::NO_SPLIT_K) {
        throw std::runtime_error("[FT Error][MoeGemm] Grouped gemm does not support split-k");
    }

#ifdef ENABLE_BF16
    static_assert(cutlass::platform::is_same<T, __nv_bfloat16>::value || cutlass::platform::is_same<T, half>::value
                      || cutlass::platform::is_same<T, float>::value,
                  "Specialized for bfloat16, half, float");
#else
    static_assert(cutlass::platform::is_same<T, half>::value || cutlass::platform::is_same<T, float>::value,
                  "Specialized for half, float");
#endif

    static_assert(cutlass::platform::is_same<T, WeightType>::value
                      || cutlass::platform::is_same<WeightType, uint8_t>::value
                      || cutlass::platform::is_same<WeightType, cutlass::uint4b_t>::value,
                  "");

    // The cutlass type for the input elements. This is needed to convert to cutlass::half_t if necessary.
    using ElementType_ =
        typename cutlass::platform::conditional<cutlass::platform::is_same<T, half>::value, cutlass::half_t, T>::type;
#ifdef ENABLE_BF16
    using ElementType =
        typename cutlass::platform::conditional<cutlass::platform::is_same<ElementType_, __nv_bfloat16>::value,
                                                cutlass::bfloat16_t,
                                                ElementType_>::type;
#else
    using ElementType       = ElementType_;
#endif

    using CutlassWeightType_ = typename cutlass::platform::
        conditional<cutlass::platform::is_same<WeightType, half>::value, cutlass::half_t, WeightType>::type;
#ifdef ENABLE_BF16
    using CutlassWeightType =
        typename cutlass::platform::conditional<cutlass::platform::is_same<CutlassWeightType_, __nv_bfloat16>::value,
                                                cutlass::bfloat16_t,
                                                CutlassWeightType_>::type;
#else
    using CutlassWeightType = CutlassWeightType_;
#endif

    // We need separate config for each architecture since we will target different tensorcore instructions. For float,
    // we do not target TCs.
    using MixedGemmArchTraits = cutlass::gemm::kernel::MixedGemmArchTraits<ElementType, CutlassWeightType, arch>;
    using ElementAccumulator  = typename MixedGemmArchTraits::AccType;

    using EpilogueOp =
        typename Epilogue<ElementType, MixedGemmArchTraits::ElementsPerAccessC, ElementAccumulator, EpilogueTag>::Op;

    // Finally, set up the kernel.
    using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        ElementType,
        cutlass::layout::RowMajor,
        cutlass::ComplexTransform::kNone,
        MixedGemmArchTraits::ElementsPerAccessA,
        CutlassWeightType,
        typename MixedGemmArchTraits::LayoutB,
        cutlass::ComplexTransform::kNone,
        MixedGemmArchTraits::ElementsPerAccessB,
        ElementType,
        cutlass::layout::RowMajor,
        ElementAccumulator,
        typename MixedGemmArchTraits::OperatorClass,
        arch,
        ThreadblockShape,
        WarpShape,
        typename MixedGemmArchTraits::InstructionShape,
        EpilogueOp,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
        Stages,
        cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
        typename MixedGemmArchTraits::Operator>::GemmKernel;

    using GemmKernel = cutlass::gemm::kernel::MoeFCGemm<typename GemmKernel_::Mma,
                                                        typename GemmKernel_::Epilogue,
                                                        typename GemmKernel_::ThreadblockSwizzle,
                                                        arch,  // Ensure top level arch is used for dispatch
                                                        GemmKernel_::kGroupScheduleMode>;

    using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

    if (kernel_occupancy != nullptr) {
        *kernel_occupancy = compute_occupancy_for_kernel<GemmKernel>();
        ;
        return;
    }
    int occupancy = std::min(2, GemmGrouped::maximum_active_blocks());
    if (occupancy == 0) {
        throw std::runtime_error(
            "[FT Error][MoE Runner] GPU lacks the shared memory resources to run GroupedGEMM kernel");
    }
    const int threadblock_count = multi_processor_count * occupancy;

    typename EpilogueOp::Params epilogue_op(ElementAccumulator(1.f), ElementAccumulator(0.f));

    typename GemmGrouped::Arguments args(num_experts,
                                         threadblock_count,
                                         epilogue_op,
                                         reinterpret_cast<const ElementType*>(A),
                                         reinterpret_cast<const CutlassWeightType*>(B),
                                         reinterpret_cast<const ElementType*>(weight_scales),
                                         reinterpret_cast<const ElementType*>(biases),
                                         reinterpret_cast<ElementType*>(C),
                                         total_rows_before_expert,
                                         gemm_n,
                                         gemm_k);

    GemmGrouped gemm;

    auto can_implement = gemm.can_implement(args);
    if (can_implement != cutlass::Status::kSuccess) {
        std::string err_msg =
            "MoEFC kernel will fail for params. Error: " + std::string(cutlassGetStatusString(can_implement));
        throw std::runtime_error("[FT Error][MoE Runner] " + err_msg);
    }

    auto init_status = gemm.initialize(args);
    if (init_status != cutlass::Status::kSuccess) {
        std::string err_msg = "Failed to initialize cutlass variable batched gemm. Error: "
                              + std::string(cutlassGetStatusString(init_status));
        throw std::runtime_error("[FT Error][MoE Runner] " + err_msg);
    }

    auto run_status = gemm.run(stream);
    if (run_status != cutlass::Status::kSuccess) {
        std::string err_msg =
            "Failed to run cutlass variable batched gemm. Error: " + std::string(cutlassGetStatusString(run_status));
        throw std::runtime_error("[FT Error][MoE Runner] " + err_msg);
    }
#else
    throw std::runtime_error(
        "[FT Error][MoE Runner] FasterTransformer was built was MoE support off. Please rebuild with cmake option -DBUILD_CUTLASS_MOE=ON");
#endif
}

template<typename T,
         typename WeightType,
         typename arch,
         typename EpilogueTag,
         typename ThreadblockShape,
         typename WarpShape,
         int Stages,
         typename Enable = void>
struct dispatch_stages {
    static void dispatch(const T*          A,
                         const WeightType* B,
                         const T*          weight_scales,
                         const T*          biases,
                         T*                C,
                         int64_t*          total_rows_before_expert,
                         int64_t           gemm_n,
                         int64_t           gemm_k,
                         int               num_experts,
                         CutlassGemmConfig gemm_config,
                         int               multi_processor_count,
                         cudaStream_t      stream,
                         int*              occupancy = nullptr)
    {

        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        std::string err_msg = "Cutlass fpA_intB gemm. Not instantiates for arch "
                              + std::to_string(arch::kMinComputeCapability) + " with stages set to "
                              + std::to_string(Stages);
        throw std::runtime_error("[FT Error][dispatch_stages::dispatch] " + err_msg);
    }
};

template<typename T,
         typename WeightType,
         typename arch,
         typename EpilogueTag,
         typename ThreadblockShape,
         typename WarpShape>
struct dispatch_stages<T, WeightType, arch, EpilogueTag, ThreadblockShape, WarpShape, 2> {
    static void dispatch(const T*          A,
                         const WeightType* B,
                         const T*          weight_scales,
                         const T*          biases,
                         T*                C,
                         int64_t*          total_rows_before_expert,
                         int64_t           gemm_n,
                         int64_t           gemm_k,
                         int               num_experts,
                         CutlassGemmConfig gemm_config,
                         int               multi_processor_count,
                         cudaStream_t      stream,
                         int*              occupancy = nullptr)
    {

        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        generic_moe_gemm_kernelLauncher<T, WeightType, arch, EpilogueTag, ThreadblockShape, WarpShape, 2>(
            A,
            B,
            weight_scales,
            biases,
            C,
            total_rows_before_expert,
            gemm_n,
            gemm_k,
            num_experts,
            gemm_config,
            multi_processor_count,
            stream,
            occupancy);
    }
};

template<typename T,
         typename WeightType,
         typename EpilogueTag,
         typename ThreadblockShape,
         typename WarpShape,
         int Stages>
struct dispatch_stages<T,
                       WeightType,
                       cutlass::arch::Sm80,
                       EpilogueTag,
                       ThreadblockShape,
                       WarpShape,
                       Stages,
                       typename std::enable_if<(Stages > 2)>::type> {
    static void dispatch(const T*          A,
                         const WeightType* B,
                         const T*          weight_scales,
                         const T*          biases,
                         T*                C,
                         int64_t*          total_rows_before_expert,
                         int64_t           gemm_n,
                         int64_t           gemm_k,
                         int               num_experts,
                         CutlassGemmConfig gemm_config,
                         int               multi_processor_count,
                         cudaStream_t      stream,
                         int*              occupancy = nullptr)
    {

        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        generic_moe_gemm_kernelLauncher<T,
                                        WeightType,
                                        cutlass::arch::Sm80,
                                        EpilogueTag,
                                        ThreadblockShape,
                                        WarpShape,
                                        Stages>(A,
                                                B,
                                                weight_scales,
                                                biases,
                                                C,
                                                total_rows_before_expert,
                                                gemm_n,
                                                gemm_k,
                                                num_experts,
                                                gemm_config,
                                                multi_processor_count,
                                                stream,
                                                occupancy);
    }
};

template<typename T,
         typename WeightType,
         typename arch,
         typename EpilogueTag,
         typename ThreadblockShape,
         typename WarpShape>
void dispatch_gemm_config(const T*          A,
                          const WeightType* B,
                          const T*          weight_scales,
                          const T*          biases,
                          T*                C,
                          int64_t*          total_rows_before_expert,
                          int64_t           gemm_n,
                          int64_t           gemm_k,
                          int               num_experts,
                          CutlassGemmConfig gemm_config,
                          int               multi_processor_count,
                          cudaStream_t      stream,
                          int*              occupancy = nullptr)
{

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    switch (gemm_config.stages) {
        case 2:
            using DispatcherStages2 = dispatch_stages<T, WeightType, arch, EpilogueTag, ThreadblockShape, WarpShape, 2>;
            DispatcherStages2::dispatch(A,
                                        B,
                                        weight_scales,
                                        biases,
                                        C,
                                        total_rows_before_expert,
                                        gemm_n,
                                        gemm_k,
                                        num_experts,
                                        gemm_config,
                                        multi_processor_count,
                                        stream,
                                        occupancy);
            break;
        case 3:
            using DispatcherStages3 = dispatch_stages<T, WeightType, arch, EpilogueTag, ThreadblockShape, WarpShape, 3>;
            DispatcherStages3::dispatch(A,
                                        B,
                                        weight_scales,
                                        biases,
                                        C,
                                        total_rows_before_expert,
                                        gemm_n,
                                        gemm_k,
                                        num_experts,
                                        gemm_config,
                                        multi_processor_count,
                                        stream,
                                        occupancy);
            break;
        case 4:
            using DispatcherStages4 = dispatch_stages<T, WeightType, arch, EpilogueTag, ThreadblockShape, WarpShape, 4>;
            DispatcherStages4::dispatch(A,
                                        B,
                                        weight_scales,
                                        biases,
                                        C,
                                        total_rows_before_expert,
                                        gemm_n,
                                        gemm_k,
                                        num_experts,
                                        gemm_config,
                                        multi_processor_count,
                                        stream,
                                        occupancy);
            break;
        default:
            std::string err_msg = "dispatch_gemm_config does not support stages " + std::to_string(gemm_config.stages);
            throw std::runtime_error("[FT Error][MoE][dispatch_gemm_config] " + err_msg);
            break;
    }
}

// This overload will handle tensorop gemms. It is disabled via SFINAE for fp32.
// This overload is only enabled when T == WeightType.
template<typename T,
         typename WeightType,
         typename arch,
         typename EpilogueTag,
         typename std::enable_if<!std::is_same<T, float>::value && std::is_same<T, WeightType>::value>::type* = nullptr>
void dispatch_moe_gemm_to_cutlass(const T*          A,
                                  const WeightType* B,
                                  const T*          weight_scales,
                                  const T*          biases,
                                  T*                C,
                                  int64_t*          total_rows_before_expert,
                                  int64_t           total_rows,
                                  int64_t           gemm_n,
                                  int64_t           gemm_k,
                                  int               num_experts,
                                  CutlassGemmConfig gemm_config,
                                  int               sm_version,
                                  int               multi_processor_count,
                                  cudaStream_t      stream,
                                  int*              occupancy = nullptr)
{

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    switch (gemm_config.tile_config) {
        case CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
            dispatch_gemm_config<T,
                                 WeightType,
                                 arch,
                                 EpilogueTag,
                                 cutlass::gemm::GemmShape<32, 128, 64>,
                                 cutlass::gemm::GemmShape<32, 32, 64>>(A,
                                                                       B,
                                                                       weight_scales,
                                                                       biases,
                                                                       C,
                                                                       total_rows_before_expert,
                                                                       gemm_n,
                                                                       gemm_k,
                                                                       num_experts,
                                                                       gemm_config,
                                                                       multi_processor_count,
                                                                       stream,
                                                                       occupancy);
            break;
        case CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64:
            dispatch_gemm_config<T,
                                 WeightType,
                                 arch,
                                 EpilogueTag,
                                 cutlass::gemm::GemmShape<64, 128, 64>,
                                 cutlass::gemm::GemmShape<32, 64, 64>>(A,
                                                                       B,
                                                                       weight_scales,
                                                                       biases,
                                                                       C,
                                                                       total_rows_before_expert,
                                                                       gemm_n,
                                                                       gemm_k,
                                                                       num_experts,
                                                                       gemm_config,
                                                                       multi_processor_count,
                                                                       stream,
                                                                       occupancy);
            break;
        case CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64:
            dispatch_gemm_config<T,
                                 WeightType,
                                 arch,
                                 EpilogueTag,
                                 cutlass::gemm::GemmShape<128, 128, 64>,
                                 cutlass::gemm::GemmShape<64, 32, 64>>(A,
                                                                       B,
                                                                       weight_scales,
                                                                       biases,
                                                                       C,
                                                                       total_rows_before_expert,
                                                                       gemm_n,
                                                                       gemm_k,
                                                                       num_experts,
                                                                       gemm_config,
                                                                       multi_processor_count,
                                                                       stream,
                                                                       occupancy);
            break;
        case CutlassTileConfig::Undefined:
            throw std::runtime_error("[FT Error][dispatch_moe_gemm_to_cutlass] gemm config undefined.");
            break;
        case CutlassTileConfig::ChooseWithHeuristic:
            throw std::runtime_error(
                "[FT Error][dispatch_moe_gemm_to_cutlass] gemm config should have already been set by heuristic.");
            break;
        default:
            throw std::runtime_error(
                "[FT Error][dispatch_moe_gemm_to_cutlass] Config is invalid for same type MoE tensorop GEMM.");
            break;
    }
}

// Tensorop GEMM overload
// Overload for quantize MoE GEMMs. We disable some warp configs here since they will not be used and we can improve
// compile time
template<
    typename T,
    typename WeightType,
    typename arch,
    typename EpilogueTag,
    typename std::enable_if<!std::is_same<T, float>::value && !std::is_same<T, WeightType>::value>::type* = nullptr>
void dispatch_moe_gemm_to_cutlass(const T*          A,
                                  const WeightType* B,
                                  const T*          weight_scales,
                                  const T*          biases,
                                  T*                C,
                                  int64_t*          total_rows_before_expert,
                                  int64_t           total_rows,
                                  int64_t           gemm_n,
                                  int64_t           gemm_k,
                                  int               num_experts,
                                  CutlassGemmConfig gemm_config,
                                  int               sm_version,
                                  int               multi_processor_count,
                                  cudaStream_t      stream,
                                  int*              occupancy = nullptr)
{

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    switch (gemm_config.tile_config) {
        case CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
            dispatch_gemm_config<T,
                                 WeightType,
                                 arch,
                                 EpilogueTag,
                                 cutlass::gemm::GemmShape<32, 128, 64>,
                                 cutlass::gemm::GemmShape<32, 32, 64>>(A,
                                                                       B,
                                                                       weight_scales,
                                                                       biases,
                                                                       C,
                                                                       total_rows_before_expert,
                                                                       gemm_n,
                                                                       gemm_k,
                                                                       num_experts,
                                                                       gemm_config,
                                                                       multi_processor_count,
                                                                       stream,
                                                                       occupancy);
            break;
        case CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64:
            dispatch_gemm_config<T,
                                 WeightType,
                                 arch,
                                 EpilogueTag,
                                 cutlass::gemm::GemmShape<64, 128, 64>,
                                 cutlass::gemm::GemmShape<64, 32, 64>>(A,
                                                                       B,
                                                                       weight_scales,
                                                                       biases,
                                                                       C,
                                                                       total_rows_before_expert,
                                                                       gemm_n,
                                                                       gemm_k,
                                                                       num_experts,
                                                                       gemm_config,
                                                                       multi_processor_count,
                                                                       stream,
                                                                       occupancy);
            break;
        case CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64:
            dispatch_gemm_config<T,
                                 WeightType,
                                 arch,
                                 EpilogueTag,
                                 cutlass::gemm::GemmShape<128, 128, 64>,
                                 cutlass::gemm::GemmShape<128, 32, 64>>(A,
                                                                        B,
                                                                        weight_scales,
                                                                        biases,
                                                                        C,
                                                                        total_rows_before_expert,
                                                                        gemm_n,
                                                                        gemm_k,
                                                                        num_experts,
                                                                        gemm_config,
                                                                        multi_processor_count,
                                                                        stream,
                                                                        occupancy);
            break;
        case CutlassTileConfig::Undefined:
            throw std::runtime_error("[FT Error][dispatch_moe_gemm_to_cutlass] gemm config undefined.");
            break;
        case CutlassTileConfig::ChooseWithHeuristic:
            throw std::runtime_error(
                "[FT Error][dispatch_moe_gemm_to_cutlass] gemm config should have already been set by heuristic.");
            break;
        default:
            throw std::runtime_error(
                "[FT Error][dispatch_moe_gemm_to_cutlass] Config is invalid for mixed type tensorop GEMM.");
            break;
    }
}

// This overload will handle simt gemms. It is disabled via SFINAE for tensorop.
template<typename T,
         typename WeightType,
         typename arch,
         typename EpilogueTag,
         typename std::enable_if<std::is_same<T, float>::value>::type* = nullptr>
void dispatch_moe_gemm_to_cutlass(const T*          A,
                                  const WeightType* B,
                                  const T*          weight_scales,
                                  const T*          biases,
                                  T*                C,
                                  int64_t*          total_rows_before_expert,
                                  int64_t           total_rows,
                                  int64_t           gemm_n,
                                  int64_t           gemm_k,
                                  int               num_experts,
                                  CutlassGemmConfig gemm_config,
                                  int               sm_version,
                                  int               multi_processor_count,
                                  cudaStream_t      stream,
                                  int*              occupancy = nullptr)
{

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    switch (gemm_config.tile_config) {
        case CutlassTileConfig::CtaShape128x128x8_WarpShape64x64x8:
            dispatch_gemm_config<T,
                                 WeightType,
                                 arch,
                                 EpilogueTag,
                                 cutlass::gemm::GemmShape<128, 128, 8>,
                                 cutlass::gemm::GemmShape<64, 64, 8>>(A,
                                                                      B,
                                                                      weight_scales,
                                                                      biases,
                                                                      C,
                                                                      total_rows_before_expert,
                                                                      gemm_n,
                                                                      gemm_k,
                                                                      num_experts,
                                                                      gemm_config,
                                                                      multi_processor_count,
                                                                      stream,
                                                                      occupancy);
            break;
        case CutlassTileConfig::Undefined:
            throw std::runtime_error("[FT Error][dispatch_moe_gemm_to_cutlass][SIMT] gemm config undefined.");
            break;
        case CutlassTileConfig::ChooseWithHeuristic:
            throw std::runtime_error(
                "[FT Error][dispatch_moe_gemm_to_cutlass][SIMT] gemm config should have already been set by heuristic.");
            break;
        default:
            throw std::runtime_error(
                "[FT Error][dispatch_moe_gemm_to_cutlass][SIMT] Unsupported config for float MoE gemm.");
            break;
    }
}

template<typename T, typename WeightType>
MoeGemmRunner<T, WeightType>::MoeGemmRunner()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    int device{-1};
    check_cuda_error(cudaGetDevice(&device));
    sm_ = getSMVersion();
    check_cuda_error(cudaDeviceGetAttribute(&multi_processor_count_, cudaDevAttrMultiProcessorCount, device));
}

template<typename T, typename WeightType>
template<typename EpilogueTag>
void MoeGemmRunner<T, WeightType>::dispatch_to_arch<EpilogueTag>(const T*          A,
                                                                 const WeightType* B,
                                                                 const T*          weight_scales,
                                                                 const T*          biases,
                                                                 T*                C,
                                                                 int64_t*          total_rows_before_expert,
                                                                 int64_t           total_rows,
                                                                 int64_t           gemm_n,
                                                                 int64_t           gemm_k,
                                                                 int               num_experts,
                                                                 CutlassGemmConfig gemm_config,
                                                                 cudaStream_t      stream,
                                                                 int*              occupancy)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (sm_ >= 70 && sm_ < 75) {
        dispatch_moe_gemm_to_cutlass<T, WeightType, cutlass::arch::Sm70, EpilogueTag>(A,
                                                                                      B,
                                                                                      weight_scales,
                                                                                      biases,
                                                                                      C,
                                                                                      total_rows_before_expert,
                                                                                      total_rows,
                                                                                      gemm_n,
                                                                                      gemm_k,
                                                                                      num_experts,
                                                                                      gemm_config,
                                                                                      sm_,
                                                                                      multi_processor_count_,
                                                                                      stream,
                                                                                      occupancy);
    }
    else if (sm_ >= 75 && sm_ < 80) {
        dispatch_moe_gemm_to_cutlass<T, WeightType, cutlass::arch::Sm75, EpilogueTag>(A,
                                                                                      B,
                                                                                      weight_scales,
                                                                                      biases,
                                                                                      C,
                                                                                      total_rows_before_expert,
                                                                                      total_rows,
                                                                                      gemm_n,
                                                                                      gemm_k,
                                                                                      num_experts,
                                                                                      gemm_config,
                                                                                      sm_,
                                                                                      multi_processor_count_,
                                                                                      stream,
                                                                                      occupancy);
    }
    else if (sm_ >= 80 && sm_ < 90) {
        dispatch_moe_gemm_to_cutlass<T, WeightType, cutlass::arch::Sm80, EpilogueTag>(A,
                                                                                      B,
                                                                                      weight_scales,
                                                                                      biases,
                                                                                      C,
                                                                                      total_rows_before_expert,
                                                                                      total_rows,
                                                                                      gemm_n,
                                                                                      gemm_k,
                                                                                      num_experts,
                                                                                      gemm_config,
                                                                                      sm_,
                                                                                      multi_processor_count_,
                                                                                      stream,
                                                                                      occupancy);
    }
    else {
        throw std::runtime_error("[FT Error][MoE][GEMM Dispatch] Arch unsupported for MoE GEMM");
    }
}

template<typename T, typename WeightType>
template<typename EpilogueTag>
void MoeGemmRunner<T, WeightType>::run_gemm<EpilogueTag>(const T*          A,
                                                         const WeightType* B,
                                                         const T*          weight_scales,
                                                         const T*          biases,
                                                         T*                C,
                                                         int64_t*          total_rows_before_expert,
                                                         int64_t           total_rows,
                                                         int64_t           gemm_n,
                                                         int64_t           gemm_k,
                                                         int               num_experts,
                                                         cudaStream_t      stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    static constexpr bool          is_weight_only    = !std::is_same<T, WeightType>::value;
    static constexpr bool          only_simt_configs = std::is_same<T, float>::value;
    std::vector<CutlassGemmConfig> candidate_configs = get_candidate_configs(sm_, is_weight_only, only_simt_configs);
    std::vector<int>               occupancies(candidate_configs.size());

    for (size_t ii = 0; ii < candidate_configs.size(); ++ii) {
        dispatch_to_arch<EpilogueTag>(A,
                                      B,
                                      weight_scales,
                                      biases,
                                      C,
                                      total_rows_before_expert,
                                      total_rows,
                                      gemm_n,
                                      gemm_k,
                                      num_experts,
                                      candidate_configs[ii],
                                      stream,
                                      &occupancies[ii]);
    }

    static constexpr int workspace_bytes = 0;  // No workspace for MoE GEMMs.
    static constexpr int split_k_limit   = 1;  // MoE GEMM does not support split-k.
    CutlassGemmConfig    chosen_config   = estimate_best_config_from_occupancies(candidate_configs,
                                                                            occupancies,
                                                                            total_rows,
                                                                            gemm_n,
                                                                            gemm_k,
                                                                            num_experts,
                                                                            split_k_limit,
                                                                            workspace_bytes,
                                                                            multi_processor_count_,
                                                                            is_weight_only);

    dispatch_to_arch<EpilogueTag>(A,
                                  B,
                                  weight_scales,
                                  biases,
                                  C,
                                  total_rows_before_expert,
                                  total_rows,
                                  gemm_n,
                                  gemm_k,
                                  num_experts,
                                  chosen_config,
                                  stream);
}

template<typename T, typename WeightType>
void MoeGemmRunner<T, WeightType>::moe_gemm_bias_act(const T*          A,
                                                     const WeightType* B,
                                                     const T*          weight_scales,
                                                     const T*          biases,
                                                     T*                C,
                                                     int64_t*          total_rows_before_expert,
                                                     int64_t           total_rows,
                                                     int64_t           gemm_n,
                                                     int64_t           gemm_k,
                                                     int               num_experts,
                                                     ActivationType    activation_type,
                                                     cudaStream_t      stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    switch (activation_type) {
        case ActivationType::Relu:
            run_gemm<EpilogueOpBiasReLU>(A,
                                         B,
                                         weight_scales,
                                         biases,
                                         C,
                                         total_rows_before_expert,
                                         total_rows,
                                         gemm_n,
                                         gemm_k,
                                         num_experts,
                                         stream);
            break;
        case ActivationType::Gelu:
            run_gemm<EpilogueOpBiasFtGelu>(A,
                                           B,
                                           weight_scales,
                                           biases,
                                           C,
                                           total_rows_before_expert,
                                           total_rows,
                                           gemm_n,
                                           gemm_k,
                                           num_experts,
                                           stream);
            break;
        case ActivationType::Silu:
            run_gemm<EpilogueOpBiasSilu>(A,
                                         B,
                                         weight_scales,
                                         biases,
                                         C,
                                         total_rows_before_expert,
                                         total_rows,
                                         gemm_n,
                                         gemm_k,
                                         num_experts,
                                         stream);
            break;
        case ActivationType::Identity:
            run_gemm<EpilogueOpBias>(A,
                                     B,
                                     weight_scales,
                                     biases,
                                     C,
                                     total_rows_before_expert,
                                     total_rows,
                                     gemm_n,
                                     gemm_k,
                                     num_experts,
                                     stream);
            break;
        case ActivationType::InvalidType:
            FT_CHECK_WITH_INFO(false, "Activation type for fpA_intB must be valid.");
            break;
        default: {
            if (isGatedActivation(activation_type)) {
                FT_CHECK_WITH_INFO(false, "Fused gated activations not supported");
            }
            else {
                FT_CHECK_WITH_INFO(false, "Invalid activation type.");
            }
        }
    }
}

template<typename T, typename WeightType>
void MoeGemmRunner<T, WeightType>::moe_gemm(const T*          A,
                                            const WeightType* B,
                                            const T*          weight_scales,
                                            T*                C,
                                            int64_t*          total_rows_before_expert,
                                            int64_t           total_rows,
                                            int64_t           gemm_n,
                                            int64_t           gemm_k,
                                            int               num_experts,
                                            cudaStream_t      stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    run_gemm<EpilogueOpNoBias>(
        A, B, weight_scales, nullptr, C, total_rows_before_expert, total_rows, gemm_n, gemm_k, num_experts, stream);
}

}  // namespace fastertransformer