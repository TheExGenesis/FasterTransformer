# Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.8)

add_library(WhisperDecoder STATIC WhisperDecoder.cc WhisperDecoderLayerWeight.cc)
set_property(TARGET WhisperDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET WhisperDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(WhisperDecoder PUBLIC -lcudart cublasMMWrapper TensorParallelDecoderSelfAttentionLayer
                    TensorParallelDecoderCrossAttentionLayer TensorParallelReluFfnLayer TensorParallelSiluFfnLayer
                    layernorm_kernels add_residual_kernels nccl_utils memory_utils tensor)

add_library(WhisperDecoding STATIC WhisperDecoding.cc WhisperDecodingWeight.cc)
set_property(TARGET WhisperDecoding PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET WhisperDecoding PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(WhisperDecoding PUBLIC -lcudart cublasMMWrapper WhisperDecoder bert_preprocess_kernels
                                        decoding_kernels DynamicDecodeLayer BaseBeamSearchLayer 
                                        beam_search_topk_kernels gpt_kernels tensor)

add_library(WhisperEncoder STATIC WhisperEncoder.cc WhisperEncoderWeight.cc WhisperEncoderLayerWeight.cc)
set_property(TARGET WhisperEncoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET WhisperEncoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(WhisperEncoder PUBLIC -lcudart bert_preprocess_kernels cublasMMWrapper 
                        TensorParallelUnfusedAttentionLayer FusedAttentionLayer TensorParallelReluFfnLayer
                        TensorParallelGeluFfnLayer TensorParallelSiluFfnLayer layernorm_kernels add_residual_kernels nccl_utils tensor)
