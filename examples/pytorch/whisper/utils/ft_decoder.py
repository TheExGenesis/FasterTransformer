# %%


# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist

USE_CACHE_BATCH_MAJOR_ATTENTION = True

def get_op_cache_config(size_per_head, is_fp16):
    x = 8 if is_fp16 else 4
    use_batch_major_op_cache = True if USE_CACHE_BATCH_MAJOR_ATTENTION == True and \
                                       size_per_head % x == 0 \
                                    else False
    x = x if use_batch_major_op_cache else 1
    return use_batch_major_op_cache, x

class FTWhisperDecoderWeight(object):
    def __init__(
            self,
            config,
            tensor_para_size,
            pipeline_para_size,
            *,
            whisper_with_bias=True,
            mwhisper=True,
            use_gated_activation=False,
            position_embedding_type=1,
            weight_data_type
    ):
        self.config = config
        self.num_layer = config.decoder_layers
        self.tensor_para_size = tensor_para_size
        self.pipeline_para_size = pipeline_para_size
        self.whisper_with_bias = whisper_with_bias
        self.mwhisper = mwhisper
        self.use_gated_activation = use_gated_activation
        self.position_embedding_type = position_embedding_type
        self.real_weights_num = 24  # assume all weights are allocated and converted to specific data type
        self.weight_data_type = weight_data_type
        self.w = []
        self.use_mpi = dist.is_mpi_available()

        if self.use_mpi:
            try:
                dist.init_process_group(backend='mpi')
            except:
                print("[INFO] WARNING: Exception occurred in dist.init_process_group(backend = 'mpi'). Maybe the process group has been initialized somewhere else.")
        else:
            print("[INFO] MPI is not available in this PyTorch build.")
            assert tensor_para_size == 1, "[FATAL] MPI is required for tensor_para_size > 1."
            assert pipeline_para_size == 1, "[FATAL] MPI is required for pipeline_para_size > 1."

        self.rank = dist.get_rank() if self.use_mpi else 0
        self.device_count = torch.cuda.device_count()
        self.device = self.rank % self.device_count
        torch.cuda.set_device(self.device)

        world_size = dist.get_world_size() if self.use_mpi else 1
        assert world_size == tensor_para_size * \
            pipeline_para_size, "[ERROR] world_size != tensor_para_size * pipeline_para_size"
        self.tensor_para_rank = self.rank % self.tensor_para_size
        self.pipeline_para_rank = self.rank // self.tensor_para_size
    
    def load_from_model(self, model):
        '''Only applies to HuggingFace models. 
        Weight loading order: PyTorch tensor order should conform to src/fastertransformer/th_op/WhisperDecodingOp.h:FasterTransformerWhisperDecoding. For per-layer weights, the tensor is a stack of the weight across all layers.
        '''
        start_layer = self.pipeline_para_rank * self.num_layer // self.pipeline_para_size
        end_layer = (self.pipeline_para_rank + 1) * self.num_layer // self.pipeline_para_size

        np_weight_dtype = self.weight_data_type
        torch_weight_dtype = {np.float32: torch.float32, np.float16: torch.float16}[np_weight_dtype]

        weight_dict = {}
        qkv_weight_tmp = ["q", "k", "v"] # must respect this order
        qkv_weight_len = 0
        qkv_bias_tmp = ["q", torch.zeros((model.config.d_model), dtype=torch_weight_dtype, device=model.device), "v"]
        qkv_bias_len = 1
        for name, param in model.state_dict().items():
            if name.find(".encoder.") != -1:
                continue
            name = name.replace("model.", "")
            if param.dim() == 2:
                param_t = param.transpose(1, 0)
            elif param.dim() == 1:
                param_t = param
            else:
                assert False, f"The dimension of param {name} should be 2"
            if name.find("decoder.layers") != -1:
                if name.find(".self_attn.q_proj.weight") != -1 or name.find(".self_attn.k_proj.weight") != -1 or name.find(".self_attn.v_proj.weight") != -1:
                    qkv_weight_tmp[0 if "q_proj" in name else 1 if "k_proj" in name else 2] = param_t # qkv order in weight dict is not guaranteed
                    qkv_weight_len += 1
                    if qkv_weight_len == 3:
                        qkv_weight = torch.cat(qkv_weight_tmp, dim=-1)
                        weight_dict[name.partition("self_attn")[0] + "self_attn.qkv_proj.weight"] = qkv_weight
                        qkv_weight_tmp = ["q", "k", "v"]
                        qkv_weight_len = 0
                elif name.find(".self_attn.q_proj.bias") != -1 or name.find(".self_attn.v_proj.bias") != -1:
                    qkv_bias_tmp[0 if "q_proj" in name else 2] = param_t # qkv order in weight dict is not guaranteed
                    qkv_bias_len += 1
                    if qkv_bias_len == 3:
                        qkv_bias = torch.cat(qkv_bias_tmp, dim=-1)
                        weight_dict[name.partition("self_attn")[0] + "self_attn.qkv_proj.bias"] = qkv_bias
                        qkv_bias_tmp = ["q", torch.zeros((model.config.d_model), dtype=torch_weight_dtype, device=model.device), "v"]
                        qkv_bias_len = 1
                else:
                    weight_dict[name] = param_t
            elif name.find("decoder.layernorm_embedding") != -1 or name.find("decoder.layer_norm") != -1 or name.find("final_logits_bias") != -1 or name.find("lm_head") != -1:
                weight_dict[name] = param_t
            elif name.find("decoder.embed_positions") != -1:
                weight_dict[name] = param

        # load by torch model directly
        # [0] self-attention
        t = torch.stack([weight_dict["decoder.layers.{}.self_attn_layer_norm.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t)
        # [1] QKV weight concatenated
        t = torch.stack([weight_dict["decoder.layers.{}.self_attn.qkv_proj.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        t = t.reshape([t.shape[0], t.shape[1], 3, t.shape[2] // 3])
        t = t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous()
        self.w.append(t)
        # [2]
        t = torch.stack([weight_dict["decoder.layers.{}.self_attn.out_proj.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t.split(t.shape[1] // self.tensor_para_size, dim=1)[self.tensor_para_rank].contiguous())
        # [3] cross-attention
        t = torch.stack([weight_dict["decoder.layers.{}.encoder_attn_layer_norm.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t)
        # [4]
        t = torch.stack([weight_dict["decoder.layers.{}.encoder_attn.q_proj.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous())
        # [5]
        t = torch.stack([weight_dict["decoder.layers.{}.encoder_attn.k_proj.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous())
        # [6]
        t = torch.stack([weight_dict["decoder.layers.{}.encoder_attn.v_proj.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous())
        # [7]
        t = torch.stack([weight_dict["decoder.layers.{}.encoder_attn.out_proj.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t.split(t.shape[1] // self.tensor_para_size, dim=1)[self.tensor_para_rank].contiguous())
        # [8]
        t = torch.stack([weight_dict["decoder.layers.{}.final_layer_norm.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t)
        # [9]
        t = torch.stack([weight_dict["decoder.layers.{}.fc1.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous())
        # [10] add empty weight for gated activation for now (BART/mBART model by default don't use gated activation)
        self.w.append(torch.empty((1, 1), dtype=torch_weight_dtype).contiguous().cuda())
        # [11]
        t = torch.stack([weight_dict["decoder.layers.{}.fc2.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t.split(t.shape[1] // self.tensor_para_size, dim=1)[self.tensor_para_rank].contiguous()) 

        if self.whisper_with_bias:
            # [17]
            t = torch.stack([weight_dict["decoder.layers.{}.self_attn_layer_norm.bias".format(i)]
                            for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            # [18]
            t = torch.stack([weight_dict["decoder.layers.{}.self_attn.qkv_proj.bias".format(i)]
                            for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            t = t.reshape([t.shape[0], 3, t.shape[-1] // 3])
            t = t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous()
            self.w.append(t)
            # [19]
            t = torch.stack([weight_dict["decoder.layers.{}.self_attn.out_proj.bias".format(i)]
                            for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            # [20]
            t = torch.stack([weight_dict["decoder.layers.{}.encoder_attn_layer_norm.bias".format(i)]
                            for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            # [21]
            t = torch.stack([weight_dict["decoder.layers.{}.encoder_attn.q_proj.bias".format(i)]
                            for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            t = t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous()
            self.w.append(t)
            # [22]
            t = torch.stack([torch.zeros((model.config.d_model), dtype=torch_weight_dtype, device=model.device)
                            for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            t = t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous()
            self.w.append(t)
            # [23]
            t = torch.stack([weight_dict["decoder.layers.{}.encoder_attn.v_proj.bias".format(i)]
                            for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            t = t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous()
            self.w.append(t)
            # [24]
            t = torch.stack([weight_dict["decoder.layers.{}.encoder_attn.out_proj.bias".format(i)]
                            for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            # [25]
            t = torch.stack([weight_dict["decoder.layers.{}.final_layer_norm.bias".format(i)]
                            for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            # [26]
            t = torch.stack([weight_dict["decoder.layers.{}.fc1.bias".format(i)]
                            for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            t = t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous()
            self.w.append(t)
            # [27] add empty bias for gated activation for now (BART/mBART model by default don't use gated activation)
            t = torch.empty((1, 1), dtype=torch_weight_dtype).contiguous().cuda()
            self.w.append(t)
            # [28]
            t = torch.stack([weight_dict["decoder.layers.{}.fc2.bias".format(i)]
                            for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
        else:
            # TODO: pass None Type to Torch Op
            for i in range(12):
                self.w.append(torch.empty((1, 1), dtype=torch_weight_dtype).contiguous().cuda())

    def to_cuda(self):
        for i in range(self.real_weights_num):
            self.w[i] = self.w[i].cuda()

    def to_float(self):
        for i in range(self.real_weights_num):
            self.w[i] = self.w[i].float()

    def to_half(self):
        for i in range(self.real_weights_num):
            self.w[i] = self.w[i].half()

    def to_single(self):
        for i in range(self.real_weights_num):
            self.w[i] = self.w[i].float()

    def to_bfloat16(self):
        for i in range(self.real_weights_num):
            self.w[i] = self.w[i].bfloat16()

class FTWhisperDecoder(nn.Module):
    def __init__(self, decoder_weight_list, lib_path, head_num, head_size, inter_size,
                 mem_d_model, d_model, num_layer, tensor_para_size=1, 
                 pipeline_para_size=1, whisper_with_bias=True, mwhisper=True, position_embedding_type=1,
                 activation_type="gelu", layernorm_type="post_layernorm", max_seq_len = 448):
        super().__init__()
        self.use_mpi = dist.is_mpi_available()
        self.num_layer=num_layer
        self.head_num=head_num
        self.max_seq_len=max_seq_len
        if self.use_mpi:
            try:
                dist.init_process_group(backend='mpi')
            except:
                print("[INFO] WARNING: Exception occurred in dist.init_process_group(backend = 'mpi'). Maybe the process group has been initialized somewhere else.")
        else:
            print("[INFO] MPI is not available in this PyTorch build.")
            assert tensor_para_size == 1, "[FATAL] MPI is required for tensor_para_size > 1."
            assert pipeline_para_size == 1, "[FATAL] MPI is required for pipeline_para_size > 1."

        torch.classes.load_library(lib_path)
        
        try:
            self.decoder = torch.classes.FasterTransformer.WhisperDecoder(*decoder_weight_list, head_num, head_size, inter_size, d_model,
                                                                     num_layer, mem_d_model, tensor_para_size, pipeline_para_size,
                                                                     whisper_with_bias, mwhisper, position_embedding_type, activation_type, layernorm_type)
        except:
            self.decoder = torch.classes.FasterTransformerWhisperDecoder(*decoder_weight_list, head_num, head_size, inter_size, d_model,
                                                                    num_layer, mem_d_model, tensor_para_size, pipeline_para_size,
                                                                    whisper_with_bias, mwhisper, position_embedding_type, activation_type, layernorm_type)

    
    def forward(self, inputs, memory, self_cache, mem_cache, step, beam_width=1):
        """
        inputs: hidden states, input token_embeds 
        memory: encoder_outputs, embeddings
        memory_seq_lens: length of encoder_outputs
        self_cache: self-attention kv-cache
        mem_cache: cross-attention kv-cache
        step: number of the current decoding step
        """
        # step = inputs.shape[0] # according to hf: hidden_states (`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
        # batch_size = inputs_shape[1] # # according to hf: hidden_states (`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
        inputs_shape = inputs.shape
        batch_size = inputs_shape[0]
        hidden_dim = inputs_shape[-1]
        encoder_outputs = memory
        encoder_seq_len = encoder_outputs.shape[1]
        mem_seq_lens = torch.tensor([encoder_seq_len]*encoder_outputs.shape[0]).type(torch.int32).to("cuda")
        seq_lens = torch.tensor([step]*batch_size).type(torch.int32).to("cuda")
        dtype = inputs.dtype
        size_per_head = hidden_dim // self.head_num
        element_size = inputs.element_size()
        if self_cache is None:
            self_cache = [
                torch.empty(self.num_layer, batch_size * beam_width, self.head_num, int(size_per_head/(16/element_size)), self.max_seq_len+1, int((16/element_size)), dtype=dtype, device='cuda'), # k_cache, the 1 is bc in decoding.cc it uses sizeof(T)/16, we're assuming T=fp16
                torch.empty(self.num_layer, batch_size * beam_width, self.head_num, self.max_seq_len+1, size_per_head, dtype=dtype, device='cuda') #v_cache
                ] 
        # inputs = inputs.reshape([-1, hidden_dim])
        if mem_cache is None:
            mem_cache = torch.empty(2, self.num_layer, batch_size * beam_width, encoder_seq_len, hidden_dim, dtype=dtype, device='cuda')
        relative_attention_bias_tensor=torch.empty((1, self.head_num, step, step))
        finished = torch.zeros(batch_size * beam_width, dtype=torch.bool, device='cuda')
        output, self_key_cache, self_val_cache, mem_key_cache, mem_val_cache = \
                self.decoder.forward2(step, inputs, memory, mem_seq_lens, seq_lens,
                # self.decoder.forward2(step, inputs.squeeze(), memory, mem_seq_lens, seq_lens,
                                       self_cache[0], self_cache[1], 
                                       mem_cache[0], mem_cache[1], 
                                       relative_attention_bias_tensor,
                                       finished)
        output = output.reshape(inputs_shape)

        return output, [self_key_cache, self_val_cache], [mem_key_cache, mem_val_cache]

