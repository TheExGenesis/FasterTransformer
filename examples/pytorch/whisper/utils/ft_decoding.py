# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
# %%
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np

class FTWhisperDecodingWeight(object):
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
        self.real_weights_num = 32  # assume all weights are allocated and converted to specific data type
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
        # [12] (1) positional embedding table should NOT be transposed, [max position embeddings, hidden size] (2) need to apply offset of 2 for absolute position embeddings in BART/mBART
        # t = weight_dict["decoder.embed_positions.weight"][2:, :].contiguous().cuda()
        t = weight_dict["decoder.embed_positions.weight"].contiguous().cuda()
        self.w.append(t)
        # [13] input embedding table should NOT be transposed, [vocab, hidden size]. Directly obtained from raw weight is untransposed
        t = model.get_input_embeddings().weight.contiguous().cuda()
        # input word embedding may be scaled (mBART), instead of customize this in FT, it's better to modify the embedding loading part in PyT
        embedding_scale = np.sqrt(model.config.d_model) if model.config.scale_embedding else 1.0
        t = t * embedding_scale
        self.w.append(t)
        # [14] output embedding table should NOT be transposed, [vocab, hidden size]. Directly obtained from raw weight is untransposed
        t = model.get_output_embeddings().weight.contiguous().cuda() # same as weight_dict["lm_head.weight"].transpose(1, 0).contiguous().cuda() 
        self.w.append(t)
        # [15] LayerNorm after embedding & before transformer block, special in BART/mBART
        # t = weight_dict["decoder.layernorm_embedding.weight"].contiguous().cuda()
        t = torch.empty((model.config.d_model), dtype=torch_weight_dtype, device=model.device)
        self.w.append(t)
        # [16] LayerNorm after transformer block, special in mBART
        if self.mwhisper:
            t = weight_dict["decoder.layer_norm.weight"].contiguous().cuda()
        else:
            t = torch.empty((1, 1), dtype=torch_weight_dtype).contiguous().cuda()
        self.w.append(t)
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
            # [29]
            # t = weight_dict["decoder.layernorm_embedding.bias"].contiguous().cuda()
            t = torch.empty((model.config.d_model), dtype=torch_weight_dtype, device=model.device)
            self.w.append(t)
            # [30]
            if self.mwhisper:
                t = weight_dict["decoder.layer_norm.bias"].contiguous().cuda()
            else:
                t = torch.empty((1, 1), dtype=torch_weight_dtype).contiguous().cuda()
            self.w.append(t)
            # [31] embedding bias aka final_logits_bias (may not exist, keys to ignore)
            t = weight_dict.get("final_logits_bias", torch.zeros((1, self.config.vocab_size), dtype=torch_weight_dtype)).contiguous().cuda()
            self.w.append(t)
        else:
            # TODO: pass None Type to Torch Op
            for i in range(15):
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

bad_words_ids = [[1,2,7,8,9,10,14,25,26,27,28,29,31,58,59,60,61,62,63,90,91,92,93,359,503,522,542,873,893,902,918,922,931,1350,1853,1982,2460,2627,3246,3253,3268,3536,3846,3961,4183,4667,6585,6647,7273,9061,9383,10428,10929,11938,12033,12331,12562,13793,14157,14635,15265,15618,16553,16604,18362,18956,20075,21675,22520,26130,26161,26435,28279,29464,31650,32302,32470,36865,42863,47425,49870,50254,50258,50360,50361,50362],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85]]

class FTWhisperDecoding(nn.Module):
    def __init__(self, decoding_weight_list, lib_path, head_num, head_size, inter_size,
                 mem_d_model, d_model, num_layer, start_id, end_id, vocab_size, q_scaling=1.0, num_bucket=32,
                 max_distance=128, tensor_para_size=1, pipeline_para_size=1, whisper_with_bias=True, mwhisper=True, position_embedding_type=1,
                 activation_type="gelu", layernorm_type="post_layernorm", bad_words_list=None):
        super().__init__()

        self.end_id = end_id
        self.use_mpi = dist.is_mpi_available()
        self.bad_words_list = torch.tensor(bad_words_ids, dtype=torch.int32).to("cuda") if bad_words_list is None else bad_words_list

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
            self.decoding = torch.classes.FasterTransformer.WhisperDecoding(head_num, head_size, inter_size, mem_d_model, d_model, num_layer,
                                                                       vocab_size, num_bucket, max_distance, q_scaling, start_id, end_id,
                                                                       tensor_para_size, pipeline_para_size, whisper_with_bias, mwhisper,
                                                                       position_embedding_type, activation_type, layernorm_type, *decoding_weight_list)
        except:
            self.decoding = torch.classes.FasterTransformerWhisperDecoding(head_num, head_size, inter_size, mem_d_model, d_model, num_layer,
                                                                       vocab_size, num_bucket, max_distance, q_scaling, start_id, end_id,
                                                                       tensor_para_size, pipeline_para_size, whisper_with_bias, mwhisper,
                                                                       position_embedding_type, activation_type, layernorm_type, *decoding_weight_list)

    def forward(self, beam_width, max_seq_len, top_k, top_p,
                beam_search_diversity_rate, temperature,
                len_penalty, repetition_penalty, random_seed,
                mem_hidden_states, mem_seq_len,
                is_return_output_log_probs, is_return_cum_log_probs, is_return_cross_attentions=False):
        # TODO (bhsueh) Not found an method to put a None Type into op forward function
        # So, the top_k and top_p must be some values now.
        results = self.decoding.forward2(beam_width, max_seq_len,
                                        top_k, top_p, beam_search_diversity_rate,
                                        temperature, len_penalty, repetition_penalty,
                                        random_seed, mem_hidden_states, mem_seq_len,
                                        is_return_output_log_probs, is_return_cum_log_probs, is_return_cross_attentions, self.bad_words_list)
        return results


class FTWhisper(nn.Module):
    def __init__(self, encoder, decoding):
        super().__init__()
        self.encoder = encoder
        self.decoding = decoding

    def forward(self, input_ids, attention_mask, inputs_embeds, beam_size, max_seq_len,
                top_k, top_p, beam_search_diversity_rate,
                temperature=1.0, len_penalty=0.0, repetition_penalty=1.0, random_seed=0,
                is_return_output_log_probs=False, is_return_cum_log_probs=False, is_return_cross_attentions=False, encoder_outputs=None):
        
        if encoder_outputs is None:
            if self.encoder is None or input_ids is None or attention_mask is None:
                raise ValueError("input_ids and attention_mask must be provided if encoder_outputs is None")
            input_ids = input_ids.to("cuda").type(torch.int32)
            mem_seq_len = torch.sum(attention_mask, dim=1).type(torch.int32).to("cuda")
            ft_encoder_outputs = self.encoder.forward(input_ids, mem_seq_len, inputs_embeds)
        else:
            mem_seq_len = torch.tensor([encoder_outputs.shape[1]]*encoder_outputs.shape[0]).type(torch.int32).to("cuda")
            ft_encoder_outputs = encoder_outputs
        results = self.decoding.forward(beam_size,  # optional, can be None
                                        max_seq_len,
                                        top_k,  # optional, can be None
                                        top_p,  # optional, can be None
                                        beam_search_diversity_rate,  # optional, can be None
                                        temperature,  # optional, can be None
                                        len_penalty,  # optional, can be None
                                        repetition_penalty,  # optional, can be None
                                        random_seed,  # optional, can be None
                                        is_return_output_log_probs,  # optional, can be None
                                        is_return_cum_log_probs,  # optional, can be None
                                        is_return_cross_attentions,  # optional, can be None
                                        ft_encoder_outputs,
                                        mem_seq_len,
                                        )
        return_dict = {}
        return_dict['output_ids'] = results.pop(0).reshape([-1, beam_size, max_seq_len]).cpu()
        return_dict['sequence_lengths'] = results.pop(0).reshape([-1, beam_size]).cpu()
        if is_return_output_log_probs:
            return_dict['output_log_probs'] = results.pop(0).cpu()
        if is_return_cum_log_probs:
            return_dict['cum_log_probs'] = results.pop(0).cpu()
        if is_return_cross_attentions:
            return_dict['cross_attentions'] = results.pop(0).cpu()

        return_dict['output_ids'] = pad_ft_eos(return_dict['output_ids'], return_dict['sequence_lengths'], self.decoding.end_id)
        return return_dict

def pad_ft_eos(output_ids, sequence_lengths, eos_token):
    """
    pad the output_ids[sequence_lengths:] with eos_token

    items: tensor (batch_size, seq_len)

    out: tensor (batch_size, seq_len)
    """
    batch_size, beam_size, seq_len = output_ids.shape
    for i in range(batch_size):
        for j in range(beam_size):
            output_ids[i, j, sequence_lengths[i]:] = eos_token
    return output_ids
