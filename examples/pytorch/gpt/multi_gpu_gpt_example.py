# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

import argparse
import os
import sys
import timeit

import torch
from torch.nn.utils.rnn import pad_sequence

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../../.."))
import examples.pytorch.gpt.utils.gpt_token_encoder as encoder
from examples.pytorch.gpt.utils import comm
from examples.pytorch.gpt.utils import gpt_decoder
from examples.pytorch.gpt.utils.parallel_gpt import ParallelGPT


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer_num', type=int, default=24,
                        help='number of layers')
    parser.add_argument('--input_len', type=int, default=1,
                        help='input sequence length to generate.')
    parser.add_argument('--output_len', type=int, default=32,
                        help='output sequence length to generate.')
    parser.add_argument('--head_num', type=int, default=16,
                        help='head number')
    parser.add_argument('--size_per_head', type=int, default=64,
                        help='size per head')
    parser.add_argument('--vocab_size', type=int, default=50304,
                        help='vocab size')
    parser.add_argument('--beam_width', type=int, default=1,
                        help='beam width for beam search. Using sampling when beam width is 1.')
    parser.add_argument('--top_k', type=int, default=1,
                        help='top k candidate num')
    parser.add_argument('--top_p', type=float, default=0.,
                        help='top p probability threshold')
    parser.add_argument('--temperature', type=float, default=1.,
                        help='temperature')
    parser.add_argument('--len_penalty', type=float, default=0.,
                        help='len_penalty')
    parser.add_argument('--beam_search_diversity_rate', type=float, default=0.,
                        help='beam_search_diversity_rate')
    parser.add_argument('--tensor_para_size', type=int, default=1,
                        help='tensor parallel size')
    parser.add_argument('--pipeline_para_size', type=int, default=1,
                        help='pipeline parallel size')
    parser.add_argument('--ckpt_path', type=str, default='../models/megatron-models/c-model/345m/1-gpu',
                        help='path to the checkpoint file.')
    parser.add_argument('--lib_path', type=str, default='./lib/libth_parallel_gpt.so',
                        help='path to the pyt_fastertransformer dynamic lib file.')
    parser.add_argument('--vocab_file', type=str, default="../models/gpt2-vocab.json",
                        help='vocabulary file.')
    parser.add_argument('--merges_file', type=str, default="../models/gpt2-merges.txt",
                        help='merges file.')
    parser.add_argument('--start_id', type=int, default=50256,
                        help='start token id.')
    parser.add_argument('--end_id', type=int, default=50256,
                        help='end token id.')
    parser.add_argument('--max_batch_size', type=int, default=8,
                        help='max batch size.')
    parser.add_argument('--repetition_penalty', type=float, default=1.,
                        help='repetition penalty')
    parser.add_argument('--max_seq_len', type=int, default=1024,
                        help='max sequence length for position embedding table.')
    parser.add_argument('--inference_data_type', '--data_type', type=str, choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument('--time', action='store_true',
                        help='whether or not to measure time elapsed.')
    parser.add_argument('--sample_input_file', type=str, default=None,
                        help='path to sample input file. If not set, it runs with no context inputs.')
    parser.add_argument('--sample_output_file', type=str, default=None,
                        help='path to sample output file.')
    parser.add_argument('--enable_random_seed', action='store_true',
                        help='is use the random seed for sentences in a batch.')
    parser.add_argument('--int8_mode', type=int, default=0, choices=[0, 1],
                        help='The level of quantization to perform.'
                             ' 0: No quantization. All computation in data_type'
                             ' 1: Quantize weights to int8, all compute occurs in fp16/bf16. Not supported when data_type is fp32')
    parser.add_argument(
        '--weights_data_type',
        type=str,
        default="fp32",
        choices=["fp32", "fp16"],
        help='Data type of FT checkpoint weights',
    )
    parser.add_argument('--return_cum_log_probs', type=int, default=0, choices=[0, 1, 2],
                        help='Whether to compute the cumulative log probsbility of sentences.'
                             ' 0: do not return the cumulative log probs '
                             ' 1: return the cumulative log probs of generated sequences'
                             ' 2: return the cumulative log probs of sequences')
    parser.add_argument('--shared_contexts_ratio', type=float, default=1.0,
                        help='Triggers the shared context optimization when'
                             'compact_size <= shared_contexts_ratio * batch_size'
                             'A value of 0.0 deactivate the optimization')
    parser.add_argument('--use_gpt_decoder_ops', action='store_true',
                        help='Use separate decoder FT operators instead of end-to-end model op.')
    args = parser.parse_args()

    layer_num = args.layer_num
    output_len = args.output_len
    head_num = args.head_num
    size_per_head = args.size_per_head
    vocab_size = args.vocab_size
    beam_width = args.beam_width
    top_k = args.top_k
    top_p = args.top_p
    temperature = args.temperature
    len_penalty = args.len_penalty
    beam_search_diversity_rate = args.beam_search_diversity_rate
    tensor_para_size = args.tensor_para_size
    pipeline_para_size = args.pipeline_para_size
    start_id = args.start_id
    end_id = args.end_id
    max_batch_size = args.max_batch_size
    max_seq_len = args.max_seq_len
    repetition_penalty = args.repetition_penalty
    weights_data_type = args.weights_data_type
    return_cum_log_probs = args.return_cum_log_probs
    return_output_length = return_cum_log_probs > 0
    shared_contexts_ratio = args.shared_contexts_ratio

    print('\n=================== Arguments ===================')
    for k, v in vars(args).items():
        print(f'{k.ljust(30, ".")}: {v}')
    print('=================================================\n')

    enc = encoder.get_encoder(args.vocab_file, args.merges_file)
    torch.manual_seed(0)

    comm.initialize_model_parallel(args.tensor_para_size, args.pipeline_para_size)
    rank = comm.get_rank()
    device = comm.get_device()

    # Inputs
    contexts = []
    if args.sample_input_file:  # conditional case
        with open(args.sample_input_file, "r") as f:
            contexts = f.read().splitlines()
            batch_size = min(len(contexts), max_batch_size)
        contexts = contexts[:batch_size]
        start_ids = [torch.tensor(enc.encode(c), dtype=torch.int32, device=device) for c in contexts]
    else:  # unconditional case
        batch_size = max_batch_size
        contexts = ['<|endoftext|>'] * batch_size
        start_ids = [torch.IntTensor([end_id for _ in range(args.input_len)])] * batch_size

    start_lengths = [len(ids) for ids in start_ids]

    start_ids = pad_sequence(start_ids, batch_first=True, padding_value=end_id)
    start_lengths = torch.IntTensor(start_lengths)


    # Prepare model.
    if not args.use_gpt_decoder_ops:
        gpt = ParallelGPT(head_num, size_per_head, vocab_size, start_id, end_id,
                          layer_num, max_seq_len, tensor_para_size, pipeline_para_size,
                          lib_path=args.lib_path, inference_data_type=args.inference_data_type,
                          int8_mode=args.int8_mode, weights_data_type=weights_data_type,
                          shared_contexts_ratio=shared_contexts_ratio)
        if not gpt.load(ckpt_path=args.ckpt_path):
            print("[WARNING] Checkpoint file not found. Model loading is skipped.")
    else:
        gpt = gpt_decoder.Gpt(
            num_heads=head_num,
            size_per_head=size_per_head,
            num_layers=layer_num,
            vocab_size=vocab_size,
            start_id=start_id,
            end_id=end_id,
            tensor_para_size=tensor_para_size,
            pipeline_para_size=pipeline_para_size,
            lib_path = args.lib_path,
            max_seq_len=max_seq_len,
            int8_mode=args.int8_mode,
            weights_data_type=args.weights_data_type)
        gpt.load(args.ckpt_path, args.inference_data_type)

    if args.enable_random_seed:
        random_seed_tensor = torch.randint(0, 10000, size=[max_batch_size], dtype=torch.int64)
    else:
        random_seed_tensor = torch.zeros([max_batch_size], dtype=torch.int64)

    infer_decode_args = dict(
        beam_width=beam_width,
        top_k=top_k * torch.ones(max_batch_size, dtype=torch.int32),
        top_p=top_p * torch.ones(max_batch_size, dtype=torch.float32),
        temperature=temperature * torch.ones(max_batch_size, dtype=torch.float32),
        repetition_penalty=repetition_penalty * torch.ones(max_batch_size, dtype=torch.float32),
        beam_search_diversity_rate=beam_search_diversity_rate * torch.ones(max_batch_size, dtype=torch.float32),
        len_penalty=len_penalty * torch.ones(size=[max_batch_size], dtype=torch.float32),
        random_seed=random_seed_tensor
    )

    if not args.use_gpt_decoder_ops:
        def gpt_generate_fn():
            tokens_batch = gpt(start_ids,
                               start_lengths,
                               output_len,
                               return_output_length=return_output_length,
                               return_cum_log_probs=return_cum_log_probs,
                               **infer_decode_args)
            return tokens_batch
    else:
        def gpt_generate_fn():
            output_dict = gpt.generate(input_token_ids=start_ids,
                                       input_lengths=start_lengths,
                                       gen_length=output_len,
                                       eos_token_id=end_id,
                                       return_output_length=return_output_length,
                                       return_log_probs=return_cum_log_probs,
                                       **infer_decode_args)
            return output_dict

    # Generate tokens.
    gen_outputs = gpt_generate_fn()

    if rank == 0:
        if not args.use_gpt_decoder_ops:
            if return_cum_log_probs > 0:
                tokens_batch, _, cum_log_probs = gen_outputs
            else:
                tokens_batch, cum_log_probs = gen_outputs, None
        else:
            tokens_batch = gen_outputs['output_token_ids']
            cum_log_probs = gen_outputs['cum_log_probs'] if return_cum_log_probs > 0 else None
        if cum_log_probs is not None:
            print('[INFO] Log probs of sentences:', cum_log_probs)

        outputs = []
        tokens_batch = tokens_batch.cpu().numpy()
        for i, (context, tokens) in enumerate(zip(contexts, tokens_batch)):
            for beam_id in range(beam_width):
                token = tokens[beam_id][start_lengths[i]:]  # exclude context input from the output
                output = enc.decode(token)
                outputs.append(output)
                print(f'[INFO] batch {i}, beam {beam_id}:\n[Context]\n{context}\n\n[Output]\n{output}\n')

        if args.sample_output_file:
            with open(args.sample_output_file, "w+") as f:
                outputs = [o.replace("\n", "\\n") for o in outputs]
                f.writelines("\n".join(outputs))

    # Measure inference time.
    if args.time:
        iterations = 10
        for _ in range(iterations):
            gpt_generate_fn()
        time = timeit.default_timer()
        for _ in range(iterations):
            gpt_generate_fn()
        time_elapsed = timeit.default_timer() - time
        print(f'[INFO] GPT time costs: {time_elapsed * 1000 / iterations:.2f} ms')


if __name__ == '__main__':
    main()
