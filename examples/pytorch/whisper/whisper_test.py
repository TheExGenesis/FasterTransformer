# %%
import os
import sys
ROOT_DIR = os.path.abspath("../../../")
sys.path.append(ROOT_DIR)
lib_path = os.path.join(ROOT_DIR, 'build/lib/libth_whisper.so')
# lib_path = os.path.join(ROOT_DIR, 'build/lib/libth_bart.so')

# disable warning in notebook
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %%
import configparser
import numpy as np
import torch
import os
import numpy as np
import time
import math
from transformers import PreTrainedTokenizerFast
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from examples.pytorch.whisper.utils.ft_encoder import FTWhisperEncoderWeight, FTWhisperEncoder
from examples.pytorch.whisper.utils.ft_decoding import FTWhisperDecodingWeight, FTWhisperDecoding, FTWhisper
from utils.dataset import LibriSpeech
import whisper

# %% [markdown]
# ## Setup HuggingFace Whisper/MWhisper Model

# %%
# specify model name or checkpoint path
# model_name = 'facebook/Whisper-base' # Whisper
device = torch.device("cuda")
whisper_name = "large-v2"
processor = WhisperProcessor.from_pretrained(f"openai/whisper-{whisper_name}")
tokenizer = processor.tokenizer
model = (
    WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{whisper_name}")
    .cpu()
    .half()
    .eval()
    .cuda()
)
dataset = LibriSpeech("test-clean", half=True)
layernorm_type = "pre_layernorm"
is_mwhisper = True
# %%
# prep data
batch_size = 256
# batch_size = 192
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,
    pin_memory=True,
)
options = whisper.DecodingOptions(language="en", without_timestamps=True)
cwd = "/root/francisco-long-trial"
inputs = next(iter(loader))[0].to("cuda")
all_results = []

generated_ids = model.generate(inputs=inputs, max_new_tokens=model.config.max_length)
print(generated_ids.shape)


# %% [markdown]
# ## Setup FT Whisper Model

# %% [markdown]
# ### FT parameters

# %%
config = model.config
activation_type = config.activation_function
# single-gpu so set TP=1, PP=1
tensor_para_size = 1
pipeline_para_size = 1
whisper_with_bias = True
use_gated_activation = False
position_embedding_type = 1 # absolute positional embedding
weight_data_type = np.float32
encoder_head_size = config.d_model // config.encoder_attention_heads
decoder_head_size = config.d_model // config.decoder_attention_heads
remove_padding = False
use_fp16 = True

# %% [markdown]
# ### Load layer weights 

# ft_encoder_weight = FTWhisperEncoderWeight(
#     config,
#     tensor_para_size,
#     pipeline_para_size,
#     whisper_with_bias=whisper_with_bias,
#     mwhisper=is_mwhisper,
#     use_gated_activation=use_gated_activation,
#     position_embedding_type=position_embedding_type,
#     weight_data_type=weight_data_type,
# )
# ft_encoder_weight.load_from_model(model.float())

ft_decoding_weight = FTWhisperDecodingWeight(
    config,
    tensor_para_size,
    pipeline_para_size,
    whisper_with_bias=whisper_with_bias,
    mwhisper=is_mwhisper,
    use_gated_activation=use_gated_activation,
    position_embedding_type=position_embedding_type,
    weight_data_type=weight_data_type,
)
# ft_decoding_weight.load_from_model(model.float())
ft_decoding_weight.load_from_model(model)

if use_fp16:
    # ft_encoder_weight.to_half()
    ft_decoding_weight.to_half()

# %% [markdown]
# ### Setup Encoder, Decoder, and E2E model

# ft_encoder = FTWhisperEncoder(ft_encoder_weight.w, lib_path, config.encoder_attention_heads,
#                         encoder_head_size, config.encoder_ffn_dim,
#                         config.d_model, remove_padding, config.encoder_layers, 
#                         tensor_para_size=tensor_para_size, pipeline_para_size=pipeline_para_size, 
#                         whisper_with_bias=whisper_with_bias, mwhisper=is_mwhisper,
#                         position_embedding_type=position_embedding_type, 
#                         activation_type=activation_type, layernorm_type=layernorm_type)

ft_decoding = FTWhisperDecoding(ft_decoding_weight.w, lib_path,
                        config.decoder_attention_heads, decoder_head_size,
                        config.decoder_ffn_dim, config.d_model,
                        config.d_model, config.decoder_layers,
                        config.decoder_start_token_id, config.eos_token_id, config.vocab_size,
                        tensor_para_size=tensor_para_size, pipeline_para_size=pipeline_para_size, 
                        whisper_with_bias=whisper_with_bias, mwhisper=is_mwhisper,
                        position_embedding_type=position_embedding_type, 
                        activation_type=activation_type, layernorm_type=layernorm_type)

ft_whisper = FTWhisper(None, ft_decoding)

# %% [markdown]
# ## Example input and Inference parameters 

# %%
# profile single decoder step w memory - data prep
encoder_batch_size = batch_size
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=encoder_batch_size,
    num_workers=4,
    pin_memory=True,
)
inputs = next(iter(loader))[0].to("cuda")
encoder_kwargs = {
    "output_attentions": False,
    "output_hidden_states": False,
    "return_dict": True,
    "input_features": inputs[:encoder_batch_size],
}
input_ids = model._prepare_decoder_input_ids_for_generation(
    batch_size,
    device="cuda",
)
with torch.inference_mode():
    encoder_outputs = model.model.encoder(**encoder_kwargs)
torch.cuda.synchronize()

# %%
max_output_len = 99
ft_max_output_len = max_output_len - 2  # to achieve identical results w/ HF, exclude start & end tokens
num_beams = 1
beam_search_diversity_rate = 0.0
topk = None
topp = None
measurement_iters = 10

# %% [markdown]
# ## HF output and timing

# %%
if use_fp16:
    model.half()
else:
    model.float()
hf_outputs = model.generate(encoder_outputs=encoder_outputs, max_new_tokens=model.config.max_length, use_cache=True, num_beams=num_beams)
hf_tokens = tokenizer.batch_decode(hf_outputs, skip_special_tokens=True)
print("HF output ids",hf_outputs)
print("HF output text",hf_tokens)

# %%
hf_latencies = []
for _ in range(measurement_iters):
    start_time = time.time()
    model.generate(encoder_outputs=encoder_outputs, max_new_tokens=model.config.max_length, use_cache=True, num_beams=num_beams)
    end_time = time.time()
    hf_latencies.append(end_time - start_time)
hf_p50 = np.percentile(hf_latencies, 50)
hf_p99 = np.percentile(hf_latencies, 99)
print(f"HF p50: {hf_p50*1000:.2f} ms, p99: {hf_p99*1000:.2f} ms ")

# %% [markdown]
# ## FT output and timing

# %%
return_dict = ft_whisper(None,
                      None,
                      inputs_embeds=None,
                      beam_size=num_beams,
                      max_seq_len=ft_max_output_len,
                      top_k=topk,
                      top_p=topp,
                      beam_search_diversity_rate=beam_search_diversity_rate,
                      is_return_output_log_probs=False,
                      is_return_cum_log_probs=False,
                      encoder_outputs=encoder_outputs["last_hidden_state"])

# ft_whisper returns output_ids of shape [batch_size, beam_width, max_output_seq_len]
# ft_whisper returns sequence_length of shape [batch_size, beam_width]
ft_output_ids = return_dict['output_ids']
ft_sequence_length = return_dict['sequence_lengths']

# %%
ft_outputs = []
for i in range(batch_size):
    # selecting the top sequence from beam width number of sequences
    ft_outputs.append(list(ft_output_ids[i, 0, :][1:ft_sequence_length[i , 0]])) # start from 1 to exclude the 1st token
ft_tokens = tokenizer.batch_decode(ft_outputs, skip_special_tokens=True)
print("FT output ids", ft_outputs)
print("FT output text", ft_tokens)

# %%
ft_latencies = []
tok_counts = []
for _ in range(measurement_iters):
    start_time = time.time()
    return_dict = ft_whisper(None,
                      None,
                      inputs_embeds=None,
                      beam_size=num_beams,
                      max_seq_len=ft_max_output_len,
                      top_k=topk,
                      top_p=topp,
                      beam_search_diversity_rate=beam_search_diversity_rate,
                      is_return_output_log_probs=False,
                      is_return_cum_log_probs=False,
                      encoder_outputs=encoder_outputs["last_hidden_state"])
    end_time = time.time()
    ft_latencies.append(end_time - start_time)
    tok_counts.append(sum(return_dict['sequence_lengths']))
ft_p50 = np.percentile(ft_latencies, 50)
ft_p99 = np.percentile(ft_latencies, 99)
ft_p50_tok = np.percentile(tok_counts, 50)
print(f"FT p50: {ft_p50*1000:.2f} ms, p99: {ft_p99*1000:.2f} ms, tok/s: {ft_p50_tok/ft_p50} ")

# %% [markdown]
# ## Performance summary

# %%
print(f"Precision: {'FP16' if use_fp16 else 'FP32'}")
print(f"Input length: 0, Output length: {model.config.max_length}")
print(f"HF p50: {hf_p50*1000:.2f} ms, p99: {hf_p99*1000:.2f} ms, tok/s: {ft_p50_tok/hf_p50} ")
print(f"FT p50: {ft_p50*1000:.2f} ms, p99: {ft_p99*1000:.2f} ms, tok/s: {ft_p50_tok/ft_p50} ")

# %%



