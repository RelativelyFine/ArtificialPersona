import torch

from Voicebox import (
    VoiceBox,
    EncodecVoco,
    ConditionalFlowMatcherWrapper,
    HubertWithKmeans,
    TextToSemantic
)

# https://github.com/facebookresearch/fairseq/tree/main/examples/hubert

wav2vec = HubertWithKmeans(
    checkpoint_path = '/path/to/hubert/checkpoint.pt',
    kmeans_path = '/path/to/hubert/kmeans.bin'
)

text_to_semantic = TextToSemantic(
    wav2vec = wav2vec,
    dim = 512,
    source_depth = 1,
    target_depth = 1,
    use_openai_tokenizer = True
)

text_to_semantic.load('/path/to/trained/spear-tts/model.pt')

model = VoiceBox(
    dim = 512,
    audio_enc_dec = EncodecVoco(),
    num_cond_tokens = 500,
    depth = 2,
    dim_head = 64,
    heads = 16
)

cfm_wrapper = ConditionalFlowMatcherWrapper(
    voicebox = model,
    text_to_semantic = text_to_semantic
)

# mock data

audio = torch.randn(2, 12000)

# train

loss = cfm_wrapper(audio)
loss.backward()

# after much training

texts = [
    'the rain in spain falls mainly in the plains',
    'she sells sea shells by the seashore'
]

cond = torch.randn(2, 12000)
sampled = cfm_wrapper.sample(cond = cond, texts = texts) # (2, 1, <audio length>)