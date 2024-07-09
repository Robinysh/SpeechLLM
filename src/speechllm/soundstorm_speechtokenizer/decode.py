import re

import torch
from einops import rearrange

from speechllm.soundstorm_speechtokenizer.soundstorm import ConformerWrapper, SoundStorm


def load_soundstorm(model_path):
    conformer = ConformerWrapper(
        codebook_size=1024,
        num_quantizers=7,
        conformer={
            "dim": 1024,
            "depth": 12,
            "heads": 8,
            "dim_head": 128,
            "attn_flash": False,
        },
    )

    soundstorm = SoundStorm(
        net=conformer,
        num_semantic_token_ids=1024,
        semantic_pad_id=1024,
        pad_id=1024,
        schedule="cosine",
    )
    soundstorm.load(model_path)
    return soundstorm


def semantic2acoustic(semantic_tokens, prompt_tokens, soundstorm, tokenizer, steps=1):
    generated = soundstorm.generate(
        semantic_tokens=semantic_tokens,
        prompt_tokens=prompt_tokens,
        steps=steps,
        greedy=True,
    )
    wavs = tokenizer.decode(
        rearrange(generated, "b n q -> q b n", b=semantic_tokens.size(0))
    )  # wav: (b, 1, t)
    return wavs


@torch.no_grad()
def decode_speech(content, soundstorm, speech_tokenizer, prompt_tokens=None):
    semantic_codes = [[int(num) for num in re.findall(r"\d+", content)]]
    device = next(soundstorm.parameters()).device
    wav = semantic2acoustic(
        torch.Tensor(semantic_codes).int().to(device=device),
        prompt_tokens,
        soundstorm,
        speech_tokenizer,
        steps=4,
    )
    wav = wav.squeeze(0).detach().cpu().float()
    return wav
