import random
from pathlib import Path

import torchaudio

# pylint: disable-next=no-name-in-module
from samplerate import resample


def read_audio_tokens(fpath):
    output_tokens = fpath.read_text(encoding="utf-8")
    return {"output_tokens": output_tokens}


def filter_long_audio(data, limit=2000):
    total_len = 0
    if "input_tokens" in data:
        total_len += len(data["input_tokens"].split("><"))
    if "output_tokens" in data:
        total_len += len(data["output_tokens"].split("><"))
    return total_len <= limit


def load_audio(fpath):
    audio, sr = torchaudio.load(fpath)
    audio = audio[0]
    if sr != 16000:
        ratio = 16000 / sr
        audio = resample(audio, ratio, "sinc_fastest")

    return {
        "output_audio": audio,
    }


def sample_context_interval(context, min_length=2, max_length=2):
    num_contexts = random.randint(
        min(min_length, len(context)), min(max_length, len(context))
    )
    num_contexts = num_contexts - num_contexts % 2
    start_idx = random.randint(0, len(context) - num_contexts)
    return (start_idx, start_idx + num_contexts)


def get_input_audio(path_root, fname, context_interval):
    return load_audio(Path(path_root) / fname / f"{context_interval[1]-2}.opus")[
        "output_audio"
    ]


def get_output_audio(path_root, fname, context_interval):
    return load_audio(Path(path_root) / fname / f"{context_interval[1]-1}.opus")[
        "output_audio"
    ]


def get_input_transcript(context, context_interval):
    return context[context_interval[1] - 2]


def get_output_transcript(context, context_interval):
    return context[context_interval[1] - 1]
