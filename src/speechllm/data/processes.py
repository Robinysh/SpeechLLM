from pathlib import Path

import torch
import torchaudio

# pylint: disable-next=no-name-in-module
from samplerate2 import resample
from transformers import AutoTokenizer

from speechllm.utils.qwen_generation_utils import make_context


def rename_cols(row):
    row["id"] = row.pop("ID")
    row["fpath"] = Path(row.pop("AUDIO"))
    row["duration"] = row.pop("DURATION")
    row["text"] = row.pop("TEXT")
    return row


def load_audio(row):
    audio, sr = torchaudio.load(row["fpath"])
    row["audio"] = audio[0]
    # assert sr == 16000, 'Data sampling rate does not match hubert sampling rate'
    if sr != 16000:
        ratio = 16000 / sr
        row["audio"] = resample(row["audio"], ratio, "sinc_fastest")
    return row


def load_hubert(row):
    fpath = Path(row["hubert"]) / Path(*Path(row["fpath"]).parts[-2:]).with_suffix(
        ".pt"
    )
    row["hubert_embs"] = torch.load(fpath)
    return row


# pylint: disable-next=too-few-public-methods
class Tokenize:
    def __init__(self):
        pretrained_model_dir = "Qwen/Qwen-Audio-Chat"
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_dir, use_fast=True, trust_remote_code=True
        )

    def __call__(self, row):
        # system = "You are a helpful assistant."
        query = self.tokenizer.from_list_format(
            [
                {"audio": str(row["fpath"])},
            ]
        )
        raw_text, _, audio_info = make_context(
            self.tokenizer, query, chat_format="chatml"
        )
        tokens = self.tokenizer(raw_text, audio_info=audio_info)
        raw_tokens = self.tokenizer.tokenize(raw_text, audio_info=audio_info)
        row |= tokens
        row["raw_text"] = raw_text
        row["raw_tokens"] = raw_tokens
        row["audio_info"] = audio_info
        return row


def add_cols(row, cols):
    row |= cols
    return row


def filter_outputs(row):
    output = {}
    gets = [
        "hubert_embs",
        "text",
        "fpath",
        "duration",
        "text",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "audio",
        "audio_info",
        "raw_text",
        "raw_tokens",
    ]
    for key in gets:
        output[key] = row[key]
    return output
