from pathlib import Path

import torch
import torchaudio

# pylint: disable-next=no-name-in-module
from samplerate2 import resample

from speechllm.data.utils import tokenize_func


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


def group_texts(examples, block_size=256):
    result = {}
    for k in examples.keys():
        v = examples[k]
        if k == "input_ids":
            grouped_input_ids = []
            current_chunk = []
            current_length = 0
            for ids in v:
                if current_length + len(ids) > block_size:
                    grouped_input_ids.append(current_chunk)
                    current_chunk = []
                    current_length = 0
                current_chunk.extend(ids)
                current_length += len(ids)
            # Make sure the last chunk is added
            if current_length > 0:
                grouped_input_ids.append(current_chunk)
            result[k] = grouped_input_ids
    result["attention_mask"] = [[1 for _ in row] for row in result["input_ids"]]
    result["labels"] = result["input_ids"].copy()
    return result


def template_and_tokenize(data, tokenizer, prompter):
    prompt = prompter.generate_template(
        data["input"],
        data["output"],
    )
    return tokenize_func(prompt, tokenizer)


def read_audio_tokens(data, root_fpath):
    input_fpath = root_fpath / f'{data["text"]}_1.txt'
    output_fpath = root_fpath / f'{data["text"]}_2.txt'

    # TODO
    # input_tokens = input_fpath.read_text()
    # output_tokens = output_fpath.read_text()

    input_tokens = str(input_fpath)
    output_tokens = str(output_fpath)

    return {"input_tokens": input_tokens, "output_tokens": output_tokens}
