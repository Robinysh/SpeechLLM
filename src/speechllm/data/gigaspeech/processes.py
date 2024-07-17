import json
from pathlib import Path

import torchaudio

# pylint: disable-next=no-name-in-module
from samplerate import resample

from speechllm.data.utils import tokenize_func


def rename_cols(row):
    row["id"] = row.pop("ID")
    row["fpath"] = Path(row.pop("AUDIO"))
    row["duration"] = row.pop("DURATION")
    row["text"] = row.pop("TEXT")
    return row


def load_paired_audio(fpath):
    input_fpath = f"{fpath}_1.opus"
    output_fpath = f"{fpath}_2.opus"

    def f(x):
        audio, sr = torchaudio.load(x)
        audio = audio[0]
        if sr != 16000:
            ratio = 16000 / sr
            audio = resample(audio, ratio, "sinc_fastest")
        return audio

    input_audio = f(input_fpath)
    output_audio = f(output_fpath)

    return {
        "input_audio": input_audio,
        "output_audio": output_audio,
    }


def read_transcript(fpath):
    index = int(str(fpath).rsplit("_", maxsplit=1)[-1])
    txt_fpath = fpath.parent.with_suffix(".json")
    txt = json.loads(txt_fpath.read_text(encoding="utf-8"))
    info_pair = txt[index]
    return {
        "input_transcript": info_pair[0]["text_tn"],
        "output_transcript": info_pair[1]["text_tn"],
    }


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
    input_fpath = root_fpath / f'{data["fname"]}_1.txt'
    output_fpath = root_fpath / f'{data["fname"]}_2.txt'

    input_tokens = input_fpath.read_text(encoding="utf-8")
    output_tokens = output_fpath.read_text(encoding="utf-8")

    return {"input_tokens": input_tokens, "output_tokens": output_tokens}


def filter_long_audio(data, limit=2000):
    return (
        len(data["input_tokens"].split("><")) + len(data["output_tokens"].split("><"))
        <= limit
    )
