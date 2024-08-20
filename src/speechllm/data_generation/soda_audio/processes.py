import io
import json
import logging
from copy import deepcopy
from pathlib import Path

import torch
import torchaudio
from huggingface_hub import snapshot_download
from speechtokenizer import SpeechTokenizer

from speechllm.data_generation.audio2tokens import speech_tokens_to_string


def export_json(row, output_path):
    save_path = Path(output_path) / "json" / f"{row['original_index']}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    row["json_fpath"] = save_path
    if save_path.exists():
        return row
    data = deepcopy(row)
    data.pop("audio_second_last_turn")
    save_path.write_text(json.dumps(data, ensure_ascii=False, indent=4))
    return row


def export_audio(row, output_path):
    save_path = Path(output_path) / "audio" / f"{row['original_index']}.opus"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    row["audio_fpath"] = save_path
    if save_path.exists():
        return row

    f = io.BytesIO(row["audio_second_last_turn"]["bytes"])
    data, sr = torchaudio.load(f)
    assert sr == 16000, f'Sampling rate is {sr} for item {row["original_index"]}'
    torchaudio.save(
        save_path,
        data,
        sr,
    )
    return row


class SpeechTokenizerGenerator:
    def __init__(self, device="cuda", dtype=torch.half):
        speech_modules_path = snapshot_download(
            repo_id="fnlp/AnyGPT-speech-modules", repo_type="model"
        )
        model_fpath = Path(speech_modules_path) / "speechtokenizer"
        self.speech_tokenizer = SpeechTokenizer.load_from_checkpoint(
            model_fpath / "config.json", model_fpath / "ckpt.dev"
        ).to(device, dtype=dtype)
        self.device = device
        self.dtype = dtype

    def __call__(self, row, output_path, input_path):
        # audio_basepath = Path(output_path) / "dialogue_audio" / str(row["original_index"])
        audio_basepath = (
            Path(input_path) / "dialogue_audio" / str(row["original_index"])
        )
        save_basepath = Path(output_path) / "tts_tokens" / str(row["original_index"])
        save_basepath.mkdir(parents=True, exist_ok=True)
        if save_basepath.exists() and len(list(save_basepath.glob("*.txt"))) == len(
            row["dialogue"]
        ):
            return row

        for i in range(len(row["dialogue"])):
            audio_fpath = audio_basepath / f"{i}.opus"
            save_fpath = save_basepath / f"{i}.txt"
            if not audio_fpath.exists():
                logging.warning(f"Audio file {audio_fpath} does not exist")
                continue
            code = self.encode_speech(audio_fpath)
            code_string = speech_tokens_to_string(code)
            save_fpath.write_text(code_string, encoding="utf-8")
        return row

    def encode_speech(self, audio_path):
        wav, sr = torchaudio.load(audio_path)
        # monophonic checking
        if wav.shape[0] > 1:
            wav = wav[:1,]
        if sr != self.speech_tokenizer.sample_rate:
            wav = torchaudio.functional.resample(
                wav, sr, self.speech_tokenizer.sample_rate
            )
        wav = wav.unsqueeze(0).to(self.device, dtype=self.dtype)
        # Extract discrete codes from SpeechTokenizer
        with torch.no_grad():
            codes = self.speech_tokenizer.encode(wav)  # codes: (n_q, B, T)
        return codes[0, 0, :]
