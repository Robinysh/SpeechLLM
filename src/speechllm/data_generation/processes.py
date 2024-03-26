import os
import pickle
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from icecream import ic
from pyannote.audio import Pipeline
from resemble_enhance.enhancer.inference import enhance, load_enhancer

from speechllm.data_generation.speechcolab.datasets.gigaspeech import GigaSpeech


def rename_cols(row):
    row["id"] = row.pop("ID")
    row["fpath"] = Path(row.pop("AUDIO"))
    row["duration"] = row.pop("DURATION")
    row["text"] = row.pop("TEXT")
    return row


def extract_rows(row):
    ic(row)


# pylint: disable-next=too-few-public-methods
class AudioEnhancer:
    def __init__(self):
        self.model = load_enhancer(None, "cuda")

    def __call__(self, row):
        device = "cuda"
        solver = "midpoint"
        nfe = 64
        tau = 0.5
        lambd = 0.9

        fpath = Path(row["data_path"]) / row["path"]
        if not fpath.exists():
            # print(f"{fpath} not found, skipping.")
            return row

        # save_path = Path(args.dst_dir) / "enhanced" / fpath.with_suffix(".wav").name
        save_path = (
            Path(row["output_path"]) / "enhanced" / fpath.with_suffix(".flac").name
        )
        row["enhanced_audio"] = str(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.exists():
            # print(f"{fpath} already enhanced.")
            return row

        dwav, sampling_rate = torchaudio.load(fpath)
        dwav = dwav.mean(dim=0)

        # print(f"Enhancing {fpath}.")
        wav, new_sr = enhance(
            dwav,
            sampling_rate,
            device,
            nfe=nfe,
            solver=solver,
            lambd=lambd,
            tau=tau,
            dtype=torch.half,
        )
        wav = wav.cpu().numpy()

        # write(save_path, new_sr, wav)
        sf.write(save_path, wav, new_sr)
        return row


def add_cols(row, cols):
    row |= cols
    return row


# pylint: disable=too-few-public-methods
class Downloader:
    def __init__(self, data_path):
        self.gigaspeech = GigaSpeech(data_path)
        self.gigaspeech.password = os.getenv("GIGASPEECH_PASSWORD")
        self.gigaspeech.gigaspeech_release_url = (
            "https://freedata.oss-cn-beijing.aliyuncs.com/magichub/GigaSpeech"
        )

    def __call__(self, row):
        self.gigaspeech.download_and_process_object_from_release(
            row["md5"], Path(row["path"]).parent.with_suffix(".tgz.aes")
        )
        return row


# from ray.util import inspect_serializabilit4
# inspect_serializability(download_audio, name="test")


class Diarizer:
    def __init__(self):
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        self.pipeline.to(torch.device("cuda:0"))

    def __call__(self, row):
        # fname = Path(row['data_path'])/'enhanced'/Path(row['path']).with_suffix('.flac').name
        fname = Path(row["data_path"]) / Path(row["path"])
        waveform, sample_rate = torchaudio.load(fname)
        diarization = self.pipeline({"waveform": waveform, "sample_rate": sample_rate})
        save_path = (
            Path(row["data_path"])
            / "diarization"
            / Path(row["path"]).with_suffix(".pkl").name
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("wb") as fp:
            pickle.dump(diarization, fp)
        return row


def split_dialogues(row):
    return row


def filter_dialogues(row):
    return row
