from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from icecream import ic
from resemble_enhance.enhancer.inference import enhance


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
        pass

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
        row["enhanced_audio"] = save_path
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


def download_audio(row):
    return row


class Diarizer:
    def __init__(self):
        pass

    def __call__(self, row):
        return row


def split_dialogues(row):
    return row


def filter_dialogues(row):
    return row
