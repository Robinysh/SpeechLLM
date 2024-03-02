from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from resemble_enhance.enhancer.inference import enhance


def rename_cols(row):
    row["id"] = row.pop("ID")
    row["fpath"] = Path(row.pop("AUDIO"))
    row["duration"] = row.pop("DURATION")
    row["text"] = row.pop("TEXT")
    return row


# pylint: disable-next=too-few-public-methods
class AudioEnhancer:
    def __init__(self):
        pass

    def __call__(self, row):
        gigaspeech_dataset_dir = None
        dst_dir = None
        audio = None
        device = "cuda"
        solver = "midpoint"
        nfe = 64
        tau = 0.5
        lambd = 0.9

        fpath = Path(gigaspeech_dataset_dir) / audio["path"]
        if not fpath.exists():
            print(f"{fpath} not found, skipping.")
            return

        # save_path = Path(args.dst_dir) / "enhanced" / fpath.with_suffix(".wav").name
        save_path = Path(dst_dir) / "enhanced" / fpath.with_suffix(".flac").name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.exists():
            print(f"{fpath} already enhanced.")
            return
        save_path = Path(dst_dir) / "enhanced" / fpath.with_suffix(".flac").name

        dwav, sampling_rate = torchaudio.load(fpath)
        dwav = dwav.mean(dim=0)

        print(f"Enhancing {fpath}.")
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


def add_cols(row, cols):
    row |= cols
    return row


def download_audio():
    pass


def diarization():
    pass


def split_dialogues():
    pass


def filter_dialogues():
    pass
