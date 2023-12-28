#!/usr/bin/env python3
# coding=utf8
# Copyright 2022  Jiayu DU

"""
This tool is used to extract supervised segments from GigaSpeech,
segments are saved in .wav format, supervisions are saved in a simple .tsv file:

--- exampler tsv begin ---
ID  AUDIO   BEGIN   DURATION    TEXT
POD1000000004_S0000017	audio/POD1000000004_S0000017.wav	0   3.163	YOU KNOW TO PUT THIS STUFF TOGETHER
...
...

--- exampler tsv end---

It can be, but not should be used to extract large subsets such as L, XL (because it would be extremely slow).
"""

import argparse
import csv
import os
from pathlib import Path

import torchaudio
from resemble_enhance.enhancer.inference import enhance
from scipy.io.wavfile import write
from speechcolab.datasets.gigaspeech import GigaSpeech

gigaspeech_punctuations = [
    "<COMMA>",
    "<PERIOD>",
    "<QUESTIONMARK>",
    "<EXCLAMATIONPOINT>",
]


# pylint: disable-next=too-many-locals
def split_audio(audio, args):
    audio_path = (
        Path(args.dst_dir) / "enhanced" / Path(audio["path"]).with_suffix(".wav").name
    )

    audio_info = torchaudio.info(audio_path)
    assert audio_info.num_channels == 1
    old_sample_rate = audio_info.sample_rate
    # encodec sampling rate
    new_sample_rate = 24000

    long_waveform, _ = torchaudio.load(audio_path)
    long_waveform = torchaudio.transforms.Resample(old_sample_rate, new_sample_rate)(
        long_waveform
    )

    utts = []
    for segment in audio["segments"]:
        text = segment["text_tn"]
        for punctuation in gigaspeech_punctuations:
            text = text.replace(punctuation, "").strip()
            text = " ".join(text.split())

        begin = segment["begin_time"]
        duration = segment["end_time"] - segment["begin_time"]
        frame_offset = int(begin * new_sample_rate)
        num_frames = int(duration * new_sample_rate)

        waveform = long_waveform[0][frame_offset : frame_offset + num_frames]  # mono

        segment_path = (
            Path(args.dst_dir) / "splitted" / audio["aid"] / f"{segment['sid']}.wav"
        )
        segment_path.parent.mkdir(parents=True, exist_ok=True)

        write(segment_path, new_sample_rate, waveform.numpy())

        utts.append(
            {
                "ID": segment["sid"],
                "AUDIO": segment_path,
                "DURATION": f"{duration:.4f}",
                "TEXT": text,
            }
        )
    return utts


def enhance_audio(audio, args):
    device = "cuda"
    solver = "midpoint"
    nfe = 64
    tau = 0.5
    lambd = 0.9

    fpath = Path(args.gigaspeech_dataset_dir) / audio["path"]
    save_path = Path(args.dst_dir) / "enhanced" / fpath.with_suffix(".wav").name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists():
        print(f"{fpath} already processed")
        return

    dwav, sampling_rate = torchaudio.load(fpath)
    dwav = dwav.mean(dim=0)

    wav, new_sr = enhance(
        dwav, sampling_rate, device, nfe=nfe, solver=solver, lambd=lambd, tau=tau
    )
    wav = wav.cpu().numpy()

    write(save_path, new_sr, wav)


def process_audio(args):
    os.makedirs(args.dst_dir, exist_ok=True)

    gigaspeech = GigaSpeech(args.gigaspeech_dataset_dir)
    subset = "{" + args.subset + "}"
    csv_rows = []
    for audio in gigaspeech.audios(subset):
        enhance_audio(audio, args)

    for audio in gigaspeech.audios(subset):
        utt = split_audio(audio, args)
        csv_rows += utt

    with open(
        os.path.join(args.dst_dir, "metadata.tsv"), "w+", encoding="utf8"
    ) as csv_file:
        csv_header_fields = ["ID", "AUDIO", "DURATION", "TEXT"]
        csv_writer = csv.DictWriter(
            csv_file, delimiter="\t", fieldnames=csv_header_fields, lineterminator="\n"
        )
        csv_writer.writeheader()

        for utt in csv_rows:
            csv_writer.writerow(utt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save the audio segments into wav, and meta into tsv."
    )
    parser.add_argument(
        "--subset",
        choices=["XS", "S", "M", "L", "XL", "DEV", "TEST"],
        default="XS",
        help="The subset name",
    )
    parser.add_argument(
        "gigaspeech_dataset_dir", help="The GigaSpeech corpus directory"
    )
    parser.add_argument("dst_dir", help="Ouput subset directory")

    process_audio(parser.parse_args())
