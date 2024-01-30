import argparse
import csv
import os
from functools import cache
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from bark_hubert_quantizer.customtokenizer import CustomTokenizer
from bark_hubert_quantizer.hubert_manager import HuBERTManager
from bark_hubert_quantizer.pre_kmeans_hubert import CustomHubert
from encodec.utils import convert_audio
from resemble_enhance.enhancer.inference import enhance, load_enhancer
from speechcolab.datasets.gigaspeech import GigaSpeech
from tqdm import tqdm

hubert_manager = HuBERTManager()
#


gigaspeech_punctuations = {
    ("<COMMA>", ","),
    ("<PERIOD>", "."),
    ("<QUESTIONMARK>", "?"),
    ("<EXCLAMATIONPOINT>", "!"),
}


# pylint: disable-next=too-many-locals
def split_audio(audio, args):
    audio_path = (
        Path(args.dst_dir) / "enhanced" / Path(audio["path"]).with_suffix(".flac").name
    )

    if not audio_path.exists():
        print(f"{audio_path} not found, skipping.")
        return []

    utts = []
    for segment in audio["segments"]:
        segment_path = (
            Path(args.dst_dir) / "splitted" / audio["aid"] / f"{segment['sid']}.flac"
        )
        text = segment["text_tn"]
        for punctuation, replacement in gigaspeech_punctuations:
            text = text.replace(punctuation, replacement).strip()
            text = " ".join(text.split())

        duration = segment["end_time"] - segment["begin_time"]
        utts.append(
            {
                "ID": segment["sid"],
                "AUDIO": segment_path,
                "DURATION": f"{duration:.4f}",
                "TEXT": text,
            }
        )

    if (Path(args.dst_dir) / "splitted" / audio["aid"]).exists():
        print(f"{audio_path} already splitted.")
        return utts

    audio_info = torchaudio.info(audio_path)
    assert audio_info.num_channels == 1
    old_sample_rate = audio_info.sample_rate
    # encodec sampling rate
    new_sample_rate = 24000

    long_waveform, _ = torchaudio.load(audio_path)
    long_waveform = torchaudio.transforms.Resample(old_sample_rate, new_sample_rate)(
        long_waveform
    )

    print(f"Splitting {audio_path}.")
    for segment in audio["segments"]:
        begin = segment["begin_time"]
        duration = segment["end_time"] - segment["begin_time"]
        frame_offset = int(begin * new_sample_rate)
        num_frames = int(duration * new_sample_rate)

        waveform = long_waveform[0][frame_offset : frame_offset + num_frames]  # mono

        segment_path = (
            Path(args.dst_dir) / "splitted" / audio["aid"] / f"{segment['sid']}.flac"
        )
        segment_path.parent.mkdir(parents=True, exist_ok=True)

        # write(segment_path, new_sample_rate, waveform.numpy())
        sf.write(segment_path, waveform.numpy(), new_sample_rate)
    return utts


def enhance_audio(audio, args):
    device = "cuda"
    solver = "midpoint"
    nfe = 64
    tau = 0.5
    lambd = 0.9

    fpath = Path(args.gigaspeech_dataset_dir) / audio["path"]
    if not fpath.exists():
        print(f"{fpath} not found, skipping.")
        return

    # save_path = Path(args.dst_dir) / "enhanced" / fpath.with_suffix(".wav").name
    save_path = Path(args.dst_dir) / "enhanced" / fpath.with_suffix(".flac").name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists():
        print(f"{fpath} already enhanced.")
        return
    save_path = Path(args.dst_dir) / "enhanced" / fpath.with_suffix(".flac").name

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


@cache
def load_hubert(model="quantifier_V1_hubert_base_ls960_23.pth", device="cuda"):
    hubert_model = CustomHubert(
        hubert_manager.make_sure_hubert_installed(), device=device
    ).half()
    tokenizer = (
        CustomTokenizer.load_from_checkpoint(
            hubert_manager.make_sure_tokenizer_installed(
                model=model, local_file="tokenizer_large.pth"
            )
        )
        .to(device)
        .half()
    )
    return hubert_model, tokenizer


def generate_hubert_semantics_emb(audio, segment, args, device="cuda"):
    fpath = Path(args.dst_dir) / "splitted" / audio["aid"] / f"{segment['sid']}.flac"

    if not fpath.exists():
        print(f"{fpath} not found, skipping.")
        return

    save_path = (
        Path(args.dst_dir)
        / "hubert_tokens"
        / audio["aid"]
        / fpath.with_suffix(".pt").name
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists():
        print(f"{fpath} already converted with hubert.")
        return

    model, tokenizer = load_hubert(device=device)

    # Load and pre-process the audio waveform
    wav, sr = torchaudio.load(fpath)
    wav = convert_audio(wav, sr, 16000, 1)
    wav = wav.to(device).half()
    semantic_vectors = model.forward(wav, input_sample_hz=16000)
    semantic_tokens = tokenizer.get_token(semantic_vectors)
    semantic_tokens = semantic_tokens.cpu()

    torch.save(semantic_tokens, save_path)


@torch.inference_mode()
def process_audio(args):
    os.makedirs(args.dst_dir, exist_ok=True)

    gigaspeech = GigaSpeech(args.gigaspeech_dataset_dir)
    subset = "{" + args.subset + "}"
    total_len = len([0 for _ in gigaspeech.audios(subset)])
    # total_len = 100

    for i, audio in enumerate(gigaspeech.audios(subset)):
        if i % 15 == 0:
            load_enhancer.cache_clear()
            torch.cuda.empty_cache()
            print("Reloading enhancer")
        enhance_audio(audio, args)
    load_enhancer.cache_clear()

    csv_rows = []
    for audio in tqdm(gigaspeech.audios(subset), total=total_len):
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

    for audio in tqdm(gigaspeech.audios(subset), total=total_len):
        for segment in audio["segments"]:
            generate_hubert_semantics_emb(audio, segment, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save the audio segments into flac, and meta into tsv."
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
