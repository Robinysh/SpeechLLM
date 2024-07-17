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
