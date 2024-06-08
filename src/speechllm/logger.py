import re
from functools import cache
from pathlib import Path

import numpy as np
import torch
from lightningtools import reporter
from speechtokenizer import SpeechTokenizer

import wandb
from speechllm.soundstorm_speechtokenizer.decode import decode_speech, load_soundstorm


def save_figure_to_numpy(fig, spectrogram=False):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if spectrogram:
        return data
    data = np.transpose(data, (2, 0, 1))
    return data


def log_image(image):
    image = ((image[0] + 0.5)).float()
    return wandb.Image(image)


def log_text(text):
    return text[0]


def log_audio(audio):
    return wandb.Audio(audio[0], sample_rate=16000)


@cache
def get_speech_tokens_models(fpath=None):
    if fpath is None:
        fpath = "/data3/robinysh/models/"
    fpath = Path(fpath)
    soundstorm = load_soundstorm(fpath / "soundstorm/speechtokenizer_soundstorm_mls.pt")
    speech_tokenizer = SpeechTokenizer.load_from_checkpoint(
        fpath / "speechtokenizer/config.json", fpath / "speechtokenizer/ckpt.dev"
    )
    soundstorm = torch.compile(soundstorm, backend="onnxrt")
    speech_tokenizer = torch.compile(speech_tokenizer, backend="onnxrt")
    return soundstorm, speech_tokenizer


def log_speech_tokens(tokens, decoder_fpath=None):
    soundstorm, speech_tokenizer = get_speech_tokens_models(decoder_fpath)
    sample_tokens = tokens[0]
    speech_code = re.search("(?<=<sosp>).*(?=<eosp>)", sample_tokens)
    audio = []
    if speech_code is not None:
        audio = decode_speech(
            speech_code.group(),
            soundstorm=soundstorm,
            speech_tokenizer=speech_tokenizer,
        )[0]
    return wandb.Audio(audio, sample_rate=16000)


reporter.register("image", log_image)
reporter.register("audio", log_audio)
reporter.register("speechtokens", log_speech_tokens)
reporter.register("text", log_text)
