# flake8-in-file-ignores: noqa: C901
import json
import logging
import os
import pickle
import re
from itertools import pairwise
from pathlib import Path
from random import randint

import ChatTTS
import librosa
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import snapshot_download
from icecream import ic
from nemo_text_processing.text_normalization.normalize import Normalizer
from openai import OpenAI
from pyannote.audio import Pipeline
from pyannote.core import Segment
from pydub import AudioSegment
from resemble_enhance.enhancer.inference import enhance, load_enhancer
from speechtokenizer import SpeechTokenizer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from speechllm.data_generation.audio2tokens import speech_tokens_to_string
from speechllm.data_generation.gigaspeech.speechcolab.datasets.gigaspeech import (
    GigaSpeech,
)

logging.getLogger("ChatTTS").disabled = True

DIALOGUE_PAIR_DIR = "dialogue_pairs"
AUDIO_PAIR_DIR = "audio_pairs"
TOKENS_DIR = "tokens"


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
            return row

        save_path = (
            Path(row["output_path"]) / "enhanced" / fpath.with_suffix(".flac").name
        )
        row["enhanced_audio"] = str(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.exists():
            return row

        dwav, sampling_rate = torchaudio.load(fpath)
        dwav = dwav.mean(dim=0)

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


class Diarizer:
    def __init__(self):
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.environ.get("HF_AUTH_TOKEN", True),
        )
        self.pipeline.to(torch.device("cuda:0"))

    def __call__(self, row):
        try:
            fname = Path(row["data_path"]) / Path(row["path"])
            save_path = (
                Path(row["data_path"])
                / "diarization"
                / Path(row["path"]).with_suffix(".pkl").name
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if save_path.exists():
                return row
            waveform, sample_rate = torchaudio.load(fname)

            diarization = self.pipeline(
                {"waveform": waveform, "sample_rate": sample_rate}
            )
            with save_path.open("wb") as fp:
                pickle.dump(diarization, fp)
        # pylint: disable=broad-except
        except Exception as e:
            print(f"Error: {row}")
            print(e)
        return row


def split_dialogues(row):
    try:
        dialogue_path = (
            Path(row["data_path"]) / DIALOGUE_PAIR_DIR / Path(row["path"]).stem
        ).with_suffix(".json")
        if not dialogue_path.exists():
            return row
        dialogue_data = json.loads(dialogue_path.read_text())
        if len(dialogue_data) == 0:
            return row

        save_path = Path(row["data_path"]) / AUDIO_PAIR_DIR / Path(row["path"]).stem
        save_path.mkdir(parents=True, exist_ok=True)
        audio = None
        for i, (seg1, seg2) in enumerate(dialogue_data):
            assert (
                seg1["begin_time"] < seg1["end_time"]
                and seg2["begin_time"] < seg2["end_time"]
            )
            seg1_path = (save_path / f"{Path(row['path']).stem}_{i}_1").with_suffix(
                ".opus"
            )
            seg2_path = (save_path / f"{Path(row['path']).stem}_{i}_2").with_suffix(
                ".opus"
            )
            if not seg1_path.exists():
                if audio is None:
                    audio = AudioSegment.from_file(Path(row["data_path"]) / row["path"])
                audio1 = audio[seg1["begin_time"] * 1000 : seg1["end_time"] * 1000]
                audio1.export(seg1_path, format="opus")
            if not seg2_path.exists():
                if audio is None:
                    audio = AudioSegment.from_file(Path(row["data_path"]) / row["path"])
                audio2 = audio[seg2["begin_time"] * 1000 : seg2["end_time"] * 1000]
                audio2.export(seg2_path, format="opus")
            assert os.path.getsize(seg2_path) > 200
            assert os.path.getsize(seg2_path) > 200
    # pylint: disable=broad-except
    except Exception as e:
        print(row)
        print(e)
    return row


class DialogueFilter:
    def __init__(self):
        self.client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
        self.max_rand_len = 3
        self.max_token = 100
        self.max_duration = 15
        self.max_response_delay = 10
        self.max_sequential_delay = 3

    def query_llm(self, query):
        # model="heyholetsgo/Nous-Hermes-2-Mistral-7B-DPO-AWQ"
        model = "TheBloke/Mixtral_11Bx2_MoE_19B-GPTQ"
        # model="casperhansen/llama-3-8b-instruct-awq"
        # model = "astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit"
        # model="TheBloke/SauerkrautLM-UNA-SOLAR-Instruct-AWQ"
        # model="TheBloke/mistral-ft-optimized-1218-AWQ"
        # model="TheBloke/mistral-ft-optimized-1227-AWQ"
        # model="TheBloke/OpenHermes-2.5-Mistral-7B-AWQ"

        chat_response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant and accurate analyist. The performance of your job is very important to me.",
                },
                {"role": "user", "content": query},
            ],
        )

        response = chat_response.choices[0].message.content
        return response

    def check_is_response(self, sent1, sent2, retries=3):
        for i in range(retries):
            try:
                response = self.query_llm(
                    f"""
                The following sentences happens sequentially in a dialogue.
                Is the second sentence a response to the first sentence from another speaker that is likely to happen in a daily casual chatting?
                A mere continuation of the first sentence does not count as a response.
                Give an explanation and response yes or no. Answer no if you are unsure. Follow this format of the examples strictly.

                For example:
                1. ITEMIZED .
                2. SEARS ACTUALLY PROVIDED AN ENTIRELY SEPARATE CATALOG FOR THESE KIT .
                Explanation: It is unclear what is the first sentence is about.
                Answer: no

                1. AT 7:45 ?
                2. MHMM .
                Explanation: The second sentence is directly answering the first sentence. It is also likely to be said by another speaker.
                Answer: yes

                1. IN SOME NEIGHBORHOODS , A SEARS KIT HOME MIGHT BE THE ONLY HOUSE ON THE BLOCK WITH ELECTRICITY .
                2. MEN AND WOMEN OF COLOR , AND SINGLE WOMEN WHO WOULD OTHERWISE NEVER HAVE A CHANCE OF BECOMING A HOMEOWNER
                Explanation: The second sentence is merely a continuation of the first sentence.
                Answer: no

                1. MINE IS A LONG AND A SAD TALE !
                2. SAID THE MOUSE , TURNING TO ALICE , AND SIGHING .
                Explanation: Although they are from different speakers, the second sentence is not a response, but a narration. A response is also unlikely to start with the word 'SAID'.
                Answer: no

                Now give your answer to these two sentences:
                1. {sent1}
                2. {sent2}
                """
                )
                ans_sent = re.findall("(?<=[Aa]nswer:).*$", response, re.MULTILINE)
                if len(ans_sent) > 0:
                    ans_sent = ans_sent[-1]
                    answer = any(x in ans_sent for x in ["yes", "Yes", "YES"])
                    # ic(response, ans_sent, sent1, sent2, answer)
                    return answer
            # pylint: disable=broad-except
            except Exception as e:
                print(f"Error at iter {i} for {sent1}, {sent2}")
                print(e)
        return False

    # TODO simplify this fn
    # pylint: disable=too-many-branches,too-many-locals,too-many-statements
    def __call__(self, row):  # noqa: C901
        try:
            pairs = []
            save_path = (
                Path(row["data_path"]) / DIALOGUE_PAIR_DIR / Path(row["path"]).name
            ).with_suffix(".json")
            if save_path.exists():
                return row

            diarization_path = (
                Path(row["data_path"]) / "diarization" / Path(row["path"]).name
            ).with_suffix(".pkl")
            if not diarization_path.exists():
                return row

            with diarization_path.open("rb") as fp:
                diarization = pickle.load(fp)
            segments = [dict(zip(x, y)) for x, y in row["segments"]]

            for seg in segments:
                seg["begin_time"] = float(seg["begin_time"])
                seg["end_time"] = float(seg["end_time"])

            for j, (seg1, seg2) in enumerate(pairwise(segments)):
                if seg2["begin_time"] - seg1["end_time"] > self.max_response_delay:
                    continue
                dia_result1 = diarization.crop(
                    Segment(seg1["begin_time"], seg1["end_time"])
                ).labels()
                dia_result2 = diarization.crop(
                    Segment(seg2["begin_time"], seg2["end_time"])
                ).labels()
                if len(dia_result1) != 1 or len(dia_result2) != 1:
                    continue

                if dia_result1[0] == dia_result2[0]:
                    continue

                sent1 = seg1["text_tn"]
                sent2 = seg2["text_tn"]
                seg1_begin = seg1["begin_time"]
                for backward_i in range(
                    j - 1, max(0, j - randint(1, self.max_rand_len)), -1
                ):
                    seg1_backward = segments[backward_i]
                    dia_backward = diarization.crop(
                        Segment(seg1_backward["begin_time"], seg1_backward["end_time"])
                    ).labels()
                    if len(dia_backward) != 1 or dia_backward[0] != dia_result1[0]:
                        break
                    if (
                        len(f'{seg1_backward["text_tn"]} {sent1}'.split(" "))
                        > self.max_token
                    ):
                        break
                    if (
                        seg1["begin_time"] - seg1_backward["end_time"]
                        > self.max_response_delay
                    ):
                        break
                    seg1_begin = seg1_backward["begin_time"]
                    sent1 = f'{seg1_backward["text_tn"]} {sent1}'

                seg2_end = seg2["end_time"]
                for forward_i in range(
                    j + 2, min(len(segments), j + 1 + randint(1, self.max_rand_len))
                ):
                    seg_forward = segments[forward_i]
                    dia_forward = diarization.crop(
                        Segment(seg_forward["begin_time"], seg_forward["end_time"])
                    ).labels()
                    if len(dia_forward) != 1 or dia_forward[0] != dia_result2[0]:
                        break
                    if (
                        len(f'{sent2} {seg_forward["text_tn"]}'.split(" "))
                        > self.max_token
                    ):
                        break
                    if (
                        seg_forward["begin_time"] - seg2["end_time"]
                        > self.max_response_delay
                    ):
                        break
                    seg2_end = seg_forward["end_time"]
                    sent2 = f'{sent2} {seg_forward["text_tn"]}'

                is_response = self.check_is_response(sent1, sent2)
                if not is_response:
                    continue

                pairs.append(
                    [
                        {
                            "begin_time": seg1_begin,
                            "end_time": seg1["end_time"],
                            "text_tn": sent1,
                            "audio_parent": row["path"],
                        },
                        {
                            "begin_time": seg2["begin_time"],
                            "end_time": seg2_end,
                            "text_tn": sent2,
                            "audio_parent": row["path"],
                        },
                    ]
                )

            save_path.parent.mkdir(exist_ok=True, parents=True)
            with save_path.open("w") as out_file:
                json.dump(pairs, out_file, sort_keys=True, indent=4, ensure_ascii=False)
        except Exception as e:  # pylint: disable=broad-exception-caught
            print("Error encountered")
            print(row["path"])
            print(e)

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

    def __call__(self, row, output_path):
        audio_fpath = Path(row["audio_fpath"])
        relative_audio_fpath = audio_fpath.relative_to(output_path)
        save_fpath = (
            Path(output_path) / "tokens" / Path(*relative_audio_fpath.parts[1:])
        ).with_suffix(".txt")
        save_fpath.parent.mkdir(parents=True, exist_ok=True)

        row["token_fpath"] = save_fpath
        if not audio_fpath.exists():
            return row
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


class WhisperASR:
    def __init__(self, device="cuda", dtype=torch.bfloat16):
        model_id = "distil-whisper/distil-large-v3"
        self.dtype = dtype
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=False,
            use_safetensors=True,
        )
        self.model.to(device)

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.device = device

    def __call__(self, row):
        try:
            fpath = Path(row["data_path"]) / row["path"]
            if not fpath.exists():
                return row

            save_path = (
                Path(row["output_path"])
                / "whisper_transcripts"
                / fpath.with_suffix(".json").name
            )
            if save_path.exists():
                return row
            save_path.parent.mkdir(parents=True, exist_ok=True)

            dwav, sampling_rate = torchaudio.load(fpath)
            if sampling_rate != 16000:
                dwav = torchaudio.functional.resample(
                    # dwav, sampling_rate, self.tokenizer.sample_rate
                    dwav,
                    sampling_rate,
                    16000,
                )

            dwav = dwav.mean(dim=0).numpy()

            inputs = self.processor(
                dwav, return_tensors="pt", truncation=False, sampling_rate=16_000
            )
            inputs = inputs.to(self.device, self.dtype)

            output = self.model.generate(
                **inputs,
                return_timestamps=True,
                return_segments=True,
                language="english",
                do_sample=True,
                temperature=0.2,
                compression_ratio_threshold=1.35,
                logprob_threshold=-1.0,
            )

            result = self.processor.batch_decode(
                output["sequences"], skip_special_tokens=True, output_offsets=True
            )

            for i in range(len(result[0]["offsets"])):
                result[0]["offsets"][i]["timestamp"] = (
                    output["segments"][0][i]["start"].item(),
                    output["segments"][0][i]["end"].item(),
                )
            filtered = list(filter(self.transcript_filter, result[0]["offsets"]))
            text = "".join([x["text"] for x in filtered])
            for item in filtered:
                item["text"] = item["text"].strip()
            results = {"text": text.strip(), "chunks": filtered}
            save_path.write_text(json.dumps(results, ensure_ascii=False, indent=4))
        except Exception as e:  # pylint: disable=broad-exception-caught
            print("Error encountered")
            print(row["path"])
            print(e)

        return row

    def transcript_filter(self, item):
        if None in item["timestamp"]:
            return False
        if item["timestamp"][1] - item["timestamp"][0] < 0.1:
            return False
        if not re.search(r"\w", item["text"]):
            return False
        return True


# pylint: disable-next=too-few-public-methods
class TTS:
    def __init__(self):
        self.model = ChatTTS.Chat()
        # No significant speedup with compile
        self.model.load(compile=False)
        # Seems to enhance stability
        self.model.speaker.std *= 0.5

        self.text_normalizer = Normalizer(
            input_case="cased",
            cache_dir=None,
            lang="en",
        )

    def __call__(self, row, output_path):
        save_path = Path(output_path) / "dialogue_audio" / str(row["original_index"])
        row["dialogue_audio"] = str(save_path)
        if save_path.exists() and len(list(save_path.glob("*.opus"))) == len(
            row["dialogue"]
        ):
            return row
        save_path.mkdir(parents=True, exist_ok=True)

        speakers = [self.model.sample_random_speaker() for _ in range(2)]
        for i, speaker in enumerate(speakers):
            texts = row["dialogue"][i::2]
            texts = [self.text_normalizer.normalize(x) for x in texts]
            params_infer_code = ChatTTS.Chat.InferCodeParams(
                spk_emb=speaker,  # add sampled speaker
                temperature=0.5,  # using custom temperature
                top_P=0.5,  # top P decode
                top_K=5,  # top K decode
                show_tqdm=False,
            )
            params_refine = ChatTTS.Chat.RefineTextParams(show_tqdm=False)
            wavs = self.model.infer(
                texts,
                params_infer_code=params_infer_code,
                params_refine_text=params_refine,
            )
            for j, wav in enumerate(wavs):
                wav = torchaudio.functional.resample(
                    torch.Tensor(wav), 24000, 16000
                ).cpu()
                wav = wav.unsqueeze(0)
                wav = librosa.effects.trim(wav)[0]

                torchaudio.save(
                    save_path / f"{i+j*2}.opus",
                    wav,
                    16000,
                )
        return row
