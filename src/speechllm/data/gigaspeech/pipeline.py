from pathlib import Path

import datasets
from datasets import load_dataset

from speechllm.data.gigaspeech import processes
from speechllm.data.prompter import COTPrompter, Prompter
from speechllm.data.utils import WrapInputOutput


def gigaspeech_pipeline(
    database,
    num_workers=16,
    cache_dir=None,
    use_cache=True,
    stage="train",
):
    prompter = Prompter()
    cache_dir = Path(cache_dir) / "gigaspeech_nocot"

    ds = load_dataset("text", data_files=database["metadata"])["train"]
    ds = ds.rename_column("text", "fname")

    Path(cache_dir / "audio_tokens").mkdir(parents=True, exist_ok=True)
    ds = ds.map(
        lambda x: processes.read_audio_tokens(x, Path(database["tokens"])),
        batched=False,
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        cache_file_name=str(cache_dir / "audio_tokens" / f"{stage}.arrow"),
        desc="reading audio tokens",
    )

    Path(cache_dir / "filtered").mkdir(parents=True, exist_ok=True)
    ds = ds.filter(
        lambda x: processes.filter_long_audio(x, limit=1000),
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        cache_file_name=str(cache_dir / "filtered" / f"{stage}.arrow"),
        desc="filtering long audio",
    )

    Path(cache_dir / "transcript").mkdir(parents=True, exist_ok=True)
    ds = ds.map(
        lambda x: processes.read_transcript(Path(database["transcript"]) / x["fname"]),
        batched=False,
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        desc="reading transcripts",
        cache_file_name=str(cache_dir / "transcript" / f"{stage}.arrow"),
    )

    Path(cache_dir / "audio").mkdir(parents=True, exist_ok=True)
    ds = ds.map(
        lambda x: processes.load_paired_audio(Path(database["audio"]) / x["fname"]),
        batched=False,
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        cache_file_name=str(cache_dir / "audio" / f"{stage}.arrow"),
        desc="reading audio",
    )
    ds = ds.to_iterable_dataset(num_shards=32)
    ds = ds.shuffle(seed=42, buffer_size=64)
    ds = ds.map(
        WrapInputOutput(
            prompter.generate_template,
            kwarg_maps={
                "input_tokens": "input_tokens",
                "output_tokens": "output_tokens",
            },
            output_name="prompt",
        )
    )

    ds = ds.map(
        WrapInputOutput(
            prompter.generate_template,
            kwarg_maps={
                "input_tokens": "input_tokens",
            },
            output_name="infer_prompt",
        )
    )

    ds.select_columns(
        [
            "prompt",
            "infer_prompt",
            "input_audio",
            "input_tokens",
            "input_transcript",
            "output_audio",
            "output_tokens",
            "output_transcript",
        ],
    )
    ds = ds._resolve_features()  # pylint: disable=protected-access
    ds = ds.cast_column(
        "input_audio", datasets.Sequence(datasets.Value(dtype="float32"))
    )
    ds = ds.cast_column(
        "output_audio", datasets.Sequence(datasets.Value(dtype="float32"))
    )

    return ds


def gigaspeech_asr_tts_cot_pipeline(
    database,
    num_workers=16,
    cache_dir=None,
    use_cache=True,
    stage="train",
):
    prompter = COTPrompter()
    cache_dir = Path(cache_dir) / "text_interface"

    ds = load_dataset("text", data_files=database["metadata"])["train"]
    ds = ds.rename_column("text", "fname")

    Path(cache_dir / "audio_tokens").mkdir(parents=True, exist_ok=True)
    ds = ds.map(
        lambda x: processes.read_audio_tokens(x, Path(database["tokens"])),
        batched=False,
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        cache_file_name=str(cache_dir / "audio_tokens" / f"{stage}.arrow"),
        desc="reading audio tokens",
    )

    Path(cache_dir / "filtered").mkdir(parents=True, exist_ok=True)
    ds = ds.filter(
        lambda x: processes.filter_long_audio(x, limit=1000),
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        cache_file_name=str(cache_dir / "filtered" / f"{stage}.arrow"),
        desc="filtering long audio",
    )

    Path(cache_dir / "transcript").mkdir(parents=True, exist_ok=True)
    ds = ds.map(
        lambda x: processes.read_transcript(Path(database["transcript"]) / x["fname"]),
        batched=False,
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        desc="reading transcripts",
        cache_file_name=str(cache_dir / "transcript" / f"{stage}.arrow"),
    )

    Path(cache_dir / "audio").mkdir(parents=True, exist_ok=True)
    ds = ds.map(
        lambda x: processes.load_paired_audio(Path(database["audio"]) / x["fname"]),
        batched=False,
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        cache_file_name=str(cache_dir / "audio" / f"{stage}.arrow"),
        desc="reading audio",
    )
    ds = ds.to_iterable_dataset(num_shards=32)
    ds = ds.shuffle(seed=42, buffer_size=64)
    ds = ds.map(
        WrapInputOutput(
            prompter.generate_template,
            kwarg_maps={
                "input_tokens": "input_tokens",
                "input_transcript": "input_transcript",
                "output_tokens": "output_tokens",
                "output_transcript": "output_transcript",
            },
            output_name="prompt",
        )
    )

    ds = ds.map(
        WrapInputOutput(
            prompter.generate_template,
            kwarg_maps={
                "input_tokens": "input_tokens",
            },
            output_name="infer_prompt",
        )
    )

    ds.select_columns(
        [
            "prompt",
            "infer_prompt",
            "input_audio",
            "input_tokens",
            "input_transcript",
            "output_audio",
            "output_tokens",
            "output_transcript",
        ],
    )
    ds = ds._resolve_features()  # pylint: disable=protected-access
    ds = ds.cast_column(
        "input_audio", datasets.Sequence(datasets.Value(dtype="float32"))
    )
    ds = ds.cast_column(
        "output_audio", datasets.Sequence(datasets.Value(dtype="float32"))
    )

    return ds


def gigaspeech_asr_tts_implicit_cot_pipeline(
    database,
    num_workers=16,
    cache_dir=None,
    use_cache=True,
    stage="train",
):
    prompter = COTPrompter()
    cache_dir = Path(cache_dir) / "text_interface_implicit"

    ds = load_dataset("text", data_files=database["metadata"])["train"]
    ds = ds.rename_column("text", "fname")

    Path(cache_dir / "audio_tokens").mkdir(parents=True, exist_ok=True)
    ds = ds.map(
        lambda x: processes.read_audio_tokens(x, Path(database["tokens"])),
        batched=False,
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        cache_file_name=str(cache_dir / "audio_tokens" / f"{stage}.arrow"),
        desc="reading audio tokens",
    )

    Path(cache_dir / "filtered").mkdir(parents=True, exist_ok=True)
    ds = ds.filter(
        lambda x: processes.filter_long_audio(x, limit=1000),
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        cache_file_name=str(cache_dir / "filtered" / f"{stage}.arrow"),
        desc="filtering long audio",
    )

    Path(cache_dir / "transcript").mkdir(parents=True, exist_ok=True)
    ds = ds.map(
        lambda x: processes.read_transcript(Path(database["transcript"]) / x["fname"]),
        batched=False,
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        desc="reading transcripts",
        cache_file_name=str(cache_dir / "transcript" / f"{stage}.arrow"),
    )

    Path(cache_dir / "audio").mkdir(parents=True, exist_ok=True)
    ds = ds.map(
        lambda x: processes.load_paired_audio(Path(database["audio"]) / x["fname"]),
        batched=False,
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        cache_file_name=str(cache_dir / "audio" / f"{stage}.arrow"),
        desc="reading audio",
    )
    ds = ds.to_iterable_dataset(num_shards=32)
    ds = ds.shuffle(seed=42, buffer_size=64)
    ds = ds.map(
        WrapInputOutput(
            prompter.generate_implicit_template,
            kwarg_maps={
                "input_tokens": "input_tokens",
                "input_transcript": "input_transcript",
                "output_tokens": "output_tokens",
                "output_transcript": "output_transcript",
            },
            output_name="prompt",
        )
    )

    ds = ds.map(
        WrapInputOutput(
            prompter.generate_implicit_template,
            kwarg_maps={
                "input_tokens": "input_tokens",
            },
            output_name="infer_prompt",
        )
    )

    ds.select_columns(
        [
            "prompt",
            "infer_prompt",
            "input_audio",
            "input_tokens",
            "input_transcript",
            "output_audio",
            "output_tokens",
            "output_transcript",
        ],
    )
    ds = ds._resolve_features()  # pylint: disable=protected-access
    ds = ds.cast_column(
        "input_audio", datasets.Sequence(datasets.Value(dtype="float32"))
    )
    ds = ds.cast_column(
        "output_audio", datasets.Sequence(datasets.Value(dtype="float32"))
    )

    return ds


if __name__ == "__main__":
    print(
        gigaspeech_asr_tts_cot_pipeline(
            {
                "metadata": "/scratch-1/robinysh/GigaSpeech/test_files.txt",
                "audio": "/scratch-1/robinysh/GigaSpeech/audio_pairs",
                "transcript": "/scratch-1/robinysh/GigaSpeech/dialogue_pairs",
                "tokens": "/scratch-1/robinysh/GigaSpeech/tokens",
            },
            num_workers=16,
            use_cache=False,
            cache_dir="/scratch-1/robinysh/cache/huggingface/debug",
        )[0]
    )
