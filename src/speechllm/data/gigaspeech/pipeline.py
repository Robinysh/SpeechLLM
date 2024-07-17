from pathlib import Path

from datasets import load_dataset

from speechllm.data.gigaspeech import processes
from speechllm.data.prompter import COTPrompter, Prompter


def gigaspeech_pipeline(
    database,
    num_workers=16,
    cache_dir=None,
    use_cache=True,
    stage="train",
):
    prompter = Prompter()
    cache_dir = Path(cache_dir)

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
        processes.filter_long_audio,
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        cache_file_name=str(cache_dir / "filtered" / f"{stage}.arrow"),
        desc="filtering long audio",
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

    Path(cache_dir / "transcript").mkdir(parents=True, exist_ok=True)
    ds = ds.map(
        lambda x: processes.read_transcript(Path(database["transcript"]) / x["fname"]),
        batched=False,
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        desc="reading transcripts",
        cache_file_name=str(cache_dir / "transcript" / f"{stage}.arrow"),
    )

    Path(cache_dir / "infer_template").mkdir(parents=True, exist_ok=True)
    ds = ds.map(
        lambda x: {"infer_prompt": prompter.generate_template(x["input_tokens"])},
        batched=False,
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        cache_file_name=str(cache_dir / "infer_template" / f"{stage}.arrow"),
        desc="generating infer prompts",
    )

    Path(cache_dir / "template").mkdir(parents=True, exist_ok=True)
    ds = ds.map(
        lambda x: {
            "prompt": prompter.generate_template(x["input_tokens"], x["output_tokens"])
        },
        batched=False,
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        cache_file_name=str(cache_dir / "template" / f"{stage}.arrow"),
        desc="generating prompts",
    )

    # ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

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

    Path(cache_dir / "infer_template").mkdir(parents=True, exist_ok=True)
    ds = ds.map(
        lambda x: {"infer_prompt": prompter.generate_template(x["input_tokens"])},
        batched=False,
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        cache_file_name=str(cache_dir / "infer_template" / f"{stage}.arrow"),
        desc="generating infer prompts",
    )

    Path(cache_dir / "template").mkdir(parents=True, exist_ok=True)
    ds = ds.map(
        lambda x: {
            "prompt": prompter.generate_template(
                x["input_tokens"],
                x["input_transcript"],
                x["output_tokens"],
                x["output_transcript"],
            )
        },
        batched=False,
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        cache_file_name=str(cache_dir / "template" / f"{stage}.arrow"),
        desc="generating prompts",
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
    ds = ds.shuffle(seed=42, buffer_size=100)

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
