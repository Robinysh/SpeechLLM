from functools import partial
from pathlib import Path

import datasets
from datasets import load_dataset

from speechllm.data.prompter import SodaASRTTSCOTPrompter, SodaCOTPrompter, SodaPrompter
from speechllm.data.soda import processes
from speechllm.data.utils import WrapInputOutput


def soda_pipeline(
    database,
    num_workers=16,
    cache_dir=None,
    use_cache=True,
    stage="train",
):
    prompter = SodaPrompter()
    cache_dir = Path(cache_dir) / "soda"

    ds = load_dataset("json", data_dir=database["json"])["train"]
    ds = ds.rename_column("original_index", "fname")
    ds = ds.cast_column("fname", datasets.Value(dtype="string"))

    Path(cache_dir / "context").mkdir(parents=True, exist_ok=True)
    ds = ds.map(
        lambda x: {"context": x["dialogue"][:-2]},
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        cache_file_name=str(cache_dir / "context" / f"{stage}.arrow"),
        desc="extract context",
    )

    Path(cache_dir / "output_transcript").mkdir(parents=True, exist_ok=True)
    ds = ds.map(
        lambda x: {"output_transcript": x["dialogue"][-2]},
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        cache_file_name=str(cache_dir / "output_transcript" / f"{stage}.arrow"),
        desc="extract response transcript",
    )

    Path(cache_dir / "audio_tokens").mkdir(parents=True, exist_ok=True)
    ds = ds.map(
        lambda x: processes.read_audio_tokens(
            Path(database["tokens"]) / f"{x['fname']}.txt"
        ),
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        cache_file_name=str(cache_dir / "audio_tokens" / f"{stage}.arrow"),
        desc="reading audio tokens",
    )

    Path(cache_dir / "filtered").mkdir(parents=True, exist_ok=True)
    ds = ds.filter(
        lambda x: processes.filter_long_audio(x, limit=701),
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        cache_file_name=str(cache_dir / "filtered" / f"{stage}.arrow"),
        desc="filtering long audio",
    )

    Path(cache_dir / "audio").mkdir(parents=True, exist_ok=True)
    ds = ds.map(
        lambda x: processes.load_audio(Path(database["audio"]) / f"{x['fname']}.opus"),
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        cache_file_name=str(cache_dir / "audio" / f"{stage}.arrow"),
        desc="reading audio",
    )

    # Operations below dont cache and run on-the-fly
    ds = ds.to_iterable_dataset(num_shards=32)
    ds = ds.shuffle(seed=42, buffer_size=100)
    prompt_wrapped = WrapInputOutput(
        prompter.generate_template,
        kwarg_maps={
            "context": "context",
            "response_tokens": "output_tokens",
        },
        output_name="prompt",
    )
    ds = ds.map(prompt_wrapped)

    infer_prompt_wrapped = WrapInputOutput(
        prompter.generate_template,
        kwarg_maps={
            "context": "context",
        },
        output_name="infer_prompt",
    )
    ds = ds.map(infer_prompt_wrapped)

    ds = ds.select_columns(
        [
            "prompt",
            "infer_prompt",
            "output_audio",
            "output_tokens",
            "output_transcript",
        ]
    )
    ds = ds._resolve_features()  # pylint: disable=protected-access
    ds = ds.cast_column(
        "output_audio", datasets.Sequence(datasets.Value(dtype="float32"))
    )

    return ds


def soda_tts_cot_pipeline(
    database,
    num_workers=16,
    cache_dir=None,
    use_cache=True,
    stage="train",
):
    prompter = SodaCOTPrompter()
    cache_dir = Path(cache_dir) / "soda"

    ds = load_dataset("json", data_dir=database["json"])["train"]
    ds = ds.rename_column("original_index", "fname")
    ds = ds.cast_column("fname", datasets.Value(dtype="string"))

    Path(cache_dir / "context").mkdir(parents=True, exist_ok=True)
    ds = ds.map(
        lambda x: {"context": x["dialogue"][:-2]},
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        cache_file_name=str(cache_dir / "context" / f"{stage}.arrow"),
        desc="extract context",
    )

    Path(cache_dir / "output_transcript").mkdir(parents=True, exist_ok=True)
    ds = ds.map(
        lambda x: {"output_transcript": x["dialogue"][-2]},
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        cache_file_name=str(cache_dir / "output_transcript" / f"{stage}.arrow"),
        desc="extract response transcript",
    )

    Path(cache_dir / "audio_tokens").mkdir(parents=True, exist_ok=True)
    ds = ds.map(
        lambda x: processes.read_audio_tokens(
            Path(database["tokens"]) / f"{x['fname']}.txt"
        ),
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        cache_file_name=str(cache_dir / "audio_tokens" / f"{stage}.arrow"),
        desc="reading audio tokens",
    )

    Path(cache_dir / "filtered").mkdir(parents=True, exist_ok=True)
    ds = ds.filter(
        lambda x: processes.filter_long_audio(x, limit=701),
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        cache_file_name=str(cache_dir / "filtered" / f"{stage}.arrow"),
        desc="filtering long audio",
    )

    Path(cache_dir / "audio").mkdir(parents=True, exist_ok=True)
    ds = ds.map(
        lambda x: processes.load_audio(Path(database["audio"]) / f"{x['fname']}.opus"),
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        cache_file_name=str(cache_dir / "audio" / f"{stage}.arrow"),
        desc="reading audio",
    )

    # Operations below dont cache and run on-the-fly
    ds = ds.to_iterable_dataset(num_shards=32)
    ds = ds.shuffle(seed=42, buffer_size=100)
    prompt_wrapped = WrapInputOutput(
        prompter.generate_template,
        kwarg_maps={
            "context": "context",
            "response_tokens": "output_tokens",
            "output_transcript": "output_transcript",
        },
        output_name="prompt",
    )
    ds = ds.map(prompt_wrapped)

    infer_prompt_wrapped = WrapInputOutput(
        prompter.generate_template,
        kwarg_maps={
            "context": "context",
        },
        output_name="infer_prompt",
    )
    ds = ds.map(infer_prompt_wrapped)

    ds = ds.select_columns(
        [
            "prompt",
            "infer_prompt",
            "output_audio",
            "output_tokens",
            "output_transcript",
        ]
    )
    ds = ds._resolve_features()  # pylint: disable=protected-access
    ds = ds.cast_column(
        "output_audio", datasets.Sequence(datasets.Value(dtype="float32"))
    )

    return ds


def soda_tts_implicit_cot_pipeline(
    database,
    num_workers=16,
    cache_dir=None,
    use_cache=True,
    stage="train",
):
    prompter = SodaCOTPrompter()
    cache_dir = Path(cache_dir) / "soda_implicit"

    ds = load_dataset("json", data_dir=database["json"])["train"]
    ds = ds.rename_column("original_index", "fname")
    ds = ds.cast_column("fname", datasets.Value(dtype="string"))

    Path(cache_dir / "context").mkdir(parents=True, exist_ok=True)
    ds = ds.map(
        lambda x: {"context": x["dialogue"][:-2]},
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        cache_file_name=str(cache_dir / "context" / f"{stage}.arrow"),
        desc="extract context",
    )

    Path(cache_dir / "output_transcript").mkdir(parents=True, exist_ok=True)
    ds = ds.map(
        lambda x: {"output_transcript": x["dialogue"][-2]},
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        cache_file_name=str(cache_dir / "output_transcript" / f"{stage}.arrow"),
        desc="extract response transcript",
    )

    Path(cache_dir / "audio_tokens").mkdir(parents=True, exist_ok=True)
    ds = ds.map(
        lambda x: processes.read_audio_tokens(
            Path(database["tokens"]) / f"{x['fname']}.txt"
        ),
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        cache_file_name=str(cache_dir / "audio_tokens" / f"{stage}.arrow"),
        desc="reading audio tokens",
    )

    Path(cache_dir / "filtered").mkdir(parents=True, exist_ok=True)
    ds = ds.filter(
        lambda x: processes.filter_long_audio(x, limit=700),
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        cache_file_name=str(cache_dir / "filtered" / f"{stage}.arrow"),
        desc="filtering long audio",
    )

    Path(cache_dir / "audio").mkdir(parents=True, exist_ok=True)
    ds = ds.map(
        lambda x: processes.load_audio(Path(database["audio"]) / f"{x['fname']}.opus"),
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        cache_file_name=str(cache_dir / "audio" / f"{stage}.arrow"),
        desc="reading audio",
    )

    # Operations below dont cache and run on-the-fly
    ds = ds.to_iterable_dataset(num_shards=32)
    ds = ds.shuffle(seed=42, buffer_size=64)
    prompt_wrapped = WrapInputOutput(
        prompter.generate_implicit_template,
        kwarg_maps={
            "context": "context",
            "response_tokens": "output_tokens",
            "output_transcript": "output_transcript",
        },
        output_name="prompt",
    )
    ds = ds.map(prompt_wrapped)

    infer_prompt_wrapped = WrapInputOutput(
        prompter.generate_implicit_template,
        kwarg_maps={
            "context": "context",
        },
        output_name="infer_prompt",
    )
    ds = ds.map(infer_prompt_wrapped)

    ds = ds.select_columns(
        [
            "prompt",
            "infer_prompt",
            "output_audio",
            "output_tokens",
            "output_transcript",
        ]
    )
    ds = ds._resolve_features()  # pylint: disable=protected-access
    ds = ds.cast_column(
        "output_audio", datasets.Sequence(datasets.Value(dtype="float32"))
    )

    return ds


def soda_asr_tts_cot_pipeline(
    database,
    num_workers=16,
    cache_dir=None,
    use_cache=True,
    stage="train",
):
    prompter = SodaASRTTSCOTPrompter()
    cache_dir = Path(cache_dir) / "soda_asr_tts"

    ds = load_dataset("json", data_dir=database["json"])["train"]
    ds = ds.rename_column("original_index", "fname")
    ds = ds.cast_column("fname", datasets.Value(dtype="string"))

    ds = ds.rename_column("dialogue", "context")

    Path(cache_dir / "audio_tokens").mkdir(parents=True, exist_ok=True)
    ds = ds.map(
        lambda x: {
            "audio_tokens": [
                processes.read_audio_tokens(fpath)["output_tokens"]
                for fpath in sorted(
                    (Path(database["tts_tokens"]) / x["fname"]).glob("*.txt")
                )
            ]
        },
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        cache_file_name=str(cache_dir / "audio_tokens" / f"{stage}.arrow"),
        desc="reading audio tokens",
    )

    Path(cache_dir / "filtered").mkdir(parents=True, exist_ok=True)
    ds = ds.filter(
        lambda x: all(
            processes.filter_long_audio({"input_tokens": tokens}, limit=700)
            for tokens in x["audio_tokens"]
        ),
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        cache_file_name=str(cache_dir / "filtered" / f"{stage}.arrow"),
        desc="filtering long audio",
    )

    # Operations below dont cache and run on-the-fly
    ds = ds.to_iterable_dataset(num_shards=32)
    ds = ds.shuffle(seed=42, buffer_size=64)

    ds = ds.map(
        WrapInputOutput(
            processes.sample_context_interval,
            kwarg_maps={
                "context": "context",
            },
            output_name="context_interval",
        )
    )

    ds = ds.map(
        WrapInputOutput(
            prompter.generate_template,
            kwarg_maps={
                "audio_tokens": "audio_tokens",
                "context": "context",
                "context_interval": "context_interval",
            },
            output_name="prompt",
        )
    )

    ds = ds.map(
        WrapInputOutput(
            partial(prompter.generate_template, inference=True),
            kwarg_maps={
                "audio_tokens": "audio_tokens",
                "context": "context",
                "context_interval": "context_interval",
            },
            output_name="infer_prompt",
        )
    )

    ds = ds.map(
        WrapInputOutput(
            partial(processes.get_input_audio, path_root=database["dialogue_audio"]),
            kwarg_maps={
                "fname": "fname",
                "context_interval": "context_interval",
            },
            output_name="input_audio",
        )
    )

    ds = ds.map(
        WrapInputOutput(
            partial(processes.get_output_audio, path_root=database["dialogue_audio"]),
            kwarg_maps={
                "fname": "fname",
                "context_interval": "context_interval",
            },
            output_name="output_audio",
        )
    )

    ds = ds.map(
        WrapInputOutput(
            processes.get_output_transcript,
            kwarg_maps={
                "context": "context",
                "context_interval": "context_interval",
            },
            output_name="output_transcript",
        )
    )

    ds = ds.select_columns(
        [
            "prompt",
            "infer_prompt",
            "input_audio",
            "output_audio",
            "output_transcript",
        ]
    )
    ds = ds._resolve_features()  # pylint: disable=protected-access
    ds = ds.cast_column(
        "input_audio",
        datasets.Sequence(datasets.Value(dtype="float32")),
    )
    ds = ds.cast_column(
        "output_audio",
        datasets.Sequence(datasets.Value(dtype="float32")),
    )

    return ds


def soda_asr_tts_implicit_cot_pipeline(
    database,
    num_workers=16,
    cache_dir=None,
    use_cache=True,
    stage="train",
):
    prompter = SodaASRTTSCOTPrompter()
    cache_dir = Path(cache_dir) / "soda_asr_tts_implicit"

    ds = load_dataset("json", data_dir=database["json"])["train"]
    ds = ds.rename_column("original_index", "fname")
    ds = ds.cast_column("fname", datasets.Value(dtype="string"))

    ds = ds.rename_column("dialogue", "context")

    Path(cache_dir / "audio_tokens").mkdir(parents=True, exist_ok=True)
    ds = ds.map(
        lambda x: {
            "audio_tokens": [
                processes.read_audio_tokens(fpath)["output_tokens"]
                for fpath in sorted(
                    (Path(database["tts_tokens"]) / x["fname"]).glob("*.txt")
                )
            ]
        },
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        cache_file_name=str(cache_dir / "audio_tokens" / f"{stage}.arrow"),
        desc="reading audio tokens",
    )

    Path(cache_dir / "filtered").mkdir(parents=True, exist_ok=True)
    ds = ds.filter(
        lambda x: all(
            processes.filter_long_audio({"input_tokens": tokens}, limit=500)
            for tokens in x["audio_tokens"]
        ),
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        cache_file_name=str(cache_dir / "filtered" / f"{stage}.arrow"),
        desc="filtering long audio",
    )

    # Operations below dont cache and run on-the-fly
    ds = ds.to_iterable_dataset(num_shards=32)
    ds = ds.shuffle(seed=41, buffer_size=512)

    ds = ds.map(
        WrapInputOutput(
            processes.sample_context_interval,
            kwarg_maps={
                "context": "context",
            },
            output_name="context_interval",
        )
    )

    ds = ds.map(
        WrapInputOutput(
            prompter.generate_implicit_template,
            kwarg_maps={
                "audio_tokens": "audio_tokens",
                "context": "context",
                "context_interval": "context_interval",
            },
            output_name="prompt",
        )
    )

    ds = ds.map(
        WrapInputOutput(
            partial(prompter.generate_implicit_template, inference=True),
            kwarg_maps={
                "audio_tokens": "audio_tokens",
                "context": "context",
                "context_interval": "context_interval",
            },
            output_name="infer_prompt",
        )
    )

    ds = ds.map(
        WrapInputOutput(
            prompter.generate_template,
            kwarg_maps={
                "audio_tokens": "audio_tokens",
                "context": "context",
                "context_interval": "context_interval",
            },
            output_name="teacher_prompt",
        )
    )

    ds = ds.map(
        WrapInputOutput(
            partial(prompter.generate_template, teacher=True),
            kwarg_maps={
                "audio_tokens": "audio_tokens",
                "context": "context",
                "context_interval": "context_interval",
            },
            output_name="teacher_infer_prompt",
        )
    )

    ds = ds.map(
        WrapInputOutput(
            partial(processes.get_input_audio, path_root=database["dialogue_audio"]),
            kwarg_maps={
                "fname": "fname",
                "context_interval": "context_interval",
            },
            output_name="input_audio",
        )
    )

    ds = ds.map(
        WrapInputOutput(
            partial(processes.get_output_audio, path_root=database["dialogue_audio"]),
            kwarg_maps={
                "fname": "fname",
                "context_interval": "context_interval",
            },
            output_name="output_audio",
        )
    )

    ds = ds.map(
        WrapInputOutput(
            processes.get_input_transcript,
            kwarg_maps={
                "context": "context",
                "context_interval": "context_interval",
            },
            output_name="input_transcript",
        )
    )

    ds = ds.map(
        WrapInputOutput(
            processes.get_output_transcript,
            kwarg_maps={
                "context": "context",
                "context_interval": "context_interval",
            },
            output_name="output_transcript",
        )
    )

    ds = ds.select_columns(
        [
            "prompt",
            "infer_prompt",
            "teacher_prompt",
            "teacher_infer_prompt",
            "input_audio",
            "input_transcript",
            "output_audio",
            "output_transcript",
        ]
    )
    ds = ds._resolve_features()  # pylint: disable=protected-access
    ds = ds.cast_column(
        "input_audio",
        datasets.Sequence(datasets.Value(dtype="float32")),
    )
    ds = ds.cast_column(
        "output_audio",
        datasets.Sequence(datasets.Value(dtype="float32")),
    )

    return ds


if __name__ == "__main__":
    row = soda_tts_cot_pipeline(
        {
            "audio": "/scratch-1/robinysh/soda/test/audio",
            "json": "/scratch-1/robinysh/soda/test/json",
            "tokens": "/scratch-1/robinysh/soda/test/tokens",
        },
        num_workers=16,
        use_cache=False,
        cache_dir="/scratch-1/robinysh/cache/debug",
    )[0]
    row.pop("output_audio")
    print(row)
