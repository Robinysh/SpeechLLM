from pathlib import Path

import datasets
from datasets import load_dataset

from speechllm.data.prompter import SodaCOTPrompter
from speechllm.data.soda import processes
from speechllm.utils import dict_list_to_list_dict, list_dict_to_dict_list


def transform(batch):
    prompter = SodaCOTPrompter()
    batch = dict_list_to_list_dict(batch)

    for item in batch:
        item["prompt"] = prompter.generate_template(
            context=item["context"],
            response_tokens=item["output_tokens"],
            output_transcript=item["output_transcript"],
        )
        item["infer_prompt"] = prompter.generate_template(context=item["context"])

    batch = list_dict_to_dict_list(batch)
    return batch


class TransformDS:
    def __init__(self, ds, fn):
        super().__init__()
        self.ds = ds
        self.fn = fn

    def __iter__(self):
        return self

    def __next__(self):
        return self.fn(self.ds.__next__())


# pylint: disable=too-few-public-methods
class WrapInputOutput:
    def __init__(self, fn, kwarg_maps, output_name):
        self.fn = fn
        self.kwarg_maps = kwarg_maps
        self.output_name = output_name

    def __call__(self, input_dict):
        kwargs = {k: input_dict[v] for k, v in self.kwarg_maps.items()}
        return {self.output_name: self.fn(**kwargs)}


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
