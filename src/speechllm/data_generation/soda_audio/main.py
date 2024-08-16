# pylint: disable=wrong-import-position, wrong-import-order
from pathlib import Path

from speechllm.utils import check_hpu

if check_hpu():
    import habana_frameworks.torch.core as htcore  # noqa: F401 pylint: disable=unused-import
    from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

    adapt_transformers_to_gaudi()


import click
import ray
from datasets import load_dataset

from speechllm.data_generation.processes import (  # noqa pylint: disable=unused-import,ungrouped-imports
    TTS,
    SpeechTokenizerGenerator,
)
from speechllm.data_generation.soda_audio.processes import (  # noqa pylint: disable=unused-import
    export_audio,
    export_json,
)


def print_row(row):
    print(row)
    return row


@click.command()
@click.option(
    "--output_path",
    default="./data/soda/",
    type=click.Path(dir_okay=True, path_type=Path),
)
@click.option("--nnodes", default=1)
@click.option("--node_id", default=0)
def main(output_path, nnodes, node_id):
    print(f"Running job {node_id} out of {nnodes} jobs")
    dataset = load_dataset("fixie-ai/soda-audio")
    for split_name, split in dataset.items():
        split_output_path = output_path / split_name
        split = split.filter(lambda _, idx: idx % nnodes == node_id, with_indices=True)
        ds = ray.data.from_huggingface(split, concurrency=16)
        # ds = ds.map(
        #     export_json,
        #     concurrency=64,
        #     num_cpus=4,
        #     fn_kwargs={"output_path": split_output_path},
        # )
        # ds = ds.map(
        #     export_audio,
        #     concurrency=64,
        #     num_cpus=4,
        #     fn_kwargs={"output_path": split_output_path},
        # )
        # ds = ds.map(
        #     SpeechTokenizerGenerator,
        #     concurrency=(4, 32),
        #     #num_cpus=16,
        #     fn_constructor_kwargs={"device": "cpu", "dtype": torch.float32},
        #     fn_kwargs={"output_path": split_output_path},
        # )
        ds = ds.map(
            TTS,
            concurrency=8,
            num_gpus=1 / 8,
            fn_kwargs={"output_path": split_output_path},
        )

        # ds = ds.map(
        #     SpeechTokenizerGenerator,
        #     concurrency=4,
        #     resources={"HPU": 1},
        #     fn_constructor_kwargs={"device": "hpu", "dtype": torch.bfloat16},
        #     fn_kwargs={"output_path": split_output_path},
        # )
        ds.materialize()


if __name__ == "__main__":
    # pylint: disable-next=no-value-for-parameter
    main()
