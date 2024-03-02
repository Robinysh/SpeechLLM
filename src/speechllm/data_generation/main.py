import click
import ray
from pyarrow import csv

from speechllm.data_generation.processes import (
    AudioEnhancer,
    add_cols,
    diarization,
    download_audio,
    filter_dialogues,
    split_dialogues,
)


@click.command()
@click.option("--info_path", default="")
@click.option("--data_path", default="")
def main(info_path, data_path):
    parse_options = csv.ParseOptions(delimiter="\t")
    ds = ray.data.read_csv(
        info_path,
        parse_options=parse_options,
        shuffle=True,
    )
    ds = ds.map(add_cols, fn_kwargs={"cols": {}})
    ds = ds.map(download_audio, fn_kwargs={"cols": data_path}, concurrency=4)
    ds = ds.map(AudioEnhancer, fn_kwargs={"cols": data_path}, concurrency=2)
    # ds = ds.map(generate_hubert_embs)
    # write metadata.tsv
    ds = ds.map(diarization, fn_kwargs={"cols": data_path}, concurrency=4)
    ds = ds.map(split_dialogues, fn_kwargs={"cols": data_path}, concurrency=8)
    ds = ds.map(filter_dialogues, fn_kwargs={"cols": data_path}, concurrency=8)


if __name__ == "__main__":
    main()
