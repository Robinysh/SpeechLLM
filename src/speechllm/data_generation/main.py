import click
import ray
import torch
from speechcolab.datasets.gigaspeech import GigaSpeech

from speechllm.data_generation.processes import add_cols, split_dialogues


# pylint: disable=abstract-method,too-few-public-methods
class GigaSpeechDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_path, subset="{XS}"):
        super().__init__()
        self.gigaspeech = GigaSpeech(data_path).audios(subset)

    def __iter__(self):
        return iter(self.gigaspeech)


def print_row(row):
    print(row)
    return row


@click.command()
@click.option(
    "--output_path", default="./data/GigaSpeech/download/extracted/cleaned_flac"
)
@click.option("--data_path", default="./data/GigaSpeech/download/XS")
@click.option("--subset", default="{L}")
@click.option("--nnodes", default=1)
@click.option("--node_id", default=0)
def main(data_path, output_path, subset, nnodes, node_id):
    gigaspeech = GigaSpeech(data_path).audios(subset)
    gigaspeech = [x for i, x in enumerate(gigaspeech) if i % nnodes == node_id]
    # pyarrow is quite restrictive in terms of what it can serialize
    # so we have to transform some fields
    for row in gigaspeech:
        del row["subsets"]
        del row["speaker"]
        row["duration"] = float(row["duration"])
        for r in row["segments"]:
            del r["subsets"]
            del r["speaker"]
            r["begin_time"] = float(r["begin_time"])
            r["end_time"] = float(r["end_time"])
        row["segments"] = [[list(r.keys()), list(r.values())] for r in row["segments"]]
    # python cannot pickle generator so we have to convert GigaSpeech to a list
    ds = ray.data.from_items(gigaspeech)

    ds = ds.map(
        add_cols,
        fn_kwargs={"cols": {"data_path": data_path, "output_path": output_path}},
    )
    # ds = ds.map(print_row)
    # ds = ds.map(Downloader, concurrency=4, fn_constructor_args={"data_path": data_path})
    # ds = ds.map(AudioEnhancer, num_gpus=1, concurrency=1)
    # ds = ds.map(generate_hubert_embs)
    # write metadata.tsv
    # ds = ds.map(Diarizer, concurrency=8,  num_gpus=0.5)
    # ds = ds.map(DialogueFilter, concurrency=4)
    ds = ds.map(split_dialogues)
    ds.materialize()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
