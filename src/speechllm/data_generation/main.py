import click
import ray
import torch
from speechcolab.datasets.gigaspeech import GigaSpeech

from speechllm.data_generation.processes import Downloader, add_cols


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
@click.option("--data_path", default="./data/GigaSpeech/download/S")
@click.option("--subset", default="{S}")
def main(data_path, output_path, subset):
    # for i in torch.utils.data.DataLoader(GigaSpeechDataset(data_path, subset)):
    #    print(i)
    #    break

    # parse_options = csv.ParseOptions(delimiter="\t")
    # ds = ray.data.read_csv(
    #    info_path,
    #    parse_options=parse_options,
    #    shuffle=True,
    # )
    # gigaspeech = list(GigaSpeech(data_path))
    gigaspeech = list(GigaSpeech(data_path).audios(subset))
    for row in gigaspeech:
        del row["segments"]
    data = gigaspeech
    # gigaspeech = GigaSpeech(data_path).audios(subset)
    # data = []
    # for i in tqdm(range(5)):
    # while True:
    #    x = next(gigaspeech)
    #    del x['segments']
    #    #x['test'] = [{}]
    #    #ic(x)
    #    data.append(x)
    # ic(data[0])
    # python cannot pickle generator so we have to convert GigaSpeech to a list
    ds = ray.data.from_items(data)
    # ds = ray.data.from_torch(GigaSpeechDataset(data_path, subset))
    # ds = ds.map(extract_rows, fn_kwargs={"cols": data_path})
    ds = ds.map(
        add_cols,
        fn_kwargs={"cols": {"data_path": data_path, "output_path": output_path}},
    )
    # ds = ds.map(print_row)
    ds = ds.map(Downloader, concurrency=4, fn_constructor_args={"data_path": data_path})
    # ds = ds.map(AudioEnhancer, num_gpus=1, concurrency=1)
    # ds = ds.map(generate_hubert_embs)
    # write metadata.tsv
    # ds = ds.map(Diarizer, concurrency=2)
    # ds = ds.map(split_dialogues)
    # ds = ds.map(filter_dialogues)
    ds.materialize()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
