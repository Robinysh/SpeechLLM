import ray
from pyarrow import csv

from speechllm.data import processes


def baseline(database):
    parse_options = csv.ParseOptions(delimiter="\t")
    ds = ray.data.read_csv(
        database["metadata"],
        parse_options=parse_options,
        shuffle=True,
    )
    ds = ds.random_shuffle()
    ds = ds.map(processes.add_cols, fn_kwargs={"cols": database})
    ds = ds.map(processes.rename_cols)
    ds = ds.map(processes.load_audio)
    ds = ds.map(processes.Tokenize, concurrency=16)
    ds = ds.map(processes.load_hubert)
    ds = ds.map(processes.filter_outputs)
    return ds