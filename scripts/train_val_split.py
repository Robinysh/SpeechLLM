import re
from pathlib import Path
from random import shuffle

import click


def duplicate_pairs(fpaths):
    return sum(
        [[x.with_stem(f"{x.stem}_1"), x.with_stem(f"{x.stem}_2")] for x in fpaths], []
    )


@click.command()
@click.option("--fpath", default="/data3/public/Gigaspeech/processed/audio_pairs")
@click.option("--train_portion", default=0.9)
@click.option("--test_count", default=None)
def main(fpath, train_portion, test_count):
    fpath = Path(fpath)
    files = fpath.rglob("*.opus")
    files = [x.relative_to(fpath) for x in files]
    files = [
        x.with_stem(re.sub("_\d$", "", x.stem)) for x in files if x.stem[-1] == "1"
    ]
    shuffle(files)
    if train_portion:
        train_count = int(len(files) * train_portion)
        train_files = files[:train_count]
        test_files = files[train_count:]
    elif test_count:
        train_files = files[test_count:]
        test_files = files[:test_count]
    else:
        raise ValueError("Either train_portion or test_count must be specified")
    train_files = duplicate_pairs(train_files)
    test_files = duplicate_pairs(test_files)
    train_files = sorted(train_files)
    test_files = sorted(test_files)

    with (fpath.parent / "train_files.txt").open("w") as f:
        for file in train_files:
            f.write(f"{file}\n")

    with (fpath.parent / "test_files.txt").open("w") as f:
        for file in test_files:
            f.write(f"{file}\n")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
