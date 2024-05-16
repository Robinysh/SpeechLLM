from pathlib import Path

import lightning as L
from datasets import load_dataset
from torch.utils.data import DataLoader

from speechllm.data import processes
from speechllm.data.prompter import Prompter


class DataModule(L.LightningDataModule):
    def __init__(
        self, train_dataloader_args, val_dataloader_args, train_dataset, val_dataset
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_dataloader_args = train_dataloader_args
        self.val_dataloader_args = val_dataloader_args

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.train_dataloader_args)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.val_dataloader_args)


def hf_pipeline(
    database,
    num_workers=8,
    cache_dir=None,
    use_cache=True,
):
    prompter = Prompter()
    cache_dir = Path(cache_dir)

    ds = load_dataset("text", data_files=database["metadata"])["train"]

    ds = ds.map(
        lambda x: processes.read_audio_tokens(x, Path(database["metadata"])),
        batched=False,
        num_proc=num_workers,
        load_from_cache_file=False,
    )

    Path(cache_dir / "template").mkdir(parents=True, exist_ok=True)
    ds = ds.map(
        lambda x: {
            "prompt": prompter.generate_template(x["input_tokens"], x["output_tokens"])
        },
        batched=False,
        num_proc=num_workers,
        load_from_cache_file=use_cache,
        cache_file_name=str(cache_dir / "template" / "train.arrow"),
    )

    # Path(cache_dir / "tokenized").mkdir(parents=True, exist_ok=True)
    # ds = ds.map(
    #     lambda x: tokenize_func(x['prompt'], tokenizer),
    #     batched=True,
    #     num_proc=num_workers,
    #     load_from_cache_file=False,
    #     desc="Running tokenizer on dataset",
    #     cache_file_name=str(cache_dir / "tokenized" / "train.arrow"),
    # )

    # ic(ds[0])
    # ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    # ic(ds[0])

    return ds


if __name__ == "__main__":
    print(
        hf_pipeline(
            {
                "fpaths": "/data3/public/GigaSpeech/processed/train_files.txt",
            },
            num_workers=8,
            cache_dir="/data3/robinysh/cache",
        )[0]
    )
