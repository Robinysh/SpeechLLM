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
        lambda x: processes.load_audio(Path(database["audio"]) / x["fname"]),
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


if __name__ == "__main__":
    print(
        hf_pipeline(
            {
                "metadata": "/data3/public/GigaSpeech/processed/train_files.txt",
                "audio": "/data3/public/GigaSpeech/processed/audio_pairs",
                "transcript": "/data3/public/GigaSpeech/processed/dialogue_pairs",
                "tokens": "/data3/public/GigaSpeech/processed/tokens",
            },
            num_workers=16,
            cache_dir="/data3/robinysh/cache",
        )[0]
    )
