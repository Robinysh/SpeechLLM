import torch

from speechllm.data.utils import tokenize_func


def reflective_pad_sequence(sequences, padding_value=0):
    sequences = [torch.Tensor(seq) for seq in sequences]
    seq_lens = [len(seq) for seq in sequences]
    max_len = max(seq_lens)
    output = (
        torch.ones(len(sequences), max_len, *sequences[0].shape[1:]) * padding_value
    ).type_as(sequences[0])
    for i, seq in enumerate(sequences):
        output[i, : len(seq), ...] = seq
        if torch.is_tensor(seq):
            output[i, len(seq) : 2 * len(seq) - 1, ...] = seq.flip(0)[
                1 : max_len - len(seq) + 1
            ]
        else:
            output[i, len(seq) : 2 * len(seq) - 1, ...] = seq[
                -2 : len(seq) - max_len - 1 : -1
            ]
    return output


def to_list_of_tensor(x):
    return [torch.Tensor(i) for i in x]


def list_dict_to_dict_list(x):
    return {k: [dic[k] for dic in x] for k in x[0]}


def collate(row, tokenizer):
    """
    audio_lens = [len(row) for row in rows["audio"]]
    audio = reflective_pad_sequence(rows["audio"])

    hubert_lens = [len(row) for row in rows["hubert_embs"]]
    hubert_embs = reflective_pad_sequence(rows["hubert_embs"])

    token_type_ids = pad_sequence(
        to_list_of_tensor(rows["token_type_ids"]), padding_value=0, batch_first=True
    )

    input_ids = pad_sequence(
        to_list_of_tensor(rows["input_ids"]), padding_value=0, batch_first=True
    ).long()
    attention_mask = pad_sequence(
        to_list_of_tensor(rows["attention_mask"]), padding_value=0, batch_first=True
    )

    audio_info = rows["audio_info"]
    audio_info = {k: [dic[k][0] for dic in audio_info] for k in audio_info[0]}
    audio_info["input_audios"] = pad_sequence(
        audio_info["input_audios"], batch_first=True
    )
    audio_info["input_audio_lengths"] = torch.stack(audio_info["input_audio_lengths"])

    return {
        "audio": audio,
        "audio_lens": audio_lens,
        "hubert_lens": hubert_lens,
        "hubert_embs": hubert_embs,
        "fpath": rows["fpath"],
        "duration": torch.Tensor(rows["duration"]),
        "text": rows["text"],
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "audio_info": audio_info,
        "raw_text": rows["raw_text"],
        "raw_tokens": rows["raw_tokens"],
    }
    """
    row = list_dict_to_dict_list(row)
    row["model_input"] = tokenize_func(row["prompt"], tokenizer)
    row["model_infer_input"] = tokenize_func(row["infer_prompt"], tokenizer)
    return row


if __name__ == "__main__":
    from transformers import LlamaTokenizer

    tokenizer_ = LlamaTokenizer.from_pretrained(
        "fnlp/AnyGPT-chat",
        padding_side="right",
        use_fast=False,
    )

    row_ = [
        {
            "prompt": "<sosp><🗣️149><🗣️149><🗣️285><🗣️285><🗣️285><🗣️558><🗣️558><🗣️994><🗣️484><🗣️735><🗣️317><🗣️317><🗣️896><🗣️896><🗣️1001><🗣️680><🗣️918><🗣️918><🗣️976><🗣️976><🗣️976><🗣️399><🗣️746><🗣️972><🗣️46><🗣️46><🗣️469><eosp>"
        }
    ]

    output_ = collate(row_, tokenizer_)
    print(len(row_[0]["prompt"].split("><")))
    print(output_["model_input"]["input_ids"].shape)
