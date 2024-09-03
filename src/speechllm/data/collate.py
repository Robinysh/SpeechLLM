import torch

from speechllm.data.utils import tokenize_func
from speechllm.utils import list_dict_to_dict_list, listrfind


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


def collate(row, tokenizer):
    row = list_dict_to_dict_list(row)
    row["model_input"] = tokenize_func(row["prompt"], tokenizer)
    row["model_infer_input"] = tokenize_func(row["infer_prompt"], tokenizer)
    return row


def distill_collate(row, tokenizer):
    row = list_dict_to_dict_list(row)
    row["model_input"] = tokenize_func(row["prompt"], tokenizer)
    row["model_teacher_input"] = tokenize_func(row["teacher_prompt"], tokenizer)
    row["model_infer_input"] = tokenize_func(row["infer_prompt"], tokenizer)
    row["model_teacher_infer_input"] = tokenize_func(
        row["teacher_infer_prompt"], tokenizer
    )

    row["answer_start_position"] = []
    for item in row["model_input"].input_ids:
        row["answer_start_position"].append(
            listrfind(item, tokenizer.convert_tokens_to_ids("<sosp>"))
        )

    row["teacher_answer_start_position"] = []
    for item in row["model_teacher_input"].input_ids:
        row["teacher_answer_start_position"].append(
            listrfind(item, tokenizer.convert_tokens_to_ids("<sosp>"))
        )

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
