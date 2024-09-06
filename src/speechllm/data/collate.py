import os
from multiprocessing.shared_memory import SharedMemory

import numpy as np
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


def find_all_index(lst, value):
    return [i for i, x in enumerate(lst) if x == value]


def drop_tokens(item, index, replace_id):
    shm = SharedMemory(
        create=False, size=4, name=f"global_step_{os.environ['MASTER_PORT']}"
    )
    arr = np.ndarray([1], np.int32, shm.buf)
    global_step = arr[0]
    total_num = index[1] - index[0]
    # drop_percentile = max(min(global_step / 100000, 1), 0)
    # drop_percentile = max(min(global_step / 100000, 1), 0)
    # drop_percentile = max(min((global_step - 50000 + 30000) / 50000, 1), 0)
    # drop_percentile = max(min((66000 - 50000 + 30000) / 50000, 1), 0)
    # drop_percentile = max(min((70000 - 50000 + 30000) / 50000, 1), 0)
    # drop_percentile = max(min(10000 / 25000 + (global_step + 50000) / 200000, 1), 0)
    # drop_percentile = max(min(10000 / 25000, 1), 0)
    min_drop_num = global_step // 2500
    drop_num = min(min_drop_num + int(np.random.exponential(scale=0.25)), total_num)
    # drop_num = int(len(tokens))
    # return " ".join(tokens[drop_num:])
    if drop_num != 0:
        item[index[0] : index[0] + drop_num] = (
            replace_id  # replace tokens with unused token
        )
        # tokens = tokens[:-drop_num]
        # tokens = tokens[drop_num:]
    return item


def distill_collate(row, tokenizer):
    row = list_dict_to_dict_list(row)
    row["model_input"] = tokenize_func(row["prompt"], tokenizer)
    row["model_teacher_input"] = tokenize_func(row["teacher_prompt"], tokenizer)
    row["model_infer_input"] = tokenize_func(row["infer_prompt"], tokenizer)
    row["model_teacher_infer_input"] = tokenize_func(
        row["teacher_infer_prompt"], tokenizer
    )

    anygpt_id = tokenizer.convert_tokens_to_ids("[AnyGPT]")
    sosp_id = tokenizer.convert_tokens_to_ids("<sosp>")
    blank_id = tokenizer.convert_tokens_to_ids("—")
    row["answer_start_position"] = []
    for i, item in enumerate(row["model_input"].input_ids):
        sosp_index = listrfind(item, sosp_id)
        row["answer_start_position"].append(sosp_index)
        anygpt_index = find_all_index(item, anygpt_id)
        # magic numbers from manual prompt tokens counting
        asr_index = (anygpt_index[1] + 5, anygpt_index[2] - 2)
        tts_index = (anygpt_index[2] + 2, sosp_index - 2)

        # row['model_input'].input_ids[i, asr_index[-1][0]:asr_index[-1][1]] = blank_id
        # row['model_input'].input_ids[i, tts_index[-1][0]:tts_index[-1][0] + 2] = blank_id
        row["model_input"].input_ids[i] = drop_tokens(
            row["model_input"].input_ids[i], asr_index, blank_id
        )
        # pylint: disable=using-constant-test
        if False:
            row["model_input"].input_ids[i] = drop_tokens(
                row["model_input"].input_ids[i], tts_index, blank_id
            )
    row["model_input"]["labels"] = row["model_input"].input_ids.clone()
    row["model_input"]["labels"][row["model_input"].attention_mask == 0] = -100

    row["teacher_answer_start_position"] = []
    for item in row["model_teacher_input"].input_ids:
        row["teacher_answer_start_position"].append(listrfind(item, sosp_id))

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
