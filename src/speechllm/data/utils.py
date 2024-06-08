from functools import cache

from transformers import LlamaTokenizer


def tokenize_func(sentence, tokenizer):
    result = tokenizer(
        sentence,
        truncation=True,
        max_length=tokenizer.model_max_length,
        padding=True,
        return_tensors="pt",
    )
    result["labels"] = result["input_ids"].clone()
    return result


@cache
def get_tokenizer(offline=False, model_fpath=None):
    tokenizer = LlamaTokenizer.from_pretrained(
        offline and model_fpath or "fnlp/AnyGPT-chat",
        # model_max_length=training_args.model_max_length,
        padding_side="left",
        local_files_only=offline,
        use_fast=False,
    )
    # pylint: disable=pointless-string-statement
    """
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference
    for token in [user_name, chatbot_name, user_end, chatbot_end]:
        if token not in tokenizer.get_vocab():
            logger.info(f"Add special unit tokens {token} to tokenizer.vocab")
            tokenizer.add_tokens([token])

    for modality in modal_special_str.keys():
        prefix = modal_special_str[modality]["prefix"]
        start = modal_special_str[modality]["sos"]
        end = modal_special_str[modality]["eos"]
        modality_vocab_size = modal_special_str[modality]["vocab_size"]
        if start not in tokenizer.get_vocab():
            logger.info(
                f"Add {modality} tokens <{prefix}0>-<{prefix}{modality_vocab_size-1}> to tokenizer.vocab"
            )
            tokens = [f"<{prefix}{x}>" for x in range(modality_vocab_size)] + [
                start,
                end,
            ]
            tokenizer.add_tokens(tokens)
    """
    return tokenizer
