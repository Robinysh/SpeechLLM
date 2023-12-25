from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer

PRETRAINED_MODEL_DIR = "xun/Qwen-Audio-Chat-Int4"

tokenizer = AutoTokenizer.from_pretrained(
    PRETRAINED_MODEL_DIR, use_fast=True, trust_remote_code=True
)

model = AutoGPTQForCausalLM.from_pretrained(
    PRETRAINED_MODEL_DIR,
    trust_remote_code=True,
    max_memory={0: "20GIB"},
    quantize_config={
        "bits": 4,
        "group_size": 128,
        "damp_percent": 0.01,
        "desc_act": False,
        "static_groups": False,
        "sym": True,
        "true_sequential": True,
        "model_file_base_name": "model",
        "quant_method": "gptq",
        "use_exllama": True,
        "disable_exllama": False,
        "exllama_config": {"version": 2},
    },
)

print(
    tokenizer.decode(
        model.generate(
            **tokenizer("The meaning of life is", return_tensors="pt").to(model.device),
            eos_token_id=tokenizer.encode("<|im_end|>"),
        )[0]
    )
)
