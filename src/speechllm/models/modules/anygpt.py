import bitsandbytes as bnb
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, LlamaForCausalLM
from unsloth import FastLanguageModel


def find_all_linear_names(model):
    # cls = bnb.nn.Linear8bitLt
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def constructor(lora_alpha=32, lora_rank=32, unsloth=False):
    if unsloth:
        model, _ = FastLanguageModel.from_pretrained(
            model_name="fnlp/AnyGPT-chat",
            max_seq_length=2048,
            load_in_4bit=True,
            device_map="auto",
            trust_remote_code=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",  # attention (self_attn)
                "gate_proj",
                "down_proj",
                "up_proj",  # FFN (mlp)
            ],
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing=False,
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = LlamaForCausalLM.from_pretrained(
            "fnlp/AnyGPT-chat",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

        model = prepare_model_for_kbit_training(model)
        modules = find_all_linear_names(model)

        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=0,
            r=lora_rank,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

        model = get_peft_model(model, peft_config)
    model.gradient_checkpointing_disable()

    def forward(model_input):
        lm_output = model(**model_input)
        return {
            "lm_output": lm_output,
        }

    # def inference(audio):
    #     tokenizer.decode(
    #         model.generate(
    #             **tokenizer("The meaning of life is", return_tensors="pt").to(model.device),
    #             eos_token_id=tokenizer.encode("<|im_end|>"),
    #         )[0]
    #     )

    #     return {
    #         "logit": logit,
    #     }

    modules = {
        "anygpt": model,
    }

    methods = {
        "forward": forward,
    }

    return {"modules": modules, "methods": methods}
