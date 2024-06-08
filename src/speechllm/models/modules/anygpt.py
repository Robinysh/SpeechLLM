import bitsandbytes as bnb
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, LlamaForCausalLM
from unsloth import FastLanguageModel

from speechllm.utils import check_ampere_gpu


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


# pylint: disable-next=too-many-arguments
def constructor(
    lora_alpha=64,
    lora_rank=32,
    unsloth=True,
    gradient_checkpointing=True,
    lora=False,
    offline=False,
    model_fpath=None,
):
    if lora:
        if unsloth:
            model, _ = FastLanguageModel.from_pretrained(
                model_name=offline and model_fpath or "fnlp/AnyGPT-chat",
                max_seq_length=2048,
                load_in_4bit=True,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=offline,
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
                use_gradient_checkpointing=(
                    "unsloth" if gradient_checkpointing else False
                ),
            )
            FastLanguageModel.for_training(
                model, use_gradient_checkpointing=gradient_checkpointing
            )
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=(
                    torch.bfloat16 if check_ampere_gpu() else torch.float16
                ),
            )
            model = LlamaForCausalLM.from_pretrained(
                offline and model_fpath or "fnlp/AnyGPT-chat",
                torch_dtype=(torch.bfloat16 if check_ampere_gpu() else torch.float16),
                quantization_config=bnb_config,
                attn_implementation="flash_attention_2",
                device_map="auto",
                local_files_only=offline,
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
            if not gradient_checkpointing:
                model.gradient_checkpointing_disable()
    else:
        if unsloth:
            model, _ = FastLanguageModel.from_pretrained(
                model_name=offline and model_fpath or "fnlp/AnyGPT-chat",
                max_seq_length=2048,
                dtype=(torch.bfloat16 if check_ampere_gpu() else torch.float16),
                load_in_4bit=False,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=offline,
            )
            FastLanguageModel.for_training(
                model,
                use_gradient_checkpointing=False,  # Incompatible with per layer weight update
            )
        else:
            model = LlamaForCausalLM.from_pretrained(
                offline and model_fpath or "fnlp/AnyGPT-chat",
                torch_dtype=(torch.bfloat16 if check_ampere_gpu() else torch.float16),
                attn_implementation="flash_attention_2",
                device_map="auto",
                use_cache=False,
                local_files_only=offline,
            )

    def forward(model_input):
        lm_output = model(**model_input)

        # reporter.report(
        #     "audio/next_token_generation",
        #     torch.argmax(lm_output.logits, dim=-1),
        #     tag="speechtokens",
        # )

        return {
            "lm_output": lm_output,
        }

    def inference(model_infer_input):
        tokens = model.generate(
            **model_infer_input,
            # eos_token_id=tokenizer.encode("<eosp>"),
            max_length=2048,
            do_sample=True,
        )
        # -1 for keeping <sosp>
        tokens = tokens[:, model_infer_input["input_ids"].shape[1] - 1 :]

        return {
            "tokens": tokens,
        }

    modules = {
        "anygpt": model,
    }

    methods = {
        "forward": forward,
        "inference": inference,
    }

    return {"modules": modules, "methods": methods}
