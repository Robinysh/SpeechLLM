# pylint: disable=possibly-used-before-assignment
import logging

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, GenerationConfig, LlamaForCausalLM

from speechllm.utils import check_hpu, supports_bf16

if not check_hpu() and torch.cuda.is_available():
    import bitsandbytes as bnb
    from unsloth import FastLanguageModel
if check_hpu():
    import habana_frameworks.torch.core as htcore  # noqa: F401 pylint: disable=unused-import


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


# pylint: disable-next=too-many-arguments, too-many-locals
def constructor(
    lora_alpha=64,
    lora_rank=32,
    unsloth=True,
    gradient_checkpointing=False,
    lora=False,
    offline=False,
    model_fpath=None,
    use_gaudi_impl=False,
):
    if unsloth and (not torch.cuda.is_available() or check_hpu()):
        logging.warning("Unsloth will be disabled because no GPUs are detected.")
        unsloth = False
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
                    torch.bfloat16 if supports_bf16() else torch.float16
                ),
            )
            model = LlamaForCausalLM.from_pretrained(
                offline and model_fpath or "fnlp/AnyGPT-chat",
                torch_dtype=(torch.bfloat16 if supports_bf16() else torch.float16),
                quantization_config=bnb_config,
                attn_implementation="flash_attention_2" if not check_hpu() else None,
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
                dtype=(torch.bfloat16 if supports_bf16() else torch.float16),
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
            if not use_gaudi_impl:
                model = LlamaForCausalLM.from_pretrained(
                    offline and model_fpath or "fnlp/AnyGPT-chat",
                    torch_dtype=(torch.bfloat16 if supports_bf16() else torch.float16),
                    attn_implementation=(
                        "flash_attention_2" if not check_hpu() else None
                    ),
                    device_map="cpu",
                    use_cache=False,
                    local_files_only=offline,
                )
            else:
                # pylint: disable-next=import-outside-toplevel
                from speechllm.models.impl.gaudillama import GaudiLlamaForCausalLM

                model = GaudiLlamaForCausalLM.from_pretrained(
                    offline and model_fpath or "fnlp/AnyGPT-chat",
                    torch_dtype=(torch.bfloat16 if supports_bf16() else torch.float16),
                    attn_implementation=(
                        "flash_attention_2" if not check_hpu() else None
                    ),
                    device_map="cpu",
                    use_cache=False,
                    local_files_only=offline,
                )
                model.generation_config.use_fused_rope = True
                model.model.config.use_fused_rope = True
                model.model.config.use_fused_rms_norm = True

    def forward(model_input):
        model.train()
        lm_output = model(**model_input)
        return {
            "lm_output": lm_output,
        }

    def inference(model_infer_input):
        model.eval()
        generation_config = GenerationConfig(temperature=0.7, top_p=0.8, do_sample=True)
        with torch.inference_mode():
            tokens = model.generate(
                **model_infer_input,
                # eos_token_id=tokenizer.encode("<eosp>"),
                max_new_tokens=2048,
                # max_new_tokens=1024,
                max_time=30,
                generation_config=generation_config,
            )
            # -1 for keeping <sosp>
            tokens = tokens[:, model_infer_input["input_ids"].shape[1] - 1 :]

            return {
                "tokens": tokens,
            }

    # TODO: disabled until crash issue is resolved. hpu_backend only works in eager mode and is much slower
    # # pylint: disable-next=protected-access
    # if check_hpu() and 'hpu_backend' in torch._dynamo.list_backends():
    #     model = torch.compile(model, backend="hpu_backend")

    # TODO: only works in lazy mode
    # import habana_frameworks.torch as htorch
    # htcore.hpu.ModuleCacher()(model=model, inplace=True)

    modules = {
        "anygpt": model,
    }

    methods = {
        "forward": forward,
        "inference": inference,
    }

    return {"modules": modules, "methods": methods}
