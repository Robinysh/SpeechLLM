# pylint: disable=possibly-used-before-assignment
import torch
from transformers import LlamaForCausalLM

from speechllm.utils import check_hpu, supports_bf16

if check_hpu():
    import habana_frameworks.torch.core as htcore  # noqa: F401 pylint: disable=unused-import


# pylint: disable-next=too-many-arguments, too-many-locals
def constructor(
    offline=False,
    model_fpath=None,
    restore_ckpt=None,
):
    model = LlamaForCausalLM.from_pretrained(
        offline and model_fpath or "fnlp/AnyGPT-chat",
        torch_dtype=(torch.bfloat16 if supports_bf16() else torch.float16),
        # quantization_config=bnb_config,
        attn_implementation="flash_attention_2" if not check_hpu() else None,
        device_map="cpu",
        use_cache=False,
        local_files_only=offline,
    )
    if restore_ckpt is not None:
        ckpt = torch.load(restore_ckpt)
        ckpt = {k.removeprefix("anygpt."): v for k, v in ckpt["state_dict"].items()}
        model.load_state_dict(ckpt)

    for param in model.parameters():
        param.requires_grad = False

    def forward(model_teacher_input):
        model.eval()
        with torch.no_grad():
            lm_output = model(**model_teacher_input, output_hidden_states=True)
        return {
            "teacher_output": lm_output,
        }

    modules = {
        "teacher_anygpt": model,
    }

    methods = {
        "forward": forward,
    }

    return {"modules": modules, "methods": methods}
