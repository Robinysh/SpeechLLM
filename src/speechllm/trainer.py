import bitsandbytes as bnb
import torch
from lightningtools import reporter
from lightningtools.trainer import BaseLightningModule
from lightningtools.utils import NoamLR

import speechllm.logger  # noqa pylint: disable=unused-import
from speechllm.data.utils import get_tokenizer

# logger.remove()
# logger.add(sys.stderr, level="INFO")

torch.set_float32_matmul_precision("medium")


# pylint: disable-next=too-many-ancestors
class Model(BaseLightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = get_tokenizer()

    # pylint: disable-next=arguments-differ
    def forward(self, ret):
        token_ids = self.pipelines["inference"](**ret)["tokens"]
        response = self.tokenizer.batch_decode(
            token_ids.cpu(), skip_special_tokens=True
        )
        return response

    # pylint: disable-next=unused-argument
    def log_eval(self, batch, model_output, model_inference_output):
        reporter.report(
            "audio/sample_prediction",
            model_inference_output,
            tag="speechtokens",
        )

    def on_train_batch_start(self, *args, **kwargs):
        # on_batch_start does not work
        self.log_batch(*args, **kwargs)

    # pylint: disable-next=unused-argument
    def on_train_batch_end(self, *args, **kwargs):
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()

    def on_validation_batch_start(self, *args, **kwargs):
        # on_batch_start does not work
        self.log_batch(*args, **kwargs)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch["model_input"] = batch["model_input"].to(device)
        batch["model_infer_input"] = batch["model_infer_input"].to(device)
        return batch

    # pylint: disable-next=unused-argument
    def log_batch(self, batch, *args, **kwargs):
        reporter.report(
            "audio/input",
            batch["input_audio"],
            tag="audio",
        )

        reporter.report(
            "audio/output_label",
            batch["output_audio"],
            tag="audio",
        )

        reporter.report(
            "text/input_transcript",
            batch["input_transcript"],
            tag="text",
        )

        reporter.report(
            "text/output_transcript",
            batch["output_transcript"],
            tag="text",
        )

        reporter.report(
            "text/infer_prompt",
            batch["infer_prompt"],
            tag="text",
        )

        reporter.report(
            "text/prompt",
            batch["prompt"],
            tag="text",
        )

    def detach_values(self, model_output):
        result = {}
        for key, val in model_output.items():
            if val is torch.Tensor:
                result[key] = val.detach()
            elif val is dict:
                detached = self.detach_values(val)
                result[key] = detached
            else:
                result[key] = val

        return result

    def configure_optimizers(self):
        # NOTE: optimizer frequency messes up feature loss
        config_opt = self.config.config_optimizers

        optim_t = bnb.optim.Adam8bit(
            # optim_t = bnb.optim.PagedAdamW(
            self.param_group["default"],
            config_opt.default.learning_rate,
        )

        scheduler_t = NoamLR(optim_t, config_opt.default.warmup_step)
        lr_dict_t = {
            "scheduler": scheduler_t,
            "interval": "step",
            "monitor": "valid/loss",
            "strict": False,
            "name": "lr_dict_t",
        }

        return [
            {
                "optimizer": optim_t,
                "lr_scheduler": lr_dict_t,
            },
        ]

    # pylint: disable-next=arguments-differ
    def configure_gradient_clipping(
        self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm
    ):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            sum([p["params"] for p in optimizer.param_groups], []), gradient_clip_val
        )
        name = self.config.optimizer_order[optimizer_idx]
        reporter.report(f"{name}_grad_norm", grad_norm)

    def on_load_checkpoint(self, checkpoint):
        new_state_dict = {}
        filters = []
        clone_list = []
        for k, _ in checkpoint["state_dict"].items():
            if any(f in k for f in filters):
                continue
            new_state_dict[k] = checkpoint["state_dict"][k]

            for replace_k in clone_list:
                if f"{replace_k}." in k:
                    new_state_dict[
                        k.replace(f"{replace_k}.", f"{replace_k}_clone.")
                    ] = checkpoint["state_dict"][k]
        checkpoint["state_dict"] = new_state_dict
