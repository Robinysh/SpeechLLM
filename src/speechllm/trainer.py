import logging
import traceback
from functools import partial
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import numpy as np
import torch
from lightningtools import reporter
from lightningtools.trainer import BaseLightningModule
from lightningtools.utils import NoamLR, detach_any
from torch_warmup_lr import WarmupLR

import speechllm.logger  # noqa: F401 pylint: disable=unused-import
from speechllm.utils import check_hpu, recursive_map

logging.getLogger("habana_frameworks").setLevel(logging.WARNING)


# logger.remove()
# logger.add(sys.stderr, level="INFO")

torch.set_float32_matmul_precision("medium")


# pylint: disable-next=too-many-ancestors
class Model(BaseLightningModule):
    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.optimizer_dict = None
        self.scheduler_dict = None
        self.optimizer_state_checkpoints = None
        self.p2name_dict = None
        self.name2p_dict = None

    def on_train_start(self):
        if self.config.config_optimizers.optimizer == "galore":
            logging.info(
                "Activated GaLoRE fine-tuning, depending on your model size and hardware, the training might take a while before starting. Please be patient!"
            )
            self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.increment_completed()

    # pylint: disable-next=arguments-differ
    def forward(self, ret):
        token_ids = self.pipelines["inference"](**ret)["tokens"]
        response = self.tokenizer.batch_decode(
            token_ids.cpu(), skip_special_tokens=True
        )
        return response

    def training_step(self, batch, batch_idx):
        optimizer_idx = batch_idx % len(self.optimizer_idx_map)
        stage_name = self.optimizer_idx_map[int(optimizer_idx)]
        if stage_name not in self.pipelines:
            return None
        try:
            model_output = self.pipelines[stage_name](
                **batch, optimizer_idx=optimizer_idx, step=self.global_step
            )
        except RuntimeError as e:
            if "Graph compile failed." in str(e):
                torch.save(batch, f"err_{batch_idx}.pt")
                print(e)
                return None
            raise e

        if model_output is None:
            return None
        loss_dict = self.losses[stage_name](
            **(batch | model_output), step=self.global_step
        )
        if len(loss_dict) == 0:
            return None
        total_loss = sum(map(torch.mean, loss_dict.values()))

        self.manual_backward(total_loss)

        if len(self.optimizer_idx_map) == 1:
            opt = self.optimizers()
        else:
            opt = self.optimizers()[optimizer_idx]

        if isinstance(opt, torch.optim.Optimizer):
            self.clip_gradients(
                opt,
                gradient_clip_val=self.gradient_clip_val,
                gradient_clip_algorithm="norm",
            )
            opt.step()
            opt.zero_grad()

        loss_dict["loss"] = total_loss
        if not total_loss.requires_grad:
            total_loss = None

        reporter.report_dict(
            {f"train_{stage_name}/" + k: torch.mean(v) for k, v in loss_dict.items()}
        )
        loss_dict = {
            k: detach_any(v) if k != "loss" else v for k, v in loss_dict.items()
        }

        model_output = {k: detach_any(v) for k, v in model_output.items()}
        return {
            "loss_dict": loss_dict,
            "model_output": model_output,
            "loss": total_loss,
        }

    def validation_step(self, batch, batch_idx):
        try:
            model_output = self.pipelines[self.optimizer_idx_map[0]](
                **batch, step=self.global_step
            )
        except RuntimeError as e:
            if "Graph compile failed." in str(e):
                torch.save(batch, f"err_{batch_idx}.pt")
                print(e)
                return None
            raise e

        if model_output is None:
            return None
        loss_dict = self.losses["val"](**(batch | model_output), step=self.global_step)
        if len(loss_dict) == 0:
            return None
        total_loss = sum(map(torch.mean, loss_dict.values()))
        loss_dict["loss"] = total_loss
        reporter.report_dict(
            {"valid/" + k: torch.mean(v) for k, v in loss_dict.items()}
        )

        if hasattr(self, "log_eval") and batch_idx == 0 and self.trainer.is_global_zero:
            first_data = {
                k: v[:1] if isinstance(v, (torch.Tensor, list)) else v
                for k, v in batch.items()
            }
            first_data["model_input"] = {
                k: v[:1] for k, v in batch["model_input"].items()
            }
            first_data["model_infer_input"] = {
                k: v[:1] for k, v in batch["model_infer_input"].items()
            }
            try:
                reporter.logging_disabled = True
                model_inference_output = self.forward(first_data)
                reporter.logging_disabled = False
                if model_inference_output is not None:
                    self.log_eval(batch, model_output, model_inference_output)
            # pylint: disable-next=broad-exception-caught
            except Exception as e:
                traceback.print_exc()
                logging.error(e)

        return {
            "loss_dict": loss_dict,
            "model_output": model_output,
            "loss": total_loss,
        }

    # pylint: disable-next=unused-argument
    def log_eval(self, batch, model_output, model_inference_output):
        if self.config.paths.pretrained_models is not None:
            decoder_fpath = (
                Path(self.config.paths.pretrained_models) / "AnyGPT-speech-modules"
            )
        else:
            decoder_fpath = None
        reporter.report(
            "text/tokens_prediction",
            model_inference_output,
            tag="text",
        )

        reporter.report(
            "audio/sample_prediction",
            model_inference_output,
            decoder_fpath=decoder_fpath,
            device=self.device,
            tag="speechtokens",
        )

    def on_train_batch_start(self, *args, **kwargs):
        # on_batch_start does not work
        self.log_batch(*args, **kwargs)
        reporter.report("trainer/global_step", self.global_step)
        shm = SharedMemory(create=False, size=4, name="global_step")
        arr = np.ndarray([1], np.int32, shm.buf)
        arr[0] = self.global_step

    # pylint: disable-next=unused-argument
    def on_train_batch_end(self, *args, **kwargs):
        lr_scheduler = self.lr_schedulers()
        if lr_scheduler is not None:
            lr_scheduler.step()
        if self.config.config_optimizers.optimizer == "galore":
            self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.increment_completed()

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

    def configure_optimizers(
        self,
    ):
        config_opt = self.config.config_optimizers
        if config_opt.optimizer == "galore":
            return self.configure_galore(config_opt)

        if config_opt.optimizer.lower() == "adam":
            if check_hpu():
                # pylint: disable-next=import-outside-toplevel
                from habana_frameworks.torch.hpex.optimizers import FusedAdamW

                optimizer = FusedAdamW(
                    self.param_group["default"],
                    config_opt.learning_rate,
                )

                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, config_opt.lr_restart_period, 2
                )
                if config_opt.warmup_step > 0:
                    scheduler = WarmupLR(
                        scheduler,
                        config_opt.init_lr,
                        config_opt.warmup_step,
                        warmup_strategy="linear",
                    )
                lr_dict = {
                    "scheduler": scheduler,
                    "interval": "step",
                    "strict": False,
                    "name": "lr_dict_t",
                }
                return [
                    {
                        "optimizer": optimizer,
                        "lr_scheduler": lr_dict,
                    },
                ]

            # pylint: disable-next=import-outside-toplevel
            import bitsandbytes as bnb

            # NOTE: optimizer frequency messes up feature loss
            optimizer = bnb.optim.Adam8bit(
                self.param_group["default"],
                config_opt.learning_rate,
            )
            scheduler = NoamLR(optimizer, config_opt.warmup_step)
            scheduler = WarmupLR(
                scheduler,
                config_opt.init_lr,
                config_opt.warmup_step,
                warmup_strategy="linear",
            )

            lr_dict = {
                "scheduler": scheduler,
                "interval": "step",
                "monitor": "valid/loss",
                "strict": False,
                "name": "lr_dict_t",
            }
            return [
                {
                    "optimizer": optimizer,
                    "lr_scheduler": lr_dict,
                },
            ]
        raise ValueError(f"Unknown optimizer {config_opt.optimizer}")

    def configure_galore(  # noqa: C901 pylint: disable=too-many-locals
        self, config_opt
    ):
        # pylint: disable-next=import-outside-toplevel
        from galore_torch import GaLoreAdamW8bit

        galore_params = []
        target_modules_list = ["attn", "mlp"]
        for module_name, module in self.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue

            galore_params.append(module.weight)
        id_galore_params = [id(p) for p in galore_params]

        optimizer_dict = {}
        scheduler_dict = {}
        name2p_dict = {}
        p2name_dict = {}

        # define a hook function to update the parameter p during the backward pass
        def optimizer_hook(p, log_lr=False):
            if p.grad is None:
                return
            optimizer_dict[p].step()
            optimizer_dict[p].zero_grad()
            scheduler_dict[p].step()
            if log_lr:
                reporter.report("lr", optimizer_dict[p].param_groups[0]["lr"])

        lr_logged = False
        for name, p in self.named_parameters():
            if p.requires_grad:
                if id(p) in id_galore_params:
                    optimizer_dict[p] = GaLoreAdamW8bit(
                        [
                            {
                                "params": p,
                                "rank": 128,
                                "update_proj_gap": 250,
                                "scale": 0.25,
                                "proj_type": "std",
                            }
                        ],
                        lr=config_opt.learning_rate,
                    )
                else:
                    # pylint: disable-next=import-outside-toplevel
                    import bitsandbytes as bnb

                    optimizer_dict[p] = bnb.optim.Adam8bit(
                        [p], lr=config_opt.learning_rate
                    )
                scheduler_dict[p] = (
                    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer_dict[p], 10000, 2
                    )
                )

                # Register the hook onto every parameter
                p.register_post_accumulate_grad_hook(
                    partial(optimizer_hook, log_lr=not lr_logged)
                )
                lr_logged = True
                name2p_dict[name] = p
                p2name_dict[p] = name

        if self.optimizer_state_checkpoints is not None:
            for name in self.optimizer_state_checkpoints["optimizer"].keys():
                p = name2p_dict[name]
                optimizer_dict[p].load_state_dict(
                    self.optimizer_state_checkpoints["optimizer"][name]
                )
                # hack around for galore saving tensors inside a non-module object
                for v in optimizer_dict[p].state.values():
                    if "projector" in v and hasattr(v["projector"], "ortho_matrix"):
                        v["projector"].ortho_matrix = v["projector"].ortho_matrix.to(
                            p.device
                        )
                scheduler_dict[p].load_state_dict(
                    self.optimizer_state_checkpoints["scheduler"][name]
                )

        self.optimizer_dict = optimizer_dict
        self.scheduler_dict = scheduler_dict
        self.name2p_dict = name2p_dict
        self.p2name_dict = p2name_dict

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
        for key, value in checkpoint["callbacks"].items():
            if "ModelCheckpoint" in key:
                checkpoint["callbacks"][key] = recursive_map(
                    value, lambda x: x.to(device=self.device)
                )
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
        if "optimizer_state" in checkpoint:
            self.optimizer_state_checkpoints = {
                "optimizer": checkpoint["optimizer_state"],
                "scheduler": checkpoint["scheduler_state"],
            }

    def on_save_checkpoint(self, checkpoint):
        if self.optimizer_dict is not None:
            checkpoint["optimizer_state"] = {
                self.p2name_dict[k]: v.state_dict()
                for k, v in self.optimizer_dict.items()
            }
            checkpoint["scheduler_state"] = {
                self.p2name_dict[k]: v.state_dict()
                for k, v in self.scheduler_dict.items()
            }
        return checkpoint
