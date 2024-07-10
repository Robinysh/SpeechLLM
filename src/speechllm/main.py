# pylint: disable=wrong-import-position, wrong-import-order
from pathlib import Path

from speechllm.utils import check_hpu

if check_hpu():
    # workaround for broken frozen ddp training
    import habana_frameworks.torch.gpu_migration  # noqa: F401 pylint: disable=unused-import

    # import habana_frameworks.torch.distributed.hccl # noqa: F401 pylint: disable=unused-import
    import habana_frameworks.torch.core as htcore  # noqa: F401 pylint: disable=unused-import
    from habana_frameworks.torch.hpex.experimental.transformer_engine import recipe

import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from lightningtools import reporter

torch.multiprocessing.set_start_method("forkserver", force=True)


@hydra.main(
    config_path=str(Path.cwd()) + "/configs", config_name="config", version_base="1.3"
)
def main(cfg):
    """
    os.symlink(
        os.path.abspath(".hydra/config.yaml"),
        os.path.join(wandb.run.dir, "hydra-config.yaml"),
    )
    wandb.save("hydra-config.yaml")
    """
    # os.chdir(hydra.utils.get_original_cwd())

    L.fabric.utilities.seed.seed_everything(42, workers=True)
    with torch.no_grad():
        dm = instantiate(cfg.data_module.data_module)
        trainer = instantiate(cfg.trainer)
    trainer.callbacks.append(reporter)

    if cfg.load_optimizer or cfg.last_ckpt is None:
        lightning_module = instantiate(cfg.lightning_module)
        lightning_module.set_config(cfg)
        if trainer.precision_plugin.precision == "fp8" and check_hpu():
            trainer.precision_plugin.convert_modules(
                lightning_module,
                replace_layers=True,
                # pylint: disable-next=possibly-used-before-assignment
                recipe=recipe.DelayedScaling(),
            )
        trainer.fit(lightning_module, dm, ckpt_path=cfg.last_ckpt)
    else:
        lightning_module = hydra.utils.get_method(cfg.lightning_module["_target_"])
        params = {
            k: instantiate(v) if isinstance(v, str) else v
            for k, v in cfg.lightning_module.items()
            if k != "_target_"
        }

        lightning_module = lightning_module.load_from_checkpoint(
            cfg.last_ckpt,
            **params,
            strict=False,
        )
        lightning_module.set_config(cfg)
        trainer.fit(lightning_module, dm)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
