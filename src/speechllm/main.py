# pylint: disable=wrong-import-position, wrong-import-order
import os
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import numpy as np
import omegaconf

from speechllm.utils import check_hpu

if check_hpu():
    # Deprecated comment? workaround for broken frozen ddp training
    # modelcheckpointing on distributed would not save the best ckpt when gpu_migration is enabled
    # because dist sync checks for hccl not nccl
    # import habana_frameworks.torch.gpu_migration  # noqa: F401 pylint: disable=unused-import

    import habana_frameworks.torch.distributed.hccl  # noqa: F401 pylint: disable=unused-import
    import habana_frameworks.torch.core as htcore  # noqa: F401 pylint: disable=unused-import

    from habana_frameworks.torch.hpex.experimental.transformer_engine import recipe

    # No significant performance improvement
    # from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
    # from optimum.habana.transformers.models import GaudiLlamaAttention
    # import transformers
    # transformers.models.llama.modeling_llama.LlamaAttention = GaudiLlamaAttention
    # adapt_transformers_to_gaudi()

import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from lightningtools import reporter

# training crashes once in a while without forkserver for reasons unknown
torch.multiprocessing.set_start_method("forkserver", force=True)


@hydra.main(
    config_path=str(Path.cwd()) + "/configs", config_name="config", version_base="1.3"
)
def main(cfg):
    """
    os.symlink(
        os.path.abspath(".hydra/config.yaml"),
        os.path.join(wandb.run.dir, "hydra-config.yaml")
    )
    wandb.save("hydra-config.yaml")
    """
    # os.chdir(hydra.utils.get_original_cwd())

    # Initializes a shared memory for global step
    # I have no idea what I am doing but it works
    try:
        shm = SharedMemory(
            create=True, size=4, name=f"global_step_{os.environ['MASTER_PORT']}"
        )
    except FileExistsError:
        shm = SharedMemory(
            create=False, size=4, name=f"global_step_{os.environ['MASTER_PORT']}"
        )
    arr = np.ndarray([1], np.int32, shm.buf)
    arr[0] = 0

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
        params = {}
        for k, v in cfg.lightning_module.items():
            if k != "_target_":
                if isinstance(v, omegaconf.dictconfig.DictConfig):
                    params[k] = instantiate(v)
                else:
                    params[k] = v
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
