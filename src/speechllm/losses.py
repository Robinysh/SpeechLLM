import torch.nn.functional as F
from icecream import ic
from lightningtools import reporter


def cross_entropy_loss(hubert_preds, hubert_embs):
    ic(hubert_preds.shape)
    ic(hubert_embs.shape)
    reporter.report(
        "metrics/accuracy", (hubert_preds.argmax(-1) == hubert_embs).float().mean()
    )
    return {"cross_entropy": F.cross_entropy(hubert_preds, hubert_embs)}


def l2_loss(hubert_preds, hubert_embs):
    ic(hubert_embs.shape)
    return {"l2": F.mse_loss(hubert_preds, hubert_embs)}


def anygpt_loss(lm_output):
    return {"anygpt_loss": lm_output.loss}
