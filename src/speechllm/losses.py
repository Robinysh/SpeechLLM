import torch
import torch.nn.functional as F
from lightningtools import reporter


def cross_entropy_loss(hubert_preds, hubert_embs):
    reporter.report(
        "metrics/accuracy", (hubert_preds.argmax(-1) == hubert_embs).float().mean()
    )
    return {"cross_entropy": F.cross_entropy(hubert_preds, hubert_embs)}


def l2_loss(hubert_preds, hubert_embs):
    return {"l2": F.mse_loss(hubert_preds, hubert_embs)}


def anygpt_loss(lm_output):
    return {"anygpt_loss": lm_output.loss}


def context_hidden_state_distill_loss(
    lm_output,
    teacher_output,
    model_input,
    teacher_answer_start_position,
    answer_start_position,
):
    # TODO mask impl
    losses = []
    preds_hidden_states = torch.stack(lm_output.hidden_states, dim=1)
    teacher_hidden_states = torch.stack(teacher_output.hidden_states, dim=1)
    for idx in range(len(model_input.input_ids)):
        preds = preds_hidden_states[idx, :, answer_start_position[idx] :]
        mask = model_input.attention_mask[idx, answer_start_position[idx] :]
        soft_labels = teacher_hidden_states[
            idx, :, teacher_answer_start_position[idx] :
        ].detach()

        loss = (((preds - soft_labels) ** 2).mean((0, -1)) * mask).sum() / mask.sum()
        losses.append(loss)
    return {"context_hidden_states_distill_loss": sum(losses) / len(losses)}


def full_hidden_state_distill_loss(
    lm_output,
    teacher_output,
    model_input,
):
    # TODO mask impl
    preds = torch.stack(lm_output.hidden_states[2:-2], dim=2)
    teacher = torch.stack(teacher_output.hidden_states[2:-2], dim=2).detach()
    mask = model_input.attention_mask
    loss = (((preds - teacher) ** 2).mean((2, 3)) * mask).sum() / mask.sum()
    return {"context_hidden_states_distill_loss": loss}


def context_distill_loss(
    lm_output,
    teacher_output,
    model_input,
    teacher_answer_start_position,
    answer_start_position,
):
    # TODO mask impl
    losses = []
    for idx in range(len(model_input.input_ids)):
        preds = lm_output.logits[idx][answer_start_position[idx] :]
        mask = model_input.attention_mask[idx][answer_start_position[idx] :]
        soft_labels = teacher_output.logits[idx][teacher_answer_start_position[idx] :]
        soft_label_probs = (
            soft_labels - soft_labels.logsumexp(dim=-1, keepdim=True)
        ).detach()
        preds_probs = preds - preds.logsumexp(dim=-1, keepdim=True)
        loss = (
            F.kl_div(
                preds_probs, soft_label_probs, reduction="none", log_target=True
            ).sum(-1)
            * mask
        ).sum() / mask.sum()
        losses.append(loss)
    return {"context_distill_loss": sum(losses) / len(losses)}
