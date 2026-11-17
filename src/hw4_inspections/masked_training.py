"""Separate label-free pretraining and supervised optimization routines."""
from __future__ import annotations

import random

import numpy as np
import torch
from sklearn.metrics import average_precision_score


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def infer(model, data, batch_size, *, embeddings=False):
    model.eval()
    output = []
    for start in range(0, len(data[0]), batch_size):
        batch = tuple(a[start:start + batch_size] for a in data)
        result = model.encode(*batch)[0] if embeddings else model(*batch).sigmoid()
        output.append(result.cpu().numpy())
    return np.concatenate(output)


def pretrain(model, data, *, epochs, batch_size, mask_probability, seed, log):
    """No target argument or validation features enter this function."""
    seed_all(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    rng = np.random.default_rng(seed)
    curve = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.
        for start in range(0, len(data[0]), batch_size):
            if start == 0:
                order = rng.permutation(len(data[0]))
            idx = torch.as_tensor(order[start:start + batch_size], device=data[0].device)
            batch = tuple(a[idx] for a in data)
            mask = model.sample_mask(len(idx), mask_probability, data[0].device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.pretraining_loss(*batch, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite pretraining loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += loss.item() * len(idx)
            if epoch == 1 and start == 0:
                log("pretrain_first_batch", loss_device=str(loss.device), targets_used=False)
        row = {"epoch": epoch, "reconstruction_loss": total / len(data[0])}
        curve.append(row)
        log("pretrain_epoch_complete", **row)
    return curve


def finetune(model, data, target, *, epochs, patience, batch_size, seed, log,
             eval_data=None, eval_y=None):
    """Only the inner epoch-selection run may receive evaluation labels."""
    if (eval_data is None) != (eval_y is None):
        raise ValueError("Both inner evaluation inputs and labels are required")
    seed_all(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.01)
    rng = np.random.default_rng(seed)
    best_ap, best_epoch = -1., 1
    curve = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(target))
        total = 0.
        for start in range(0, len(order), batch_size):
            idx = torch.as_tensor(order[start:start + batch_size], device=target.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(*(a[idx] for a in data))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target[idx])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite fine-tuning loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += loss.item() * len(idx)
            if epoch == 1 and start == 0:
                log("finetune_first_batch", logits_device=str(logits.device), loss_device=str(loss.device))
        row = {"epoch": epoch, "train_loss": total / len(target)}
        if eval_data is not None:
            ap = float(average_precision_score(eval_y, infer(model, eval_data, batch_size)))
            row["inner_ap"] = ap
            if ap > best_ap:
                best_ap, best_epoch = ap, epoch
        curve.append(row)
        log("finetune_epoch_complete", **row)
        if eval_data is not None and epoch - best_epoch >= patience:
            break
    return best_epoch if eval_data is not None else epochs, curve
