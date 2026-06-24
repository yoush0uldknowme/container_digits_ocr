import torch


def compute_digit_and_sequence_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    predictions = logits.argmax(dim=-1)
    correct_digits = predictions.eq(targets)
    digit_acc = correct_digits.float().mean().item()
    sequence_acc = correct_digits.all(dim=1).float().mean().item()
    return digit_acc, sequence_acc
