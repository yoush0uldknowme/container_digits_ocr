import argparse
import copy
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.data import ContainerDigitsDataset
from src.model_v3 import Fixed6DigitAttentionCRNN
from src.train_utils import append_metrics, evaluate_model, format_predictions, save_json, sequence_loss


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds)); hours, rem = divmod(seconds, 3600); minutes, seconds = divmod(rem, 60)
    return f"{hours:d}h {minutes:02d}m {seconds:02d}s" if hours else f"{minutes:02d}m {seconds:02d}s"


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9995) -> None:
        self.model = copy.deepcopy(model).eval()
        self.decay = decay
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        source = model.state_dict()
        for name, value in self.model.state_dict().items():
            value.copy_(value * self.decay + source[name].detach() * (1.0 - self.decay)) if value.is_floating_point() else value.copy_(source[name])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V3 attention CRNN for fixed 6-digit OCR.")
    parser.add_argument("--data-root", type=Path, default=Path("dataset_container_digits"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/v3_attention_seed1"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-max-samples", type=int, default=None)
    parser.add_argument("--val-max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260611)
    return parser.parse_args()


def main(data_root: Optional[Path] = None, run_dir: Optional[Path] = None, epochs: Optional[int] = None,
         batch_size: Optional[int] = None, lr: Optional[float] = None, num_workers: Optional[int] = None,
         train_max_samples: Optional[int] = None, val_max_samples: Optional[int] = None,
         seed: Optional[int] = None) -> None:
    args = parse_args()
    for name, value in locals().copy().items():
        if name != "args" and value is not None and hasattr(args, name): setattr(args, name, value)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ContainerDigitsDataset(args.data_root, "train", augment=True, max_samples=args.train_max_samples)
    val_ds = ContainerDigitsDataset(args.data_root, "val", augment=False, max_samples=args.val_max_samples)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers,
                            pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)

    model = Fixed6DigitAttentionCRNN().to(device)
    ema = ModelEMA(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_sequence_acc = -1.0
    best_digit_acc = -1.0

    save_json(args.run_dir / "config.json", {"model": "v3_attention_crnn", "data_root": str(args.data_root),
              "run_dir": str(args.run_dir), "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
              "device": str(device), "train_samples": len(train_ds), "val_samples": len(val_ds), "seed": args.seed,
              "started_at": now_text()})
    print(f"[{now_text()}] Start V3 training | device={device} | train={len(train_ds)} val={len(val_ds)}")
    print(f"Run dir: {args.run_dir} | epochs={args.epochs} batch={args.batch_size} lr={args.lr} seed={args.seed}")

    training_start = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time(); model.train(); loss_sum = 0.0; count = 0; last_logits = None; last_labels = None
        for images, targets, labels in train_loader:
            images = images.to(device, non_blocking=True); targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images); loss = sequence_loss(logits, targets)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update(); ema.update(model)
            loss_sum += loss.item() * images.size(0); count += images.size(0); last_logits = logits.detach(); last_labels = labels
        scheduler.step()
        metrics = evaluate_model(ema.model, val_loader, device)
        train_loss = loss_sum / max(count, 1); current_lr = optimizer.param_groups[0]["lr"]
        epoch_seconds = time.time() - epoch_start; elapsed = time.time() - training_start; eta = elapsed / epoch * (args.epochs - epoch)
        improved = metrics["sequence_acc"] > best_sequence_acc or (metrics["sequence_acc"] == best_sequence_acc and metrics["digit_acc"] > best_digit_acc)
        if improved: best_sequence_acc, best_digit_acc = metrics["sequence_acc"], metrics["digit_acc"]
        row = {"time": now_text(), "epoch": epoch, "train_loss": f"{train_loss:.6f}", "val_loss": f"{metrics['loss']:.6f}",
               "val_digit_acc": f"{metrics['digit_acc']:.6f}", "val_sequence_acc": f"{metrics['sequence_acc']:.6f}",
               "best_sequence_acc": f"{best_sequence_acc:.6f}", "lr": f"{current_lr:.8f}",
               "epoch_seconds": f"{epoch_seconds:.2f}", "elapsed": format_duration(elapsed), "eta": format_duration(eta)}
        append_metrics(args.run_dir / "metrics.csv", row)
        checkpoint = {"model_state": ema.model.state_dict(), "raw_model_state": model.state_dict(), "epoch": epoch,
                      "val_sequence_acc": metrics["sequence_acc"], "val_digit_acc": metrics["digit_acc"],
                      "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
        torch.save(checkpoint, args.run_dir / "last_model.pt")
        if improved: torch.save(checkpoint, args.run_dir / "best_model.pt")
        samples = format_predictions(last_logits, last_labels) if last_logits is not None else ""
        print(f"[{now_text()}] Epoch {epoch:03d}/{args.epochs:03d} loss={train_loss:.4f} val={metrics['loss']:.4f} "
              f"digit={metrics['digit_acc']*100:.4f}% seq={metrics['sequence_acc']*100:.4f}% best={best_sequence_acc*100:.4f}% "
              f"lr={current_lr:.7f} time={format_duration(epoch_seconds)} elapsed={format_duration(elapsed)} eta={format_duration(eta)}")
        print(f"    samples pred/label: {samples}")
    print(f"[{now_text()}] Finished. Best model: {args.run_dir / 'best_model.pt'}")


if __name__ == "__main__": main()
