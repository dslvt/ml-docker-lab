"""Обучение нейроранкера на взаимодействиях user × item.

Модель: скалярное произведение эмбеддингов пользователя и объекта плюс MLP
над ними. Loss: BPR (наблюдаемый объект против случайного).
Метрика: recall@10 для последнего объекта пользователя среди 100 случайных.

Checkpoint пишется в --out после каждой эпохи и при SIGTERM.
MLflow включается, если задана переменная MLFLOW_TRACKING_URI.
"""
import argparse
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

stop_requested = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/data", help="каталог с interactions.csv")
    p.add_argument("--out", default="/outputs", help="каталог для checkpoint")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--resume", action="store_true", help="продолжить с checkpoint")
    p.add_argument("--write-test", action="store_true", help="проверить запись в --data")
    return p.parse_args()


def on_stop(signum, frame):
    global stop_requested
    stop_requested = True
    print(f"{signal.Signals(signum).name} received, saving checkpoint", flush=True)


class Ranker(nn.Module):
    def __init__(self, n_users, n_items, dim):
        super().__init__()
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user.weight, std=0.05)
        nn.init.normal_(self.item.weight, std=0.05)
        self.mlp = nn.Sequential(
            nn.Linear(3 * dim, 1024), nn.ReLU(),
            nn.Linear(1024, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, users, items):
        u, i = self.user(users), self.item(items)
        return (u * i).sum(-1) + self.mlp(torch.cat([u, i, u * i], dim=-1)).squeeze(-1)


def load_data(data_dir):
    """Последний объект каждого пользователя уходит в validation, остальные в train."""
    rows = np.loadtxt(Path(data_dir) / "interactions.csv", delimiter=",",
                      skiprows=1, dtype=np.int64)
    rows = rows[np.lexsort((rows[:, 2], rows[:, 0]))]
    is_last = np.append(rows[1:, 0] != rows[:-1, 0], True)
    return rows[~is_last, :2], rows[is_last, :2], rows[:, 0].max() + 1, rows[:, 1].max() + 1


@torch.no_grad()
def recall_at_10(model, valid, n_items, device, chunk=32):
    """Первый кандидат — настоящий объект, остальные 100 — случайные.
    Пользователи обрабатываются частями по chunk, чтобы не держать в памяти все оценки."""
    rng = np.random.default_rng(0)
    candidates = np.concatenate(
        [valid[:, 1:2], rng.integers(0, n_items, size=(len(valid), 100))], axis=1)
    users = np.repeat(valid[:, 0:1], 101, axis=1)
    hits = 0
    for start in range(0, len(valid), chunk):
        part = slice(start, start + chunk)
        scores = model(torch.as_tensor(users[part], device=device),
                       torch.as_tensor(candidates[part], device=device))
        hits += ((scores[:, 1:] > scores[:, :1]).sum(dim=1) < 10).sum().item()
    return hits / len(valid)


def save_checkpoint(path, model, optimizer, epoch, run_id):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "run_id": run_id}, tmp)
    os.replace(tmp, path)


def main():
    args = parse_args()
    if args.write_test:
        try:
            (Path(args.data) / "write_test.txt").write_text("test\n")
        except OSError as e:
            sys.exit(f"write to {args.data} failed: {e}")
        print(f"write to {args.data} succeeded")
        return

    signal.signal(signal.SIGTERM, on_stop)
    signal.signal(signal.SIGINT, on_stop)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, valid, n_users, n_items = load_data(args.data)
    model = Ranker(n_users, n_items, args.dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"device={device} torch={torch.__version__} users={n_users} items={n_items} "
          f"train={len(train)}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "ckpt.pt"
    start_epoch, run_id = 1, None
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, run_id = ckpt["epoch"] + 1, ckpt["run_id"]
        print(f"resumed from epoch {ckpt['epoch']}", flush=True)

    mlflow = None
    if os.environ.get("MLFLOW_TRACKING_URI"):
        # в образе нет git, без этой настройки MLflow пишет предупреждение при каждом запуске
        os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
        import mlflow
        mlflow.set_experiment("ranker")
        run_id = mlflow.start_run(run_id=run_id).info.run_id
        if start_epoch == 1:
            mlflow.log_params({k: v for k, v in vars(args).items()
                               if k in ("epochs", "lr", "batch_size", "dim")})
        print(f"mlflow run_id={run_id}", flush=True)

    train_t = torch.as_tensor(train, device=device)
    epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        model.train()
        total, batches = 0.0, 0
        for idx in torch.randperm(len(train_t), device=device).split(args.batch_size):
            users, pos = train_t[idx, 0], train_t[idx, 1]
            neg = torch.randint(0, n_items, (len(idx),), device=device)
            loss = -nn.functional.logsigmoid(model(users, pos) - model(users, neg)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total, batches = total + loss.item(), batches + 1
            if stop_requested:
                break
        if stop_requested:
            epoch -= 1
            break
        model.eval()
        recall = recall_at_10(model, valid, n_items, device)
        seconds = time.time() - started
        print(f"epoch {epoch}/{args.epochs} loss={total / batches:.4f} "
              f"recall@10={recall:.3f} time={seconds:.1f}s", flush=True)
        save_checkpoint(ckpt_path, model, optimizer, epoch, run_id)
        if mlflow:
            mlflow.log_metrics({"loss": total / batches, "recall_at_10": recall,
                                "epoch_seconds": seconds}, step=epoch)

    save_checkpoint(ckpt_path, model, optimizer, epoch, run_id)
    print(f"checkpoint saved: {ckpt_path} epoch={epoch}", flush=True)
    if mlflow:
        mlflow.log_artifact(str(ckpt_path))
        mlflow.end_run("KILLED" if stop_requested else "FINISHED")


if __name__ == "__main__":
    main()
