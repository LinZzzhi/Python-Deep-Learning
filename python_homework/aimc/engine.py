import csv
from contextlib import nullcontext
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import PreprocessConfig, encode_iq, load_iq, load_split
from .plots import plot_confusion, plot_examples, plot_training
from .resources import ResourceGovernor
from .upstream import MainModel, UPSTREAM_COMMIT


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def setup(seed=42, threads=4):
    if threads < 1:
        raise ValueError("threads must be positive.")
    random.seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def select_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use --device cpu.")
    return torch.device(name)


def check_sample_rate(path, expected, supplied=None):
    actual = supplied
    if Path(path).suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "sample_rate_hz" in archive:
                stored = float(archive["sample_rate_hz"].item())
                if supplied is not None and not np.isclose(supplied, stored):
                    raise ValueError("Specified sample rate differs from file metadata.")
                actual = stored
    if actual is not None and (not np.isfinite(actual) or not np.isclose(actual, expected)):
        raise ValueError(f"Sample rate mismatch: input {actual} Hz, model {expected} Hz.")


def verify_disjoint(splits):
    # Detect copied observations across splits before any training.
    seen = set()
    for name, iq in splits:
        hashes = {hashlib.sha256(pulse.tobytes()).digest() for pulse in iq}
        if seen & hashes:
            raise ValueError(f"Duplicate IQ observations across dataset splits: {name}.")
        seen.update(hashes)


def _loader(images, labels, batch_size, shuffle=False):
    return DataLoader(TensorDataset(images, torch.from_numpy(labels)),
                      batch_size=batch_size, shuffle=shuffle)


def _pass(model, loader, device, optimizer=None, governor=None):
    model.train(optimizer is not None)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            with governor.burst() if governor else nullcontext():
                images = images.to(device).repeat(1, 3, 1, 1)
                labels = labels.to(device)
                logits = model(images)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("Training loss is not finite.")
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * len(labels)
                correct += int((logits.argmax(1) == labels).sum().item())
                count += len(labels)
    return total_loss / count, correct / count


@torch.no_grad()
def probabilities(model, images, device, batch_size=128, governor=None):
    model.eval()
    outputs = []
    for start in range(0, len(images), batch_size):
        with governor.burst() if governor else nullcontext():
            batch = images[start:start + batch_size].to(device).repeat(1, 3, 1, 1)
            outputs.append(model(batch).softmax(dim=1).cpu().numpy())
    return np.concatenate(outputs)


def classification_metrics(labels, predicted, classes, snr=None):
    n_classes = len(classes)
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (labels, predicted), 1)
    tp = matrix.diagonal()
    precision = tp / np.maximum(matrix.sum(axis=0), 1)
    recall = tp / np.maximum(matrix.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    result = {
        "samples": len(labels), "accuracy": float(np.mean(labels == predicted)),
        "macro_f1": float(f1.mean()), "confusion_matrix": matrix.tolist(),
        "per_class": {
            name: {"precision": float(precision[i]), "recall": float(recall[i]),
                   "f1": float(f1[i]), "support": int(matrix[i].sum())}
            for i, name in enumerate(classes)
        },
    }
    if snr is not None:
        result["accuracy_by_snr_db"] = {
            str(float(level)): float(np.mean(predicted[snr == level] == labels[snr == level]))
            for level in np.unique(snr)
        }
    return result


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = MainModel(len(checkpoint["classes"]), **checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    config = PreprocessConfig(**checkpoint["preprocess"])
    return model, config, checkpoint


def train(data_dir, out_dir, config, epochs=30, batch_size=64, lr=0.003,
          seed=42, threads=4, device="auto", patience=12, resources=None, model_kwargs=None):
    if epochs < 1 or batch_size < 1 or not np.isfinite(lr) or lr <= 0 or patience < 1:
        raise ValueError("epochs, batch size, learning rate and patience must be positive.")
    if resources:
        threads = resources.threads
    setup(seed, threads)
    device = select_device(device)
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    governor = ResourceGovernor(resources, out_dir, device) if resources else None
    train_iq, train_y, classes, _ = load_split(data_dir / "train.npz")
    val_iq, val_y, val_classes, _ = load_split(data_dir / "val.npz")
    test_iq, test_y, test_classes, test_snr = load_split(data_dir / "test.npz")
    if classes != val_classes or classes != test_classes:
        raise ValueError("All splits must use identical classes in identical order.")
    if set(np.unique(train_y)) != set(range(len(classes))):
        raise ValueError("Training split must contain every class.")
    for name in ("train", "val", "test"):
        check_sample_rate(data_dir / f"{name}.npz", config.sample_rate_hz)
    verify_disjoint([("train", train_iq), ("val", val_iq), ("test", test_iq)])
    print(f"Device={device}; train={len(train_y)}, val={len(val_y)}, test={len(test_y)}", flush=True)
    train_x = encode_iq(train_iq, config)
    val_x = encode_iq(val_iq, config)
    # Store constructor flags with the weights so inference uses identical behavior.
    model_kwargs = dict(model_kwargs) if model_kwargs is not None else {"use_paper_preprocess": False}
    model = MainModel(len(classes), **model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_loader = _loader(train_x, train_y, batch_size, shuffle=True)
    val_loader = _loader(val_x, val_y, batch_size)
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    run_config = {
        "data_dir": str(data_dir.resolve()), "data_source": manifest.get("source", "user_supplied"),
        "preprocess": config.to_dict(), "model_kwargs": model_kwargs,
        "classes": classes, "seed": seed, "epochs": epochs, "batch_size": batch_size,
        "lr": lr, "device": str(device), "threads": threads, "patience": patience,
        "upstream_commit": UPSTREAM_COMMIT, "torch_version": str(torch.__version__),
        "split_sizes": {"train": len(train_y), "val": len(val_y), "test": len(test_y)},
        "resources": resources.to_dict() if resources else None,
    }
    write_json(out_dir / "config.json", run_config)
    plot_examples(train_x, train_y, classes, out_dir / "spectrograms.png")
    history, best_loss, stale, best_epoch = [], float("inf"), 0, 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _pass(model, train_loader, device, optimizer, governor)
        val_loss, val_acc = _pass(model, val_loader, device, governor=governor)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
                        "val_loss": val_loss, "val_accuracy": val_acc,
                        "lr": optimizer.param_groups[0]["lr"]})
        print(f"Epoch {epoch:02d}/{epochs}: loss={train_loss:.4f} "
              f"train_acc={train_acc:.3%} val_loss={val_loss:.4f} val_acc={val_acc:.3%}", flush=True)
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "classes": classes, "preprocess": config.to_dict(), "model_kwargs": model_kwargs,
                        "epoch": epoch, "val_loss": val_loss, "val_accuracy": val_acc,
                        "upstream_commit": UPSTREAM_COMMIT}, out_dir / "best.pt")
        else:
            stale += 1
        scheduler.step()
        write_json(out_dir / "history.json", history)
        if stale >= patience:
            print(f"Early stop after {patience} epochs without validation loss improvement.", flush=True)
            break
    with (out_dir / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    plot_training(history, out_dir / "training_curves.png")
    model, config, _ = load_checkpoint(out_dir / "best.pt", device)
    test_x = encode_iq(test_iq, config)
    predicted = probabilities(model, test_x, device, batch_size, governor).argmax(1)
    metrics = classification_metrics(test_y, predicted, classes, test_snr)
    metrics.update({"best_epoch": best_epoch, "training_seconds": time.perf_counter() - started,
                    "data_source": run_config["data_source"], "classes": classes})
    write_json(out_dir / "metrics.json", metrics)
    plot_confusion(np.asarray(metrics["confusion_matrix"]), classes, out_dir / "confusion_matrix.png")
    if governor:
        governor.monitor(force=True)
        governor.write_status(governor.gpu_status(), "complete")
    print(f"Independent test: accuracy={metrics['accuracy']:.3%}, "
          f"macro_f1={metrics['macro_f1']:.4f}; checkpoint={out_dir / 'best.pt'}", flush=True)
    return metrics


def predict(checkpoint_path, input_path, output_path, device="auto", threads=4,
            sample_rate_hz=None, min_confidence=0.0):
    if not 0 <= min_confidence <= 1:
        raise ValueError("Minimum confidence must be in [0,1].")
    setup(threads=threads)
    device = select_device(device)
    model, config, checkpoint = load_checkpoint(checkpoint_path, device)
    check_sample_rate(input_path, config.sample_rate_hz, sample_rate_hz)
    images = encode_iq(load_iq(input_path), config)
    scores = probabilities(model, images, device)
    classes = checkpoint["classes"]
    result = {
        "input": str(input_path), "checkpoint": str(checkpoint_path),
        "classes": classes, "sample_rate_hz": config.sample_rate_hz, "predictions": [],
    }
    for i, score in enumerate(scores):
        best = int(score.argmax())
        confidence = float(score[best])
        result["predictions"].append({
            "index": i, "label": classes[best] if confidence >= min_confidence else "uncertain",
            "best_class": classes[best], "confidence": confidence,
            "scores": {name: float(score[j]) for j, name in enumerate(classes)},
        })
    write_json(output_path, result)
    for row in result["predictions"][:20]:
        print(f"Pulse {row['index']:03d}: {row['label']:10s} confidence={row['confidence']:.3%}")
    print(f"Saved {len(scores)} predictions to {output_path}", flush=True)
    return result


def evaluate(checkpoint_path, input_path, output_path, device="auto", threads=4):
    setup(threads=threads)
    device = select_device(device)
    model, config, checkpoint = load_checkpoint(checkpoint_path, device)
    iq, labels, classes, snr = load_split(input_path)
    if classes != checkpoint["classes"]:
        raise ValueError("Dataset class order differs from checkpoint.")
    check_sample_rate(input_path, config.sample_rate_hz)
    predicted = probabilities(model, encode_iq(iq, config), device).argmax(1)
    metrics = classification_metrics(labels, predicted, classes, snr)
    write_json(output_path, metrics)
    print(f"Accuracy={metrics['accuracy']:.3%}, macro_f1={metrics['macro_f1']:.4f}")
    return metrics
