import argparse
from pathlib import Path

import numpy as np

from .data import PreprocessConfig
from .engine import evaluate, predict, train, write_json
from .signals import CLASSES, SNR_LEVELS, generate_split, synthesize


def prepare(args):
    root = Path(args.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    if args.sample_rate <= 0 or not np.isfinite(args.sample_rate):
        raise ValueError("Sample rate must be finite and positive.")
    counts = {"train": args.train_per_class, "val": args.val_per_class,
              "test": args.test_per_class}
    for offset, (name, count) in enumerate(counts.items()):
        split = generate_split(count, args.seed + offset, args.samples, args.snrs)
        np.savez_compressed(root / f"{name}.npz", **split, sample_rate_hz=args.sample_rate)
        print(f"Generated {name}: {len(split['labels'])} pulses", flush=True)
    write_json(root / "manifest.json", {
        "source": "local_synthetic_single_pulse_not_AIMC_Spec", "classes": CLASSES,
        "split_seeds": {name: args.seed + i for i, name in enumerate(counts)},
        "per_class": counts, "n_samples": args.samples, "snr_levels_db": args.snrs,
        "sample_rate_hz": args.sample_rate,
        "snr_definition": "mean clean pulse power / expected complex AWGN power",
    })


def main():
    parser = argparse.ArgumentParser(description="Radar intra-pulse modulation recognition")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Generate independent synthetic IQ splits")
    generate.add_argument("--data-dir", default="data/synthetic")
    generate.add_argument("--train-per-class", type=int, default=256)
    generate.add_argument("--val-per-class", type=int, default=64)
    generate.add_argument("--test-per-class", type=int, default=100)
    generate.add_argument("--samples", type=int, default=1024)
    generate.add_argument("--sample-rate", type=float, default=1_000_000)
    generate.add_argument("--snrs", type=float, nargs="+", default=SNR_LEVELS)
    generate.add_argument("--seed", type=int, default=42)

    training = commands.add_parser("train", help="Train upstream STFT-CNN on IQ NPZ files")
    training.add_argument("--data-dir", default="data/synthetic")
    training.add_argument("--out-dir", default="outputs/stft_cnn")
    training.add_argument("--epochs", type=int, default=30)
    training.add_argument("--batch-size", type=int, default=64)
    training.add_argument("--lr", type=float, default=0.003)
    training.add_argument("--patience", type=int, default=12)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--samples", type=int, default=1024)
    training.add_argument("--sample-rate", type=float, default=1_000_000)
    training.add_argument("--image-size", type=int, default=96)

    inference = commands.add_parser("predict", help="Classify new IQ pulses")
    inference.add_argument("--checkpoint", default="outputs/stft_cnn/best.pt")
    inference.add_argument("--input", required=True)
    inference.add_argument("--output", default="outputs/predictions.json")
    inference.add_argument("--sample-rate", type=float)
    inference.add_argument("--min-confidence", type=float, default=0.0)

    evaluation = commands.add_parser("evaluate", help="Evaluate a labeled IQ NPZ")
    evaluation.add_argument("--checkpoint", default="outputs/stft_cnn/best.pt")
    evaluation.add_argument("--input", default="data/synthetic/test.npz")
    evaluation.add_argument("--output", default="outputs/evaluation.json")

    sample = commands.add_parser("sample", help="Generate a new IQ pulse to feed the model")
    sample.add_argument("--class", dest="modulation", choices=CLASSES, default="LFM_UP")
    sample.add_argument("--snr", type=float, default=6.0)
    sample.add_argument("--seed", type=int, default=2026)
    sample.add_argument("--samples", type=int, default=1024)
    sample.add_argument("--output", default="data/new_pulse.npy")

    demo = commands.add_parser("demo", help="Generate, train, evaluate and feed fresh signals")
    demo.add_argument("--data-dir", default="data/synthetic")
    demo.add_argument("--out-dir", default="outputs/stft_cnn")
    demo.add_argument("--epochs", type=int, default=30)

    official = commands.add_parser("official-demo", help="Small low-load demo using downloaded official 30-class IQ")
    official.add_argument("--raw-dir", default="data/official/raw/AIMC_Spec_v2_test")
    official.add_argument("--data-dir", default="data/official_demo")
    official.add_argument("--out-dir", default="outputs/official_demo")
    official.add_argument("--epochs", type=int, default=20)
    official.add_argument("--train-per-class", type=int, default=100)
    official.add_argument("--val-per-class", type=int, default=25)
    official.add_argument("--test-per-class", type=int, default=50)
    official.add_argument("--threads", type=int, default=1)
    official.add_argument("--duty-cycle", type=float, default=0.25)
    for command in (training, inference, evaluation, demo):
        command.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        command.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    try:
        if args.command == "generate":
            prepare(args)
        elif args.command == "train":
            if args.samples < 128 or args.image_size < 64 or args.sample_rate <= 0:
                raise ValueError("Need samples >= 128, image size >= 64 and sample rate > 0.")
            train(args.data_dir, args.out_dir,
                  PreprocessConfig(n_samples=args.samples, sample_rate_hz=args.sample_rate,
                                   image_size=args.image_size),
                  args.epochs, args.batch_size, args.lr, args.seed,
                  args.threads, args.device, args.patience)
        elif args.command == "predict":
            predict(args.checkpoint, args.input, args.output, args.device, args.threads,
                    args.sample_rate, args.min_confidence)
        elif args.command == "evaluate":
            evaluate(args.checkpoint, args.input, args.output, args.device, args.threads)
        elif args.command == "sample":
            iq, _ = synthesize(args.modulation, np.random.default_rng(args.seed), args.samples, args.snr)
            path = Path(args.output)
            if path.suffix != ".npy":
                raise ValueError("Sample output must use .npy extension.")
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, iq)
            print(f"Saved {args.modulation} pulse at {args.snr} dB to {path}")
        elif args.command == "demo":
            prepare(argparse.Namespace(data_dir=args.data_dir, train_per_class=256,
                                      val_per_class=64, test_per_class=100, samples=1024,
                                      sample_rate=1_000_000, snrs=SNR_LEVELS, seed=42))
            train(args.data_dir, args.out_dir, PreprocessConfig(), epochs=args.epochs,
                  device=args.device, threads=args.threads)
            fresh = generate_split(2, seed=98765, snrs=[6])
            path = Path(args.data_dir) / "fresh_pulses.npz"
            np.savez_compressed(path, **fresh, sample_rate_hz=1_000_000)
            out = Path(args.out_dir)
            result = predict(out / "best.pt", path, out / "fresh_predictions.json",
                             args.device, args.threads)
            correct = sum(row["best_class"] == CLASSES[label]
                          for row, label in zip(result["predictions"], fresh["labels"]))
            result["ground_truth"] = [CLASSES[i] for i in fresh["labels"]]
            result["correct"] = correct
            result["accuracy"] = correct / len(fresh["labels"])
            write_json(out / "fresh_predictions.json", result)
            print(f"Fresh signals: {correct}/{len(fresh['labels'])} correct")
        elif args.command == "official-demo":
            from .official_data import prepare_small_official
            from .resources import ResourceConfig
            limits = ResourceConfig(threads=args.threads, duty_cycle=args.duty_cycle)
            manifest = prepare_small_official(
                args.raw_dir, args.data_dir, args.train_per_class, args.val_per_class,
                args.test_per_class, resources=limits)
            train(args.data_dir, args.out_dir,
                  PreprocessConfig(n_samples=1000, sample_rate_hz=50_000_000),
                  epochs=args.epochs, threads=args.threads, device="cpu", resources=limits,
                  lr=0.0006, model_kwargs={"use_paper_preprocess": True,
                                         "preproc_iters": 6, "denom": "var"})
            with np.load(Path(args.data_dir) / "test.npz", allow_pickle=False) as archive:
                selected = [np.flatnonzero((archive["labels"] == i) & (archive["snr_db"] == 6))[0]
                            for i in range(len(manifest["classes"]))]
                iq = archive["iq"][selected]
                labels = archive["labels"][selected]
            input_path = Path(args.data_dir) / "heldout_pulses.npz"
            np.savez(input_path, iq=iq, sample_rate_hz=50_000_000)
            output_path = Path(args.out_dir) / "heldout_predictions.json"
            result = predict(Path(args.out_dir) / "best.pt", input_path, output_path,
                             device="cpu", threads=args.threads)
            result["ground_truth"] = [manifest["classes"][i] for i in labels]
            result["correct"] = sum(row["best_class"] == name for row, name in
                                    zip(result["predictions"], result["ground_truth"]))
            write_json(output_path, result)
            print(f"Fed official held-out signals: {result['correct']}/{len(labels)} correct")
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
