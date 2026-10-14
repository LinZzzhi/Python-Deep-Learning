import numpy as np
import pytest
import torch

from aimc.data import PreprocessConfig, canonical_iq, encode_iq
from aimc.engine import classification_metrics, predict, verify_disjoint
from aimc.signals import CLASSES, generate_split, synthesize
from aimc.upstream import MainModel


def test_all_classes_and_reproducible_independent_splits():
    first = generate_split(5, 42)
    same = generate_split(5, 42)
    other = generate_split(5, 43)
    np.testing.assert_array_equal(first["iq"], same["iq"])
    assert np.bincount(first["labels"]).tolist() == [5] * len(CLASSES)
    verify_disjoint([("train", canonical_iq(first["iq"])),
                     ("val", canonical_iq(other["iq"]))])
    with pytest.raises(ValueError, match="Duplicate"):
        verify_disjoint([("train", canonical_iq(first["iq"])),
                         ("test", canonical_iq(same["iq"]))])


def test_awgn_snr_definition():
    noisy, clean = synthesize("CW", np.random.default_rng(3), 100_000, 3)
    observed = 10 * np.log10(np.mean(abs(clean) ** 2) / np.mean(abs(noisy - clean) ** 2))
    assert abs(observed - 3) < 0.1


def test_iq_layouts_and_reject_invalid_data():
    iq = generate_split(1, 7)["iq"]
    target = canonical_iq(iq)
    np.testing.assert_array_equal(canonical_iq(target.transpose(0, 2, 1)), target)
    np.testing.assert_array_equal(canonical_iq(iq[0]), target[:1])
    np.testing.assert_array_equal(canonical_iq(target[0].T), target[:1])
    with pytest.raises(ValueError, match="NaN"):
        canonical_iq(np.full(1024, complex(np.nan, 1)))
    with pytest.raises(ValueError, match="all-zero"):
        canonical_iq(np.zeros(1024, dtype=np.complex64))
    with pytest.raises(ValueError, match="Expected 1024"):
        encode_iq(iq[:, :512], PreprocessConfig())


def test_upstream_forward_backward_and_spectrogram_batch_parity():
    torch.set_num_threads(2)
    iq = generate_split(1, 9)["iq"]
    images = encode_iq(iq, PreprocessConfig())
    single = encode_iq(iq[:1], PreprocessConfig())
    torch.testing.assert_close(images[:1], single)
    assert images.shape == (len(CLASSES), 1, 96, 96)
    assert images.min() >= 0 and images.max() <= 1
    model = MainModel(len(CLASSES), use_paper_preprocess=False)
    logits = model(images.repeat(1, 3, 1, 1))
    assert logits.shape == (len(CLASSES), len(CLASSES))
    torch.nn.functional.cross_entropy(logits, torch.arange(len(CLASSES))).backward()
    assert model.conv1.weight.grad is not None
    assert torch.isfinite(model.conv1.weight.grad).all()


def test_upstream_paper_preprocessing_preserves_input_variation():
    torch.manual_seed(42)
    images = encode_iq(generate_split(1, 9)["iq"], PreprocessConfig()).repeat(1, 3, 1, 1)
    model = MainModel(len(CLASSES), use_paper_preprocess=True, preproc_iters=6, denom="var")
    logits = model(images)
    assert torch.isfinite(logits).all()
    assert logits.std(dim=0).mean() > 1e-5
    torch.nn.functional.cross_entropy(logits, torch.arange(len(CLASSES))).backward()
    assert torch.isfinite(model.conv1.weight.grad).all()


def test_known_metrics():
    metrics = classification_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]),
                                     ["A", "B"], np.array([0, 0, 6, 6]))
    assert metrics["accuracy"] == 0.75
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["accuracy_by_snr_db"] == {"0.0": 0.5, "6.0": 1.0}


def test_checkpoint_reload_and_real_file_inference(tmp_path):
    config = PreprocessConfig()
    model = MainModel(len(CLASSES), use_paper_preprocess=False).eval()
    checkpoint = tmp_path / "model.pt"
    torch.save({"state_dict": model.state_dict(), "classes": CLASSES,
                "model_kwargs": {"use_paper_preprocess": False},
                "preprocess": config.to_dict()}, checkpoint)
    iq = generate_split(1, 42)["iq"]
    input_path = tmp_path / "pulses.npz"
    np.savez(input_path, iq=iq, sample_rate_hz=config.sample_rate_hz)
    result = predict(checkpoint, input_path, tmp_path / "predictions.json", device="cpu")
    with torch.no_grad():
        expected = model(encode_iq(iq, config).repeat(1, 3, 1, 1)).softmax(1).numpy()
    actual = np.array([list(row["scores"].values()) for row in result["predictions"]])
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    assert (tmp_path / "predictions.json").is_file()
    with pytest.raises(ValueError, match="differs from file"):
        predict(checkpoint, input_path, tmp_path / "bad.json", sample_rate_hz=2_000_000)
