import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEST_ENV = os.environ.copy()
TEST_ENV["TOKENIZERS_PARALLELISM"] = "false"


@pytest.mark.slow
def test_train_pretrain_toy_model(tmp_path):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    # train scripts resolve tokenizer at ../model relative to the working dir
    (tmp_path / "model").symlink_to(ROOT / "model")
    save_dir = tmp_path / "out"

    cmd = [
        sys.executable,
        str(ROOT / "trainer/train_pretrain.py"),
        "--epochs",
        "1",
        "--batch_size",
        "2",
        "--max_seq_len",
        "24",
        "--hidden_size",
        "16",
        "--num_hidden_layers",
        "2",
        "--num_workers",
        "0",
        "--accumulation_steps",
        "1",
        "--learning_rate",
        "1e-3",
        "--save_interval",
        "1",
        "--log_interval",
        "1",
        "--device",
        "cpu",
        "--data_path",
        str(ROOT / "tests/data/pretrain_toy.jsonl"),
        "--save_dir",
        str(save_dir),
        "--from_weight",
        "none",
    ]
    result = subprocess.run(
        cmd,
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=600,
        env=TEST_ENV,
    )
    assert result.returncode == 0, f"pretrain failed:\n{result.stdout}\n{result.stderr}"
    assert (save_dir / "pretrain_16.pth").exists()
