import math

import pytest

from trainer.trainer_utils import get_lr


def test_get_lr_cosine_schedule_endpoints():
    total, lr = 100, 1e-3
    assert get_lr(0, total, lr) == pytest.approx(lr, rel=1e-6)
    assert get_lr(total, total, lr) == pytest.approx(0.1 * lr, rel=1e-6)


def test_get_lr_cosine_schedule_midpoint():
    total, lr = 100, 1e-3
    assert get_lr(50, total, lr) == pytest.approx(0.55 * lr, rel=1e-6)
