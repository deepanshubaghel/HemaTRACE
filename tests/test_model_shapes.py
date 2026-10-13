"""Smoke-test the unchanged research model forward signature and tensor shapes."""

import torch

from hematrace.data.item_maps import TEMPORAL_FEATURES
from hematrace.models.temporal import HemaTRACEModel
from hematrace.config import cfg


def test_main_temporal_forward_shapes():
    torch.manual_seed(42)
    model = HemaTRACEModel(n_temp=len(TEMPORAL_FEATURES), n_static=5)
    model.eval()
    batch, steps, features = 3, cfg.N_BINS, len(TEMPORAL_FEATURES)
    v = torch.zeros(batch, steps, features)
    static = torch.zeros(batch, 5)
    with torch.no_grad():
        out = model(v, v, v, v, static, static)
    assert out["delta_z"].shape == (batch,)
    assert out["severe_logit"].shape == (batch,)
    assert out["onset_logit"].shape == (batch,)
    assert out["attention"].shape == (batch, steps)
    torch.testing.assert_close(out["attention"].sum(1), torch.ones(batch))
