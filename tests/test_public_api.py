"""Check the comparator override and public entry-point wiring."""

from hematrace.pipeline import main
from hematrace.comparisons import deep_learning as dl
from hematrace.data.item_maps import TEMPORAL_FEATURES
from hematrace.config import cfg


def test_entrypoint_is_available():
    assert callable(main)
    assert len(TEMPORAL_FEATURES) == 35
    assert cfg.N_BINS == 6


def test_late_comparison_extensions_preserved():
    assert callable(dl._dl_train_one_model_original)
    assert callable(dl._run_deep_learning_model_comparison_original)
    assert dl._dl_train_one_model is not dl._dl_train_one_model_original
    assert dl.run_deep_learning_model_comparison is not dl._run_deep_learning_model_comparison_original
