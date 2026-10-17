import numpy as np

from research.imitation_learning.feature_shift import roc_auc


def test_auc_is_one_for_a_perfect_ranking_and_a_half_for_ties():
    positive = np.array([False, False, True, True])
    assert roc_auc(np.array([1.0, 2.0, 3.0, 4.0]), positive) == 1.0
    assert roc_auc(np.array([4.0, 3.0, 2.0, 1.0]), positive) == 0.0
    assert roc_auc(np.ones(4), np.array([False, True, False, True])) == 0.5
