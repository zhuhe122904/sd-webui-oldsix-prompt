# auto-generated test
import os
import runpy

MODULE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "pareto_qp.py")
pareto_qp_ns = runpy.run_path(MODULE_PATH)
ParetoQP = pareto_qp_ns["ParetoQP"]
minimum_norm_combination = pareto_qp_ns["minimum_norm_combination"]


def almost_equal(seq, expected, tol=1e-6):
    return all(abs(a - b) <= tol for a, b in zip(seq, expected))


def test_two_task_solution_is_balanced():
    solver = ParetoQP()
    g1 = [1.0, 0.0]
    g2 = [0.0, 1.0]
    weights = solver.solve([g1, g2])
    assert almost_equal(weights, [0.5, 0.5])


def test_three_task_closed_form_prefers_sparse_solution():
    solver = ParetoQP()
    g1 = [1.0, 0.0]
    g2 = [0.5, 0.5]
    g3 = [0.0, 1.0]
    weights = solver.solve([g1, g2, g3])
    assert len(weights) == 3
    assert abs(sum(weights) - 1.0) < 1e-6
    assert weights[1] < 0.5


def test_degenerate_gradients_fall_back_to_uniform():
    solver = ParetoQP()
    g1 = [1.0, 0.0]
    g2 = [1.0, 0.0]
    weights = solver.solve([g1, g2])
    assert almost_equal(weights, [0.5, 0.5])
    assert solver.stats.fallback_used


def test_zero_gradient_tasks_are_ignored():
    solver = ParetoQP()
    g1 = [1.0, 0.0]
    g2 = [0.0, 0.0]
    weights = solver.solve([g1, g2])
    assert almost_equal(weights, [1.0, 0.0])
    assert not solver.stats.fallback_used


def test_minimum_norm_combination_returns_combined_gradient():
    g1 = [2.0, 0.0]
    g2 = [0.0, 2.0]
    combined, weights = minimum_norm_combination([g1, g2])
    assert almost_equal(weights, [0.5, 0.5])
    assert almost_equal(combined, [1.0, 1.0])
