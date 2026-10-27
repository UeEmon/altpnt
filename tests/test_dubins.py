import numpy as np

from altpnt_sim.dubins import PNTGridPlanner, dubins_all, dubins_shortest, hex_grid
from altpnt_sim.dubins.path import WORDS, DubinsPath
from altpnt_sim.dubins.planner import sample_trajectory


def _ang(a):
    return np.angle(np.exp(1j * a))


def test_all_words_reach_goal():
    rng = np.random.default_rng(0)
    for _ in range(500):
        q0 = np.r_[rng.uniform(-2e4, 2e4, 2), rng.uniform(0, 2 * np.pi)]
        q1 = np.r_[rng.uniform(-2e4, 2e4, 2), rng.uniform(0, 2 * np.pi)]
        segs = dubins_all(q0, q1, 3000.0)
        for k, w in enumerate(WORDS):
            if np.isnan(segs[k]).any():
                continue
            e = DubinsPath(q0, 3000.0, w, segs[k]).at(np.array([1e12]))[0]
            assert np.hypot(*(e[:2] - q1[:2])) < 1e-3
            assert abs(_ang(e[2] - q1[2])) < 1e-6


def test_straight_line_is_shortest():
    p = dubins_shortest([0, 0, 0], [10000, 0, 0], 1000.0)
    assert p.length == np.float64(10000.0) or abs(p.length - 10000.0) < 1e-6


def test_planner_and_continuous_sampling():
    pts = hex_grid([50e3, 0], 7, 15e3)
    plan = PNTGridPlanner(rho=4000.0).plan([0, 0, 0], pts)
    assert sorted(plan.order.tolist()) == list(range(7))
    tr = sample_trajectory(plan.paths, 150.0, 1.0, orbit=(plan.orbit_center, plan.orbit_radius, 600))
    step = np.linalg.norm(np.diff(tr[:, 1:3], axis=0), axis=1)
    assert np.allclose(step, 150.0, atol=0.1)          # 継ぎ目でも等速（INS に人工誤差を入れない）


def test_hex_grid_bounds():
    for n in range(5, 10):
        assert hex_grid([0, 0], n, 1e4).shape == (n, 2)
