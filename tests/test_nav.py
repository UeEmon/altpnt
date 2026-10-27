import numpy as np

from altpnt_sim.nav import InsBoundingEKF, gdop, rx_tdoa_jacobian, rx_tdoa_ranges, solve_rx_tdoa

ANCH = np.array([[-350e3, 0], [-300e3, 300e3], [-280e3, -320e3], [100e3, 450e3]])


def test_rx_tdoa_noise_free_recovers_position():
    rng = np.random.default_rng(0)
    p = rng.uniform(-50e3, 200e3, (20, 2))
    z = rx_tdoa_ranges(p, ANCH)
    est = solve_rx_tdoa(z, ANCH, p + rng.normal(0, 5e3, p.shape))
    assert np.max(np.abs(est - p)) < 1e-3


def test_jacobian_matches_finite_difference():
    p = np.array([[120e3, 40e3]])
    J = rx_tdoa_jacobian(p, ANCH)[0]
    eps = 1.0
    num = np.stack([(rx_tdoa_ranges(p + [eps, 0], ANCH) - rx_tdoa_ranges(p - [eps, 0], ANCH))[0] / 2,
                    (rx_tdoa_ranges(p + [0, eps], ANCH) - rx_tdoa_ranges(p - [0, eps], ANCH))[0] / 2], 1)
    assert np.allclose(J, num, atol=1e-6)


def test_gdop_reasonable():
    g = gdop(np.array([[150e3, 50e3]]), ANCH)[0]
    assert 1.0 < g < 10.0


def test_ekf_bounds_drift():
    rng = np.random.default_rng(1)
    dt, n = 1.0, 3000
    v = np.array([150.0, 0.0]); bias = np.array([0.4, -0.3])
    p = np.zeros(2)
    ekf = InsBoundingEKF.init(p, 50.0, 0.5)
    for k in range(n):
        p = p + v * dt
        ekf.predict(v + bias, dt)
        if k % 10 == 0:
            z = rx_tdoa_ranges(p, ANCH)[0] + rng.normal(0, 50.0, len(ANCH) - 1)
            ekf.update(z, lambda q: rx_tdoa_ranges(q, ANCH)[0], lambda q: rx_tdoa_jacobian(q, ANCH)[0],
                       np.eye(len(ANCH) - 1) * 50.0**2)
    assert np.linalg.norm(ekf.pos - p) < 300.0                 # INS 単独なら ~1.5 km
    assert np.allclose(ekf.x[2:], bias, atol=0.15)            # バイアス推定が収束
