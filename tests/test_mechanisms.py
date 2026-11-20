import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import cdist

import metric_private_matching as mpm
from metric_private_matching import mechanisms as api


@pytest.mark.parametrize('mechanism', mpm.MECHANISMS)
@pytest.mark.parametrize('cost', ['euclidean', 'squared'])
def test_valid_matching_reproducible_without_mutation(mechanism, cost):
    rng = np.random.default_rng(5)
    x, y = rng.normal(size=(12, 4)), rng.normal(size=(12, 4))
    saved = x.copy(), y.copy()
    first = mpm.match(x, y, 3., mechanism=mechanism, cost=cost, rng=80)
    second = mpm.match(x, y, 3., mechanism=mechanism, cost=cost, rng=80)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.sort(first), np.arange(len(x)))
    np.testing.assert_array_equal(x, saved[0])
    np.testing.assert_array_equal(y, saved[1])


@pytest.mark.parametrize('epsilon', [0, -1, np.inf, np.nan])
@pytest.mark.parametrize('mechanism', mpm.MECHANISMS)
def test_invalid_budget(epsilon, mechanism):
    with pytest.raises(ValueError, match='epsilon'):
        mpm.match([[0.]], [[1.]], epsilon, mechanism=mechanism)


@pytest.mark.parametrize('x,y', [([], []), ([[np.nan]], [[0.]]), ([[1., 2.]], [[1.]]),
                                ([[1.],[2.]], [[1.]]), ([1.,2.], [3.,4.])])
def test_invalid_input(x, y):
    with pytest.raises(ValueError):
        mpm.match(x, y, 1.)


def test_options():
    with pytest.raises(ValueError, match='mechanism'):
        mpm.match([[1.]], [[2.]], 1., mechanism='unknown')
    with pytest.raises(ValueError, match='cost'):
        mpm.match([[1.]], [[2.]], 1., cost='unknown')
    with pytest.raises(ValueError, match='square'):
        mpm.matching_from_costs(np.ones((2, 3)))


def test_radial_calibration():
    noise = mpm.privatize_points(np.zeros((20000, 3)), 2., rng=27)
    # E||Z|| = d/epsilon, not coordinatewise Laplace calibration.
    assert abs(np.linalg.norm(noise, axis=1).mean() - 1.5) < .025
    assert np.max(np.abs(noise.mean(axis=0))) < .025


def test_projection_preserves_cross_distances_and_squared_assignment():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(9, 5))
    y = np.zeros((9, 5)); y[:, 0] = np.arange(9)
    report = mpm.privatize_projected(x, y, 1e100, rng=11)
    assert report.noise_dimension == 2
    np.testing.assert_allclose(cdist(report.values, report.public_points), cdist(x, y), atol=2e-14)
    report2 = mpm.privatize_projected(x, y, 1e100, cost='squared', rng=11)
    assert report2.noise_dimension == 1
    np.testing.assert_array_equal(report2.matching(cost='squared'), mpm.matching_from_points(x, y, cost='squared'))
    np.testing.assert_array_equal(report2.matching(), report2.matching(cost='squared'))
    with pytest.raises(ValueError, match='objective'):
        report2.matching(cost='euclidean')


@pytest.mark.parametrize('mechanism', mpm.MECHANISMS)
@pytest.mark.parametrize('cost', ['euclidean', 'squared'])
def test_coincident_candidates(mechanism, cost):
    x = np.arange(15).reshape(5,3)
    y = np.ones((5,3))
    result = mpm.match(x, y, 1., mechanism=mechanism, cost=cost, rng=6)
    np.testing.assert_array_equal(np.sort(result), np.arange(5))


def test_frozen_calibration_parity():
    root = Path(mpm.__file__).parent/'reproduction'
    sys.path.insert(0, str(root/'frozen/euclidean'))
    from private_matching.integral_mechanisms import Geometry, solve, supnorm_unit
    from private_matching.mechanisms import sample_euclidean_laplace
    rng = np.random.default_rng(12)
    x = rng.normal(size=(10, 5))
    y = np.zeros((10,5)); y[:,:2] = rng.normal(size=(10,2))
    geo = Geometry.build(x, y)
    for cost, p in [('euclidean', 1), ('squared', 2)]:
        for arm in mpm.MECHANISMS:
            gen = np.random.default_rng(103)
            epsilon = 3.
            base = geo.C1 if p == 1 else geo.C2
            if arm == 'original_points':
                z = sample_euclidean_laplace(10,5,epsilon,gen)[0]
                scores = cdist(x+z, y, metric='euclidean' if p == 1 else 'sqeuclidean')
            elif arm == 'projected_representation':
                u, v = (geo.fold_X,geo.fold_Y) if p == 1 else (geo.U,geo.V)
                z = sample_euclidean_laplace(10,u.shape[1],epsilon,gen)[0]
                scores = cdist(u+z, v, metric='euclidean' if p == 1 else 'sqeuclidean')
            elif arm == 'distance_vectors':
                scores = base + (1. if p == 1 else 2*geo.radius)/epsilon*supnorm_unit(10,gen)
            else:
                scores = base - (2. if p == 1 else 2*geo.diameter)/epsilon*gen.gumbel(size=(10,10))
            np.testing.assert_array_equal(
                mpm.match(x,y,epsilon,mechanism=arm,cost=cost,rng=103), solve(scores))


def test_original_endpoint_reuses_same_noise():
    gen = np.random.default_rng(1)
    x, y = gen.normal(size=(9,2)), gen.normal(size=(9,2))
    for cost in ('euclidean', 'squared'):
        np.testing.assert_array_equal(
            mpm.original_points(x,y,2.,cost=cost,rng=17),
            mpm.projected_representation(x,y,2.,cost=cost,rng=17))
