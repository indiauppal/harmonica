# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test the EquivalentSourcesMagnetic
"""
import numpy as np
import verde as vd
import harmonica as hm
import pytest

@pytest.fixture
def region():
    """
    Return a sample region
    """
    return [-3e3, 2e3, -4e3, 5e3]

@pytest.fixture
def shape():
    return [30, 30]

@pytest.fixture
def inc_dec():
    return 75, 60

@pytest.fixture
def dipoles(region, shape, inc_dec):
    dipole_coords = vd.grid_coordinates(region=region, shape=shape, extra_coords=-800)
    checker = vd.synthetic.CheckerBoard(1e9, region, w_east=2e3, w_north=4e3)
    dipole_magnitudes = checker.predict(dipole_coords)
    inc, dec = inc_dec
    dipole_moments = hm.magnetic_angles_to_vec(dipole_magnitudes, inc, dec)
    return dipole_coords, dipole_moments

@pytest.fixture
def grid_coords(region, shape):
    grid_coords = vd.grid_coordinates(region=region, shape=shape, extra_coords=500)
    return grid_coords

@pytest.fixture
def scatter_coords(region, shape):
    scatter_coords = vd.scatter_points(region=region, size=(shape[0]*shape[1]), extra_coords=500)
    return scatter_coords
    
@pytest.fixture
def grid_tfa(grid_coords, dipoles, inc_dec):
    dipole_coords, dipole_moments = dipoles
    inc, dec = inc_dec
    tfa = forward_tfa(grid_coords, dipole_coords, dipole_moments, inc, dec)
    rng = np.random.default_rng(seed=0)
    noise = rng.normal(0, (vd.maxabs(tfa)/1e2), size=np.shape(tfa))
    tfa += noise
    return tfa

@pytest.fixture
def scatter_tfa(scatter_coords, dipoles, inc_dec):
    dipole_coords, dipole_moments = dipoles
    inc, dec = inc_dec
    tfa = forward_tfa(scatter_coords, dipole_coords, dipole_moments, inc, dec)
    rng = np.random.default_rng(seed=0)
    noise = rng.normal(0, (vd.maxabs(tfa)/1e2), size=np.shape(tfa))
    tfa += noise
    return tfa

def forward_tfa(coords, dipole_coords, dipole_moments, inc, dec):
    be, bn, bu = hm.dipole_magnetic(coords, dipole_coords, dipole_moments, field='b')
    tfa = hm.total_field_anomaly([be, bn, bu], inc, dec)
    return tfa
    
def test_on_regular_grid(grid_tfa, grid_coords, inc_dec):
    inc, dec = inc_dec
    eqs = hm.EquivalentSourcesTotalFieldAnomaly()
    eqs.fit(grid_coords, grid_tfa, inc, dec)
    tfa_predict = eqs.predict(grid_coords, inc, dec)
    atol = vd.maxabs(grid_tfa) / 1e2
    np.testing.assert_allclose(grid_tfa, tfa_predict, atol=atol)

def test_on_scatter_grid(scatter_tfa, scatter_coords, inc_dec):
    inc, dec = inc_dec
    eqs = hm.EquivalentSourcesTotalFieldAnomaly()
    eqs.fit(scatter_coords, scatter_tfa, inc, dec)
    tfa_predict = eqs.predict(scatter_coords, inc, dec)
    atol = vd.maxabs(scatter_tfa) / 1e2
    np.testing.assert_allclose(scatter_tfa, tfa_predict, atol=atol)

def test_fit_grid_predict_scatter(grid_tfa, grid_coords, scatter_coords, dipoles, inc_dec):
    dipole_coords, dipole_moments = dipoles
    inc, dec = inc_dec
    eqs = hm.EquivalentSourcesTotalFieldAnomaly()
    eqs.fit(grid_coords, grid_tfa, inc, dec)
    tfa_predict = eqs.predict(scatter_coords, inc, dec)
    true_tfa = forward_tfa(scatter_coords, dipole_coords, dipole_moments, inc, dec)
    rmse = np.sqrt(np.nanmean((true_tfa - tfa_predict)**2))
    tol = vd.maxabs(true_tfa)/1e2
    assert rmse <= tol