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


region = [-3e3, 2e3, -4e3, 5e3]
shape=[30, 30]
inc, dec = 75, 60
source_coords = vd.grid_coordinates(region=region, shape=shape, extra_coords=-800)
checker = vd.synthetic.CheckerBoard(1e9, region, w_east=2e3, w_north=4e3)
source_magnitudes = checker.predict(source_coords)
source_dipole_moments = hm.magnetic_angles_to_vec(source_magnitudes, inc, dec)
grid_coords = vd.grid_coordinates(region=region, shape=shape, extra_coords=500)
scatter_coords = vd.scatter_points(region=region, size=(shape[0]*shape[1]), extra_coords=500)

def test_on_regular_grid():
    be, bn, bu = hm.dipole_magnetic(grid_coords, source_coords, source_dipole_moments, field='b')
    tfa = hm.total_field_anomaly([be, bn, bu], inc, dec)

    eqs = hm.EquivalentSourcesTotalFieldAnomaly()
    eqs.fit(grid_coords, tfa, inc, dec)
    tfa_predict = eqs.predict(grid_coords, inc, dec)
    atol = vd.maxabs(tfa) / 1e5

    np.testing.assert_allclose(tfa, tfa_predict, atol=atol)

def test_on_scatter_grid():
    be, bn, bu = hm.dipole_magnetic(scatter_coords, source_coords, source_dipole_moments, field='b')
    tfa = hm.total_field_anomaly([be, bn, bu], inc, dec)

    eqs = hm.EquivalentSourcesTotalFieldAnomaly()
    eqs.fit(scatter_coords, tfa, inc, dec)
    tfa_predict = eqs.predict(scatter_coords, inc, dec)
    atol = vd.maxabs(tfa) / 1e3

    np.testing.assert_allclose(tfa, tfa_predict, atol=atol)

def test_fit_grid_predict_scatter():
    be, bn, bu = hm.dipole_magnetic(grid_coords, source_coords, source_dipole_moments, field='b')
    tfa = hm.total_field_anomaly([be, bn, bu], inc, dec)
    true_be, true_bn, true_bu = hm.dipole_magnetic(scatter_coords, source_coords, source_dipole_moments, field='b')
    true_tfa = hm.total_field_anomaly([true_be, true_bn, true_bu], inc, dec)
    
    eqs = hm.EquivalentSourcesTotalFieldAnomaly()
    eqs.fit(grid_coords, tfa, inc, dec)
    tfa_predict = eqs.predict(scatter_coords, inc, dec)
    rmse = np.sqrt(np.nanmean((true_tfa - tfa_predict)**2))
    tol = vd.maxabs(tfa)/1e3
    
    assert rmse <= tol