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


# To do
# scattered coordinates, same scattered synthetic model
# Fit scatter grid, test on regular grid

def test_on_regular_grid():
    region = [-3e3, 2e3, -4e3, 5e3]
    shape=[30, 30]
    inc, dec = 75, 60
    source_coords = vd.grid_coordinates(region=region, shape=shape, extra_coords=-800)
    checker = vd.synthetic.CheckerBoard(1e9, region, w_east=2e3, w_north=4e3)
    source_magnitudes = checker.predict(source_coords)
    source_dipole_moments = hm.magnetic_angles_to_vec(source_magnitudes, inc, dec)

    data_coords = vd.grid_coordinates(region=region, shape=shape, extra_coords=500)
    be, bn, bu = hm.dipole_magnetic(data_coords, source_coords, source_dipole_moments, field='b')
    tfa = hm.total_field_anomaly([be, bn, bu], inc, dec)

    eqs = hm.EquivalentSourcesTotalFieldAnomaly()
    eqs.fit(data_coords, tfa, inc, dec)
    tfa_predict = eqs.predict(data_coords, inc, dec)
    atol = vd.maxabs(tfa) / 1e5

    np.testing.assert_allclose(tfa, tfa_predict, atol=atol)