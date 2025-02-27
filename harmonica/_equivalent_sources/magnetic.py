# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Equivalent sources for magnetic fields in Cartesian coordinates
"""
import choclo
import numba
import numpy as np
import verde as vd
import verde.base as vdb
from sklearn.utils.validation import check_is_fitted

from .. import dipole_magnetic, magnetic_angles_to_vec, total_field_anomaly

TESLA_TO_NANOTESLA = 1e9


class EquivalentSourcesTotalFieldAnomaly:
    """ """

    def __init__(
        self,
        damping=None,
        depth: float | str = "default",
        dipole_coordinates=None,
        dipole_inclination=90,
        dipole_declination=0,
        block_size=None,
        parallel=True,
        dtype=np.float64,
    ):
        if isinstance(depth, str) and depth != "default":
            raise ValueError(
                f"Found invalid 'depth' value equal to '{depth}'. "
                "It should be 'default' or a numeric value."
            )
        if depth == 0:
            raise ValueError(
                "Depth value cannot be zero. It should be a non-zero numeric value."
            )

        self.damping = damping
        self.depth = depth
        self.dipole_coordinates = dipole_coordinates
        self.dipole_inclination = dipole_inclination
        self.dipole_declination = dipole_declination
        self.block_size = block_size
        self.parallel = parallel
        self.dtype = dtype

    def fit(
        self,
        coordinates,
        data,
        inclination,
        declination,
        weights=None,
    ):
        coordinates, data, weights = vdb.check_fit_input(coordinates, data, weights)
        # Capture the data region to use as a default when gridding.
        self.region_ = vd.get_region(coordinates[:2])
        coordinates = vdb.n_1d_arrays(coordinates, 3)
        if self.dipole_coordinates is None:
            self.dipole_coordinates_ = tuple(
                p.astype(self.dtype) for p in self._build_dipoles(coordinates)
            )
        else:
            self.depth_ = None  # set depth_ to None so we don't leave it unset
            self.dipole_coordinates_ = tuple(
                p.astype(self.dtype)
                for p in vdb.n_1d_arrays(self.dipole_coordinates, 3)
            )
        dipole_moment_direction = magnetic_angles_to_vec(
            1,
            self.dipole_inclination,
            self.dipole_declination,
        )
        dipole_moment_direction = _field_direction_as_array(
            dipole_moment_direction, self.dipole_coordinates_[0].size
        )
        field_direction = magnetic_angles_to_vec(1, inclination, declination)
        field_direction = _field_direction_as_array(field_direction, data.size)
        jacobian = self.jacobian(
            coordinates,
            self.dipole_coordinates_,
            dipole_moment_direction,
            field_direction,
        )
        moment_amplitude = vdb.least_squares(jacobian, data, weights, self.damping)
        self.dipole_moments_ = magnetic_angles_to_vec(
            moment_amplitude,
            self.dipole_inclination,
            self.dipole_declination,
        )
        return self

    def _build_dipoles(self, coordinates):
        """ """
        if self.block_size is not None:
            reducer = vd.BlockReduce(
                spacing=self.block_size, reduction=np.median, drop_coords=False
            )
            # Must pass a dummy data array to BlockReduce.filter(),
            # we choose an array full of zeros. We will ignore the
            # returned reduced dummy array.
            coordinates, _ = reducer.filter(coordinates, np.zeros_like(coordinates[0]))
        if self.depth == "default":
            self.depth_ = 4.5 * np.mean(vd.median_distance(coordinates, k_nearest=1))
        else:
            self.depth_ = self.depth
        return (
            coordinates[0],
            coordinates[1],
            coordinates[2] - self.depth_,
        )

    def jacobian(
        self,
        coordinates,
        dipole_coordinates,
        dipole_moment_direction,
        field_direction,
    ):
        """ """
        n = len(coordinates[0])
        m = len(dipole_coordinates[0])
        jacobian = np.empty((n, m))
        _jacobian_fast(
            easting=coordinates[0],
            northing=coordinates[1],
            upward=coordinates[2],
            d_easting=dipole_coordinates[0],
            d_northing=dipole_coordinates[1],
            d_upward=dipole_coordinates[2],
            m_easting=dipole_moment_direction[0],
            m_northing=dipole_moment_direction[1],
            m_upward=dipole_moment_direction[2],
            f_easting=field_direction[0],
            f_northing=field_direction[1],
            f_upward=field_direction[2],
            jacobian=jacobian,
        )
        return jacobian

    def predict_magnetic_field(self, coordinates):
        """ """
        # We know the gridder has been fitted if it has the
        # estimated parameters.
        check_is_fitted(
            self,
            ["dipole_moments_", "dipole_coordinates_"],
        )
        return dipole_magnetic(
            coordinates, self.dipole_coordinates_, self.dipole_moments_, "b"
        )

    def predict(self, coordinates, inclination, declination):
        """ """
        b_field = self.predict_magnetic_field(coordinates)
        return total_field_anomaly(b_field, inclination, declination)


@numba.jit(nopython=True, parallel=True)
def _jacobian_fast(  # noqa: CFQ002
    easting,
    northing,
    upward,
    d_easting,
    d_northing,
    d_upward,
    m_easting,
    m_northing,
    m_upward,
    f_easting,
    f_northing,
    f_upward,
    jacobian,
):
    for i in numba.prange(easting.size):
        for j in range(d_easting.size):
            b_easting, b_northing, b_upward = choclo.dipole.magnetic_field(
                easting_p=easting[i],
                northing_p=northing[i],
                upward_p=upward[i],
                easting_q=d_easting[j],
                northing_q=d_northing[j],
                upward_q=d_upward[j],
                magnetic_moment_east=m_easting[j],
                magnetic_moment_north=m_northing[j],
                magnetic_moment_up=m_upward[j],
            )
            jacobian[i, j] = TESLA_TO_NANOTESLA * (
                b_easting * f_easting[i]
                + b_northing * f_northing[i]
                + b_upward * f_upward[i]
            )


def _field_direction_as_array(field_direction, size):
    """"""
    ""
    if isinstance(field_direction[0], np.ndarray):
        if field_direction[0].size != size:
            raise ValueError(
                "Inclination and declination must have the same size as the data "
                f"({size}) or be a float."
            )
        return field_direction
    field_direction = tuple(np.full(size, c) for c in field_direction)
    return field_direction
