# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Equivalent sources for magnetic fields in Cartesian coordinates
"""
import numpy as np
import verde as vd
import verde.base as vdb
import choclo
from .. import magnetic_angles_to_vec
from numba import jit
from sklearn.utils.validation import check_is_fitted

class EquivalentSourcesMagnetic():
    """
    """
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
        tfa=None,
        be=None,
        bn=None,
        bu=None,
    ):
        coordinates, data, weights, inclination, declination = tfa
        coordinates, data, weights = vdb.check_fit_input(coordinates, data, weights)
        # Capture the data region to use as a default when gridding.
        self.region_ = vd.get_region(coordinates[:2])
        coordinates = vdb.n_1d_arrays(coordinates, 3)
        if self.dipole_coordinates is None:
            self.dipole_coordinates_ = tuple(
                p.astype(self.dtype) for p in self._build_points(coordinates)
            )
        else:
            self.depth_ = None  # set depth_ to None so we don't leave it unset
            self.dipole_coordinates_ = tuple(
                p.astype(self.dtype) for p in vdb.n_1d_arrays(self.dipole_coordinates, 3)
            )
        dipole_moment_direction = magnetic_angles_to_vec(
            1, self.dipole_inclination, self.dipole_declination,
        )
        field_direction = magnetic_angles_to_vec(1, inclination, declination)
        jacobian = self.jacobian_tfa(
            coordinates, self.dipole_coordinates_, dipole_moment_direction, field_direction,
        )
        moment_amplitude = vdb.least_squares(jacobian, data, weights, self.damping)
        self.dipole_moments_ = magnetic_angles_to_vec(
            moment_amplitude, self.dipole_inclination, self.dipole_declination,
        )
        return self
    
    def _build_dipoles(self, coordinates):
        """
        """
        if self.block_size is not None:
            reducer = vd.BlockReduce(
                spacing=self.block_size, reduction=np.median, drop_coords=False
            )
            # Must pass a dummy data array to BlockReduce.filter(), we choose an
            # array full of zeros. We will ignore the returned reduced dummy array.
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
        self, coordinates, dipole_coordinates, dipole_moment_direction, field_direction,
    ):
        