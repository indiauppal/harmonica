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