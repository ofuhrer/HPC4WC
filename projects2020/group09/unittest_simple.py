"""
tests whether produced dataset is the same as baseline

    @author: verena bessenbacher
    @date: 03 07 2020
"""

import numpy as np

def test_simple(mod, obs):

    assert mod.shape == obs.shape, 'shapes are not the same'

    assert np.isclose(mod, obs, equal_nan=True).all(), 'values are not the same within tolerance'

    print('unittest passed')
    # TODO make special case for boundaries
