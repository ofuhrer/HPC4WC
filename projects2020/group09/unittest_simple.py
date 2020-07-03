"""
tests whether produced dataset is the same as baseline

    @author: verena bessenbacher
    @date: 03 07 2020
"""

import numpy as np

def test_simple(mod, obs):

    assert mod.shape == obs.shape, 'shapes are not the same'
    
    assert np.isclose(np.nan_to_num(mod)[:,2:-3,2:-3,2:-3], np.nan_to_num(obs)[:,2:-3,2:-3,2:-3], equal_nan=True).all(), 'non-nan values outside boundaries are not the same within tolerance'

    assert np.isclose(np.nan_to_num(mod), np.nan_to_num(obs), equal_nan=True).all(), 'non-nan values are not the same within tolerance'

    assert np.isclose(mod[:,2:-3,2:-3,2:-3], obs[:,2:-3,2:-3,2:-3], equal_nan=True).all(), 'values outside boundaries are not the same within tolerance'

    assert np.isclose(mod, obs, equal_nan=True).all(), 'values are not the same within tolerance'

    print('unittest passed')
    # TODO make special case for boundaries
