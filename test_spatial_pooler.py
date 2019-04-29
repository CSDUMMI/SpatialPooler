import numpy as np
from spatial_pooler import SpatialPooler

spatial_pooler_instance = SpatialPooler(5)

def test_init_collumn():
    test_cols = spatial_pooler_instance.init_collumn(500,0.75)

    assert type(test_cols[0]['permanences']) == type(np.array([0.5],dtype=np.float))
    assert np.sum(test_cols[0]['potential_pool'] > 0) > 0.6 # The potential pool should be at least 60 % of all collumns if not more
