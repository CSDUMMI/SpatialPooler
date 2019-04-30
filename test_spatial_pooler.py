import numpy as np
from spatial_pooler import SpatialPooler

spatial_pooler_instance = SpatialPooler(7)

def test_init_collumn():
    test_cols = spatial_pooler_instance.init_collumn(500,0.75)

    assert type(test_cols[0]['permanences']) == type(np.array([0.5],dtype=np.float))
    assert np.sum(test_cols[0]['potential_pool'] > 0) > 0.6 # The potential pool should be at least 60 % of all collumns if not more


def test_activation():
    state = np.random.randn(7) > 0
    for i in range(1000):
        state_n = spatial_pooler_instance.activation(state)
        assert np.bool_ == type(state_n)

        mutate = np.random.randint(0,6)
        state[mutate] = not state[mutate]

def test_permanence():
    state = np.random.randn(7)  >  0
    for i in range(1000):
        state_n = spatial_pooler_instance.permanence(state)
        assert type(state_n) == type(state)
        assert state_n.shape == state.shape

        mutate = np.random.randint(0,6)
        state[mutate] = not state[mutate]


def test_potential_pool():
    state = np.random.randn(7) > 0
    for i in range(1000):
        state_n = spatial_pooler_instance.potential_pool(state)
        assert type(state_n) == type(np.array([True]))
        assert type(state) == type(state_n)
        assert state_n.shape == state.shape

        mutate = np.random.randint(0,6)
        state[mutate] = not state[mutate]
