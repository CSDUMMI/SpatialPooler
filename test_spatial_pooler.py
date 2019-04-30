import numpy as np
from spatial_pooler import SpatialPooler

spatial_pooler_instance = SpatialPooler(7)

def test_init_collumn():
    test_cols = spatial_pooler_instance.init_collumn(500,0.75)

    assert type(test_cols[0]['permanences']) == type(np.array([0.5],dtype=np.float))
    assert np.sum(test_cols[0]['potential_pool'] > 0) > 0.6 # The potential pool should be at least 60 % of all collumns if not more


def test_activation():
    state = np.random.randn(7) > 0
    for i in range(10000): # 100 different test  cases
        assert spatial_pooler_instance.activation(state) == (np.sum(state)/state.shape[0] > spatial_pooler_instance.threshhold_activation)

        elem_to_mutate = np.random.randint(0,7)
        state[elem_to_mutate] = not state[elem_to_mutate] # mutation

def test_permanence():
    state = np.random.randn(7)  >  0
    for i in range(10000):
        state_n = spatial_pooler_instance.permanence(state)
        assert np.array_equal(state_n,(spatial_pooler_instance.collumns[spatial_pooler_instance.current_collumn]['permanences'] > spatial_pooler_instance.threshhold_permanences) * state)

        elem_to_mutate = np.random.randint(0,7)
        state[elem_to_mutate] = not state[elem_to_mutate]

        assert type(state_n) == type(state)
        assert state_n.shape == state.shape


def test_potential_pool():
    state = np.random.randn(7) > 0
    state_n = spatial_pooler_instance.potential_pool(state)
    assert type(state_n) == type(np.array([True]))
    assert state_n.shape == state.shape
