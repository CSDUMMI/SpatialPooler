import numpy as np
from spatial_pooler import SpatialPooler

spatial_pooler_instance = SpatialPooler(num_collumns = 7,input_size=8)

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

def test_overlap():
    x = np.array([True,True,False,False,True])
    y = np.array([True,True,False,True,False])
    z = overlap(x,y)
    assert z == np.array([True,True,False,False,False])

def test_run():
    input_sp = np.random.randn(8) > 0
    output_sp = spatial_pooler_instance.run(input_sp)

    # Has the same shape as desired
    assert output_sp.shape[0] == 7

    # Less than 5% of the output_sp is on.
    # Sparsity
    assert np.sum(output_sp)/output_sp.shape[0] < 0.05

    # Maintain overlap properties
    # Overlap count of input_sp with input_sp_1
    # should be similar to the overlap count
    # output_sp output_sp_1
    input_sp_1  = np.random.randn(8) > 0
    output_sp_1 = spatial_pooler_instance.run(input_sp_1)
    overlap_input = overlap(input_sp_1,input_sp))
    overlap_output = overlap(output_sp_1,output_sp))

    # Should be at least to 75%
    # the same
    assert (np.sum(overlap_input) / overlap_input.shape[0]) > 0.75
    assert (np.sum(overlap_output) / overlap_output.shape[0]) > 0.75
