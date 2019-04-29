import numpy as np
import sys, random, json

def activation(state):

def permanence(state):

def potential_pool(state):

def spatial_pooler(state,sp):
    """
    state := np.array (bool)
    Spatial Pooler:
    Input -> potetial pool -> permanence -> activation -> output
    """
    return activation(permanence(potential_pool(state)))
