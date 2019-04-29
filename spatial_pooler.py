import numpy as np
import sys, random, json



class SpatialPooler():


    def __init__(self,theta):
        self.current_collumn = 0
        self.theta = theta
        
        
    def activation(self,state):
        
    def permanence(self,state):
        return (self.collumns[self.current_collumn].permanences > self.theta) * state

    def potential_pool(self,state):
        return self.collumns[self.current_collumn].potential_pool * state

    def spatial_pooler(self,state):
        """
        state := np.array (bool)
        Spatial Pooler:
        Input -> potetial pool -> permanence -> activation -> output
        """
        return self.activation(self.permanence(self.potential_pool(state)))
