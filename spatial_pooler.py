#! /usr/bin/python3
import numpy as np
import sys, random, json



class SpatialPooler():


    def __init__(self,num_collumns=128,input_size=256,threshhold_permances=0.5,threshhold_activation=0.6,size_of_potential_pool=0.75):
        self.current_collumn = 0
        self.threshhold_permanences = threshhold_permances
        self.threshhold_activation = threshhold_activation
        self.collumns = self.init_collumn(num_collumns,input_size,1-size_of_potential_pool)


    def init_collumn(self,num_collumns,input_size,size_of_potential_pool=0.75):
        collumns = [{} for i in range(num_collumns)]
        for i in range(num_collumns):
            collumns[i]['permanences'] = np.random.rand(input_size) # Random values for the permanence between  0-1
            collumns[i]['potential_pool'] = np.random.rand(input_size) > size_of_potential_pool # Create a random potential pool with a certain percantage of potential connections
        return collumns


    def activation(self,state):
        """
        Sum state and then compare that to threshhold_activation
        """
        length = state.shape[0]
        state_n = np.sum(state)/length > self.threshhold_activation
        print(type(state_n))
        return state_n

    def permanence(self,state):
        """
        For all permanences,filter the state for those that are higher than  self.theta
        """
        return (self.collumns[self.current_collumn]['permanences'] > self.threshhold_permanences) * state


    def potential_pool(self,state):
        return self.collumns[self.current_collumn]['potential_pool'] * state


    def spatial_pooler(self,state):
        """
        state := np.array (bool)
        Spatial Pooler:
        Input -> potetial pool -> permanence -> activation -> output
        """
        return self.activation(self.permanence(self.potential_pool(state)))
