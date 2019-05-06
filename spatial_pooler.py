#! /usr/bin/python3
import numpy as np
import sys, random, json



def overlap(x,y):
    # You can't overlap sdrs
    # with different shapes
    if x.shape != y.shape:
        return None
    result = np.zeros(x.shape,dtype=np.bool_)
    for i in range(x.shape[0]):
        result[i] = x[i] and y[i]
    return result


class SpatialPooler():

    def __init__(self,num_collumns=128,input_size=256,threshhold_permances=0.5,threshhold_activation=0.6,size_of_potential_pool=0.75):
        self.current_collumn = 0
        self.threshhold_permanences = threshhold_permances
        self.threshhold_activation = threshhold_activation
        self.collumns = self.init_collumn(num_collumns,input_size,1-size_of_potential_pool)


    def init_collumn(self,num_collumns,input_size,size_of_potential_pool=0.75):
        collumns = [{} for i in range(num_collumns)]
        for i in range(num_collumns):
            collumns[i]['potential_pool'] = np.random.rand(input_size) > size_of_potential_pool # Create a random potential pool with a certain percantage of potential connections
            collumns[i]['permanences'] = np.random.rand(np.sum(collumns[i]['potential_pool'])) # Random values for the permanence between  0-1
        return collumns


    def activation(self,state):
        """
        Sum state and then compare that to threshhold_activation
        """
        length = state.shape[0]
        print(np.sum(state) / length)
        return ((np.sum(state)/length) > self.threshhold_activation)

    def permanence(self,state):
        """
        For all permanences,filter the state for those that are higher than  self.theta
        """
        return (self.collumns[self.current_collumn]['permanences'] > self.threshhold_permanences) * state


    def potential_pool(self,state):
        predicat = self.collumns[self.current_collumn]['potential_pool']
        return np.extract(predicat,state)

    def spatial_pooler(self,state):
        """
        state := np.array (bool)
        Spatial Pooler:
        Input -> potetial pool -> permanence -> activation -> output
        """
        return self.activation(self.permanence(self.potential_pool(state)))

    def run(self,input_sdr):
        self.current_collumn = 0
        output = np.zeros(len(self.collumns),dtype=np.bool_)
        for i in range(len(self.collumns)):
            output[i] = self.spatial_pooler(input_sdr)
            self.current_collumn += 1
        self.current_collumn = 0
        return output
