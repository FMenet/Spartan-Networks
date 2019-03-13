import re
import numpy as np


class NDCBDataGenerator:
    """
    A data generator for the n-dimensional checkerboard synthetic dataset
    :param n : number of dimensions
    :param length: modulo to add
    """
    
    def __init__(self, n, length):
        self.number_of_dimensions = n
        self.length = length
        print("N-checkerboard data generator declared !")
        
        return
    
    def __init__(self):
        self.number_of_dimensions = 2
        self.length = 1.0
        print("Default N-checkerboard declared !")
    
    def __str__(self):
        return "n-dimensional checkerboard, \n%s dimensions; length-of-square %s \nObject Unique Tag :\n %s" \
               % (self.number_of_dimensions, self.length, self.__repr__())
    
    def get_target(self, input_coord):
        """
        Given a list of coordinates, give back the target truth.
        :param input_coord: coordinates
        :return: an integer, 0 or 1
        """
        
        if len(input_coord) != self.number_of_dimensions:
            raise ValueError("Nombre de dimensions invalide !")
        
        # Temporary code, we need to make it faster by using the LSB
        valuesum = int(0)
        for i in input_coord:
            valuesum += int(i / self.length)
        
        if valuesum % 2:
            return 1.0
        else:
            return 0.0
    
    def get_target_onehot(self, input_coord):
        if self.get_target(input_coord):
            return [0, 1]
        else:
            return [1, 0]
