import numpy as np 

class GradObject:
    def __init__(self,value):
        self.value = value 
        self.grad = 0
    
    def __add__(self, other):
        pass 

