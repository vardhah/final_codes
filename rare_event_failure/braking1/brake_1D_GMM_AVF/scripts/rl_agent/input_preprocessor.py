import numpy as np

class InputPreprocessor():
    def __call__(self, state):
        s = np.hstack((state[0]/120.0, state[1]/45.0,state[2]))
        return s