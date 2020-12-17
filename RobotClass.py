import numpy as np


class kinematic:
    """[summary]
    """
    def __init__(self, par):
        # Denavit-Hartenberg-Parameter
        self.theta = np.array([ 0, 0, -90, 180, 180, 0, -90])*np.pi/180
        self.d =     np.array([ par['l11'], 0, 0, 0,  par['l4'], 0, par['l6']])
        self.a =     np.array([ 0, par['l12'], par['l2'], -par['l3'], 0, 0, 0])
        self.alpha = np.array([ 0, -90, 0, 90, 90, -90, 0])*np.pi/180

        print('\n')
        print('robot kinematics initialized')

class Robot:
    def __init__(self):
        pass