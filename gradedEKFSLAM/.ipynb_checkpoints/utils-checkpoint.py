import numpy as np

def wrapToPi(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

def rotmat2d(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle),  np.cos(angle)]])

def polar2cartesian(range, angle):
    return range*np.cos(angle), range*np.sin(angle - np.pi)
