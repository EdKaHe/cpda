#!/usr/bin/env python

"""

"""

from .dipole import Dipole

import numpy as np
from copy import deepcopy

 
class Ensemble:
    def __init__(self, dipoles, copy_dipoles=True):
        """ Ensemble class that contains all dipoles of the ensemble
            
            attributes: 
                dipoles (dipole or list): Dipole object or list of dipole objects
                    that is/are on the ensemble
                x (array): arrays with x-coordinates of each dipole [µm]
                y (array): arrays with y-coordinates of each dipole [µm]
                z (array): arrays with z-coordinates of each dipole [µm]
                phi (array): numpy array with orientations of each dipole in the xy
                    plane [deg]
                theta (array): numpy array with orientations of each dipole outside the
                    xy plane [deg]
        """
        
        # dipoles argument must be a Dipole object of a list of Dipole objects
        if isinstance(dipoles, list):
            for dipole in dipoles:
                if not isinstance(dipole, Dipole):
                    raise ValueError('Inappropriate type: {0} is given '
                                     + 'whereas class Dipole is expected' \
                                     .format(type(dipole)))
        elif isinstance(dipoles, Dipole):
            dipoles = [dipoles]
        else:
            raise ValueError('Inappropriate type: {0} is given '
                                     + 'whereas Dipole object or list of Dipole' 
                                     + 'is expected'.format(type(dipoles)))
            
        # construct the instance attributes
        if copy_dipoles:
            self.dipoles = deepcopy(dipoles)
        else:
            self.dipoles = dipoles
        
    # getter method to get all x-coordinates
    @property
    def x(self):
        return np.array([dipole.x for dipole in self.dipoles]).reshape((1, -1))

    # setter method to change the x-coordinates
    @x.setter
    def x(self, arr):
        # convert input array to numpy array
        arr = np.array(arr)
        # check if array has correct size
        if arr.size == len(self.dipoles):
            # flatten the array
            arr = arr.reshape((-1, ))
            for value, dipole in zip(arr, self.dipoles):
                dipole.x = value 
        else:
            raise ValueError('Number of entries of the input array must equal '
                             + 'number of dipoles')
   
    # getter method to get all y-coordinates
    @property
    def y(self):
        return np.array([dipole.y for dipole in self.dipoles]).reshape((1, -1))

    # setter method to change the y-coordinates
    @y.setter
    def y(self, arr):
        # convert input array to numpy array
        arr = np.array(arr)
        # check if array has correct size
        if arr.size == len(self.dipoles):
            # flatten the array
            arr = arr.reshape((-1, ))
            for value, dipole in zip(arr, self.dipoles):
                dipole.y = value 
        else:
            raise ValueError('Number of entries of the input array must equal '
                             + 'number of dipoles')
    
    # getter method to get all z-coordinates
    @property
    def z(self):
        return np.array([dipole.z for dipole in self.dipoles]).reshape((1, -1))

    # setter method to change the z-coordinates
    @z.setter
    def z(self, arr):
        # convert input array to numpy array
        arr = np.array(arr)
        # check if array has correct size
        if arr.size == len(self.dipoles):
            # flatten the array
            arr = arr.reshape((-1, ))
            for value, dipole in zip(arr, self.dipoles):
                dipole.z = value 
        else:
            raise ValueError('Number of entries of the input array must equal '
                             + 'number of dipoles')
            
    # getter method to get all phi orientations
    @property
    def phi(self):
        return np.array([dipole.phi for dipole in self.dipoles]).reshape((1, -1))

    # setter method to change the phi orientations
    @phi.setter
    def phi(self, arr):
        # convert input array to numpy array
        arr = np.array(arr)
        # check if array has correct size
        if arr.size == len(self.dipoles):
            # flatten the array
            arr = arr.reshape((-1, ))
            for value, dipole in zip(arr, self.dipoles):
                dipole.phi = value 
        else:
            raise ValueError('Number of entries of the input array must equal '
                             + 'number of dipoles')   
            
    # getter method to get all theta orientations
    @property
    def theta(self):
        return np.array([dipole.theta for dipole in self.dipoles]).reshape((1, -1))

    # setter method to change the theta orientations
    @theta.setter
    def theta(self, arr):
        # convert input array to numpy array
        arr = np.array(arr)
        # check if array has correct size
        if arr.size == len(self.dipoles):
            # flatten the array
            arr = arr.reshape((-1, ))
            for value, dipole in zip(arr, self.dipoles):
                dipole.theta = value 
        else:
            raise ValueError('Number of entries of the input array must equal '
                             + 'number of dipoles')