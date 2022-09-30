#!/usr/bin/env python

"""

"""

import numpy as np
from scipy.interpolate import pchip


class Dipole:
    def __init__(self, lambda_, alpha_xx, alpha_yy, alpha_zz, x=0, y=0, z=0, phi=0, theta=0):
        """ Single dipole with coordinates (x, y, z) and polarizability 
        (alpha_xx, alpha_zz, alpha_zz)
            
            attributes: 
                lambda_ (array): wavelength vector of the polarizability [µm]
                alpha_xx (array): polarizability in xx-direction [µm ** 3]
                alpha_yy (array): polarizability in yy-direction [µm ** 3]
                alpha_zz (array): polarizability in zz-direction [µm ** 3]
                x (float): x-coordinate of the dipole [µm]
                y (float): y-coordinate of the dipole [µm]
                z (float): z-coordinate of the dipole [µm]
                phi (float): rotation around z in [deg]
                theta (float): rotation around x/y in [deg]
                n1 (array): first read-only orthonormal vector
                n2 (array) : second read-only orthonormal vector
                n3 (array): third read-only orthonormal vector
        """
        # convert inputs to numpy arrays
        lambda_ = np.array(lambda_)
        alpha_xx = np.array(alpha_xx)
        alpha_yy = np.array(alpha_yy)
        alpha_zz = np.array(alpha_zz)

        # construct the instance attributes
        self.x = x
        self.y = y
        self.z = z
        self.phi = phi
        self.theta = theta
        self.lambda_ = np.array(lambda_).reshape((-1,))
        self.alpha_xx = pchip(lambda_.reshape((-1,)), alpha_xx.reshape((-1,)), extrapolate=True)
        self.alpha_yy = pchip(lambda_.reshape((-1,)), alpha_yy.reshape((-1,)), extrapolate=True)
        self.alpha_zz = pchip(lambda_.reshape((-1,)), alpha_zz.reshape((-1,)), extrapolate=True)

    # get the first orthonormal vector of the dipole
    @property
    def n1(self):
        phi = np.deg2rad(self.phi)
        theta = np.deg2rad(self.theta)
        n1 = np.array([[np.cos(theta) * np.cos(phi)],
                       [np.cos(theta) * np.sin(phi)],
                       [np.sin(theta)]])        
        return n1

    # get the first orthonormal vector of the dipole    
    @property
    def n2(self):
        phi = np.deg2rad(self.phi)      
        n2 = np.array([[-np.sin(phi)],
                       [np.cos(phi)],
                       [0]])
        return n2
    
    # get the first orthonormal vector of the dipole
    @property
    def n3(self):
        phi = np.deg2rad(self.phi)
        theta = np.deg2rad(self.theta)
        n3 = np.array([[-np.sin(theta) * np.cos(phi)],
                      [-np.sin(theta) * np.sin(phi)],
                      [np.cos(theta)]])
        return n3