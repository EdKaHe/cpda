# coding: utf-8

import numpy as np
import pandas as pd
from os.path import normpath
from time import time
from scipy.interpolate import interp1d
import glob
import re
import os
import json
import scipy.io as sio

from dipolar.disorder import Disorder, interp_alpha_1d
from dipolar.optics import Dipole, Ensemble, Simulator, polarizer, phase_plate, polarization

# define the path to the simulation configurations
configs_folder = normpath("./database/configurations/")
# define the path to the simulations
simulations_folder = normpath("./database/simulations/")

# get all files in the configurations folder
config_files = glob.glob(os.path.join(configs_folder, "*"))
config_files = [os.path.split(config_file)[-1] for config_file in config_files]

for config_file in config_files:       
    # get all files in the simulations folder
    simulation_files = glob.glob(os.path.join(simulations_folder, "*"))
    simulation_files = [os.path.split(simulation_file)[-1] for simulation_file in simulation_files]

    # construct the filename for the fourier and image space simulations
    filename_split = os.path.split(config_file)[-1]
    filename_split = os.path.splitext(filename_split)[0]
    filename_fourier = re.sub("config", "fourier", filename_split)
    filename_fourier = "{0}.mat".format(filename_fourier)
    filename_image = re.sub("config", "image", filename_split)
    filename_image = "{0}.mat".format(filename_image)
    
    if (filename_fourier not in simulation_files) & (filename_image not in simulation_files):
        # get the paths to the fourier and image plane data
        fourier_path = os.path.join(simulations_folder, filename_fourier)
        fourier_path = normpath(fourier_path)
        image_path = os.path.join(simulations_folder, filename_image)
        image_path = normpath(image_path)
        # save the simulated data
        sio.savemat(fourier_path, dict())
        sio.savemat(image_path, dict())
    
        # time the job duration
        start_time = time()
        # print some job details
        print("Simulating {0}".format(config_file))
        
        # get the path to the configuration file
        config_path = os.path.join(configs_folder, config_file)
        config_path = normpath(config_path)

        # load the current simulation configuration
        with open(config_path, "r") as file:
            config = json.load(file)

        # get the dipole parameters
        shape = config["dipole__shape"]
        material = config["dipole__material"]
        dim_x = config["dipole__dim_x"]
        dim_y = config["dipole__dim_y"]
        dim_z = config["dipole__dim_z"]

        # get the disorder parameters
        x = np.array(config["disorder__x"])
        y = np.array(config["disorder__y"])
        z = np.array(config["disorder__z"])
        phi = np.array(config["disorder__phi"])
        ds = np.array(config["disorder__ds"])


        # load the polarizabilities of the average structure
        filepath = normpath('./polarizability/{0}/{1}/{1}_l{2:.0f}nm_w{3:.0f}nm_h{4:.0f}nm.txt'.format(shape, material,
                                                                                            dim_x * 1e3, dim_y * 1e3, dim_z * 1e3)) # µm to nm
        alpha = pd.read_csv(filepath,
                        skiprows=3, header=None, delim_whitespace=True)
        alpha.columns = ['wavelength', 're_xx', 'im_xx', 're_yy', 'im_yy', 're_zz', 'im_zz']
        alpha_wavelength = alpha.wavelength / 1e3 # conversion from [nm] to [µm]
        alpha_xx = (alpha.re_xx + 1j * alpha.im_xx).values * 1e18 # conversion from [m^3] to [µm^3]
        alpha_yy = (alpha.re_yy + 1j * alpha.im_yy).values * 1e18 # conversion from [m^3] to [µm^3]
        alpha_zz = (alpha.re_zz + 1j * alpha.im_zz).values * 1e18 # conversion from [m^3] to [µm^3]

        # initialize an empty list that will contain interpolated polarizabilities and the according sizes
        interp_alpha = []
        interp_size = []
        # initialize the minimum and maximum wavelength as well as the minimum wavelength step
        wl_min = None
        wl_max = None
        dwl_min = None

        # get the path to all polarizabilities with the same shape and material
        filepath = normpath('./polarizability/{0}/{1}/*.txt'.format(shape, material))

        # get all txt files in this folder
        filepaths = glob.glob(filepath)

        # loop through all files and load all polarizabilities that only differ in size
        for filepath in filepaths:
            # split the name of the directory and the file
            path = os.path.dirname(filepath)
            file = os.path.basename(filepath)
            
            # check which files only differ in size
            if shape == "disk":
                match = re.findall("{0}_l([0-9]+)nm_w[0-9]+nm_h{1:.0f}nm.txt".format(material, dim_z * 1e3), file) # µm to nm
            if shape == "rod":
                match = re.findall("{0}_l([0-9]+)nm_w{1:.0f}nm_h{2:.0f}nm.txt".format(material, dim_y * 1e3, dim_z * 1e3), file) # µm to nm
                
            # load the polarizabilities that match the specified regular expression
            if match:
                # get the length related to the current polarizability
                interp_size.append(int(match[0]))
                
                # merge the path and matched filename
                matched_path = os.path.join(path, file)
                
                # load the polarizability
                alpha = pd.read_csv(filepath,
                                skiprows=3, header=None, delim_whitespace=True)
                alpha.columns = ['wavelength', 're_xx', 'im_xx', 're_yy', 'im_yy', 're_zz', 'im_zz']
                alpha_wavelength = alpha.wavelength / 1e3 # conversion from [nm] to [µm]
                alpha_xx = (alpha.re_xx + 1j * alpha.im_xx).values * 1e18 # conversion from [m^3] to [µm^3]
                
                # create a list of 1d interpolants
                interp_alpha.append(interp1d(alpha_wavelength, alpha_xx, kind="cubic", fill_value="extrapolate"))
                
                # get the smallest wavelength in all loaded polarizabilities
                if wl_min is None:
                    wl_min = np.min(alpha_wavelength)
                elif wl_min > np.min(alpha_wavelength):
                    wl_min = np.min(alpha_wavelength)
                # get the largest wavelength in all loaded polarizabilities            
                if wl_max is None:
                    wl_max = np.max(alpha_wavelength)
                elif wl_max < np.max(alpha_wavelength):
                    wl_max = np.ax(alpha_wavelength)
                # get the smallest wavelength step in all loaded polarizabilities
                if dwl_min is None:
                    dwl_min = np.min(np.diff(alpha_wavelength))
                elif dwl_min > np.min(np.diff(alpha_wavelength)):
                    dwl_min = np.min(np.diff(alpha_wavelength))

        # resample all polarizabilities to the same wavelength axis
        interp_wl = np.arange(wl_min, wl_max + dwl_min, dwl_min)
        interp_alpha = np.array([alpha(interp_wl) for alpha in interp_alpha])
        # convert the interpolation lengths to a numpy array
        interp_size = np.array(interp_size)

        # interpolate the polarizabilities according to the size disorder
        interp_alpha = interp_alpha_1d(interp_size, interp_alpha, (1 + ds) * dim_x * 1e3) # µm to nm

        # interpolate the mean polarizabilities
        alpha_xx = interp1d(alpha_wavelength, alpha_xx, kind="cubic", fill_value="extrapolate")(interp_wl)
        alpha_yy = interp1d(alpha_wavelength, alpha_yy, kind="cubic", fill_value="extrapolate")(interp_wl) 
        alpha_zz = interp1d(alpha_wavelength, alpha_zz, kind="cubic", fill_value="extrapolate")(interp_wl)

        # initialize a list of dipole instances
        if shape == "disk":
            dipoles = [Dipole(interp_wl, interp_alpha[ii], interp_alpha[ii], alpha_zz) for ii in range(len(x))]
        elif shape == "rod":
            dipoles = [Dipole(interp_wl, interp_alpha[ii], alpha_yy, alpha_zz) for ii in range(len(x))]

        # initialize an ensemble instance
        ensemble = Ensemble(dipoles)
        ensemble.x = x
        ensemble.y = y
        ensemble.z = z
        ensemble.phi = phi

        # get the simulation parameters
        wl_1 = config["simulation__wl_1"]
        wl_2 = config["simulation__wl_2"]
        wl_n = config["simulation__wl_n"]
        px_x = config["simulation__px_x"]
        px_y = config["simulation__px_y"]
        phi = config["simulation__phi"]
        theta = config["simulation__theta"]
        na = config["simulation__na"]
        eps = config["simulation__eps"]
        mu = config["simulation__mu"]
        delta_fourier = config["simulation__delta_fourier"]
        delta_image = config["simulation__delta_image"]

        # get the wavelength range that is to simulate
        wavelength = np.linspace(wl_1, wl_2, wl_n)

        # initialize the simulator instance
        simulator = Simulator(ensemble, wavelength, px_x=px_x, px_y=px_y, phi=phi, theta=theta, na=na,
                              eps=eps, mu=mu, delta_fourier=delta_fourier, delta_image=delta_image, dz=0.5)

        # perform some preliminary calculations
        simulator.initialize()

        # simulate the hyperspectral electric fields in the fourier and image plane
        fourier, image = simulator.reflection()

        # store the simulated fourier plane data as dict
        fourier = {
            'x': fourier.x,
            'y': fourier.y,
            'lambda_0': fourier.lambda_0,
            'E_xx': fourier.E_xx,
            'E_xy': fourier.E_xy,
            'E_yx': fourier.E_yx,
            'E_yy': fourier.E_yy,
        }
        # store the simulated image plane data as dict
        image = {
            'x': image.x,
            'y': image.y,
            'lambda_0': image.lambda_0,
            'E_xx': image.E_xx,
            'E_xy': image.E_xy,
            'E_yx': image.E_yx,
            'E_yy': image.E_yy,
        }

        # get the paths to the fourier and image plane data
        fourier_path = os.path.join(simulations_folder, filename_fourier)
        fourier_path = normpath(fourier_path)
        image_path = os.path.join(simulations_folder, filename_image)
        image_path = normpath(image_path)

        # save the simulated data
        sio.savemat(fourier_path, fourier)
        sio.savemat(image_path, image)
        
        # print some job details
        end_time = time()
        print("Job finished after {0:.02f}s\n".format(end_time - start_time))

