from pydefix import *
import utils
import numpy as np
import inifix
import matplotlib.pyplot as plt

def initflow(data): 
    params = input_parameters
    nSpecies = int(params['Dust']['nSpecies'][0])
    density = float(params['Units']['density'][0])
    length = float(params['Units']['length'][0])
    rho0 = float(params['Setup']['rho0'][0])/density
    
    massmin = float(params['Coala']['massmin'][0])/(density*length**3)
    massmax = float(params['Coala']['massmax'][0])/(density*length**3)
   
    #CI for the gas
    data.Vc[RHO,:,:,:] = rho0
    data.Vc[VX1,:,:,:] = 0.0
    data.Vc[VX2,:,:,:] = 0.0
    data.Vc[VX3,:,:,:] = 0.0
    
    #CI for the dust
    if nSpecies is not None:
        #Same init than in the Coala test
        _, massgrid = utils.Compute_dustgrid(nSpecies, massmin, massmax)
        dust_mass_fraction_init = utils.Compute_Coala_mass(massgrid, massgrid[0], massgrid[-1], float(params['Coala']['eps'][0]))
        h = massgrid[1:] - massgrid[:-1]
        dust_mass_fraction = dust_mass_fraction_init*h
        dust_mass_fraction /= np.sum(dust_mass_fraction)
        if int(params['Coala']['kernel'][0]) != 3:
            vx1, vx2, vx3 = np.zeros((nSpecies)), np.zeros((nSpecies)), np.zeros((nSpecies))
        else:
            vx1, vx2, vx3, _ = utils.Compute_test_velocity(nSpecies)   # WARNING: do not change values in utils.Compute_test_velocity() without uploading the COALA solution
        for n in range(nSpecies):
            data.dustVc[n][RHO,:,:,:] = dust_mass_fraction[n]
            data.dustVc[n][VX1,:,:,:] = vx1[n]
            data.dustVc[n][VX2,:,:,:] = vx2[n]
            data.dustVc[n][VX3,:,:,:] = vx3[n]