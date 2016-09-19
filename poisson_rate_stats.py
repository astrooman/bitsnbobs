#!/bin/env python

import numpy as np
import matplotlib.pyplot as py
from scipy.stats import poisson
from numpy import pi

#Observed FRB rate from Parkes
frb_rate=6.e3	#bursts per sky per day (Champion+15)

#PAF properties
fwhm=14			#arcmin
fov_one_beam=pi*(fwhm/(2.*60))**2	#degrees
fov_all_deg=27*fov_one_beam
fov_all=fov_all_deg/41252.9

#Scale factor for relative sensitivies
#Note, it does *not* take cosmology into account,
# which you should well before a redshift of 5
Tsys_factor=(50./25.)**-1.5

#Time request
Ttot=800./24	#Total time in days

#Rate per observing program
mu=frb_rate*Tsys_factor*fov_all*Ttot
print mu
#Number observed dummy array
k=np.linspace(0.0,10.0, num=200)

#Probability mass function
frb_pmf=poisson.pmf(k, mu)

#frb_cdf=poisson.cdf(k, mu)

#Calculate the interval that gives gives 0.68% of distribution
print poisson.interval(0.68, mu)
print poisson.interval(0.95, 6000)
