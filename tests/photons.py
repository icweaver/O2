# Example taken from http://www.vikdhillon.staff.shef.ac.uk/teaching/phy217/instruments/phy217_inst_phot_problems.html
import sys
sys.path.append("../")
from utils import count_photons
from astropy import units as u

observation_params = {
    "m_1":20.0, # Apparent magnitude
    'D':4.2*u.m, # Diameter of GMT
    "tau":1*u.s, # Duration of observation
    "delta_lambda":88*u.nm, # I-band
    "lambda_0":550*u.nm, # I-band central wavelength
    }

# Get total number of photons collected
N_ph = count_photons(**observation_params)

# N_ph should be around 365 photons
print(f"{0.3*N_ph}")
