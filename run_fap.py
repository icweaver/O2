import numpy as np
import pickle
import utils
import pandas as pd

from PyAstronomy import pyasl
from spectres import spectres

#################
# Load data
#################
# Star
#
# Note: according to http://phoenix.astro.physik.uni-goettingen.de/?page_id=10
# there is typo in header. Mass should be in grams and luminosity in erg/s
# wav (Å)
# flux (erg/s/cm^2/cm)
fpath = "PHOENIXmodels/lte03600-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
header_star, flux_star = utils.read_fits(fpath)

fpath = "PHOENIXmodels/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
header_star_wav, wav_star = utils.read_fits(fpath)

# Telluric, wav (um), flux (normalized)
# filename described here: http://etimecalprev.hq.eso.org/observing/etc/data/SM-01_5000_old/SM-01_5000/sm-01_mod2/data/libstruct_pwv.dat
#fpath = "telluric_data/pwv_R1000000_airmass1.0/LBL_A10_s0_w005_R1000000_T.fits"
fpath = "telluric_data/pwv_R300k_airmass1.0/LBL_A10_s0_w005_R0300000_T.fits"
header_tell, data_tell = utils.read_fits(fpath, ext_idx=1)
wav_tell, flux_tell = data_tell["lam"], data_tell["trans"]

# Planet, wav (nm), flux (normalized)
fpath = "ExoplanetO2models/O2_3E5.txt"
names = ["Wavnum", "Wav", "Trans"] # [Wavenumber (cm-1), Wavlength (nm), T]
pd_kwargs = {
    "names": names,
    "sep": " ",
    "skiprows": 4,
    "skipinitialspace": True,
}
df = pd.read_csv(fpath, **pd_kwargs)
# Reverse so wavelength increasing like the others
wav_planet, flux_planet = df["Wav"][::-1], df["Trans"][::-1]

# Put spectra in common units (cm)
wav_S = wav_star * 1e-8 # Stellar (Å -> cm)
wav_T = wav_tell * 1e-4 # Telluric (micrometer -> cm)
wav_P = wav_planet.values * 1e-7 # Planet (nm -> cm)

# Rename spectra to S, T, P for convenience
flux_S = flux_star / np.max(flux_star) # Normalize
flux_T = flux_tell
flux_P = flux_planet.values

################################
# Generate single model spectrum
################################
# Load data
data = {
    'S': {"wav": wav_S, "flux": flux_S},
    'T': {"wav": wav_T, "flux": flux_T},
    'P': {"wav": wav_P, "flux": flux_P},
}
wav_band = [759*1e-7, 772*1e-7] # In cm
R = 3e5

# Put spectra all on the same wavelength grid
_, data['S']["flux_resampled"] = utils.resample(wav_S, wav_band, R, flux_S)
_, data['T']["flux_resampled"] = utils.resample(wav_T, wav_band, R, flux_T)
data["wav_resampled"], data['P']["flux_resampled"] = utils.resample(wav_P, wav_band, R, flux_P)

# Doppler shift S and P to simulate v_star
data['S']["flux_resampled_shifted"], _ = pyasl.dopplerShift(
                                         data["wav_resampled"], 
                                         data['S']["flux_resampled"], 
                                         20., 
                                         edgeHandling="firstlast",
                                         )
data['P']["flux_resampled_shifted"], _ = pyasl.dopplerShift(
                                         data["wav_resampled"], 
                                         data['P']["flux_resampled"], 
                                         20., 
                                         edgeHandling="firstlast",
                                         )

###############################
# Run CCF for different sigma_w
###############################
# Unloaded dict for trying numbda. TODO: put back
wav = data["wav_resampled"]
S = data['S']["flux_resampled_shifted"]
T = data['T']["flux_resampled"]
P =  data['P']["flux_resampled_shifted"]
P0_wav = data['P']['wav']
P0_flux = data['P']['flux']
wav_min = wav_band[0]
wav_max = wav_band[1]
M = 1 # number of bootstrap iterations
I = 1_000 # size of pool
N = 24 # number of transit samples from pool
sigma_ws = [1e-6]

print("Running for the following inputs:")
print(f"M: {M}")
print(f"I: {I}")
print(f"N: {N}")
print(f"sigma_ws: {sigma_ws}")

bootstrap_data = {}
for sigma_w in sigma_ws:
    bootstrap_data[f"{sigma_w}"] = {}
    print(f"Working on sigma_w = {sigma_w} ...")
    for i in range(M):
        if i%10 == 0: print(f"iter {i}/{M}")
        data = utils.CCF_data(wav,wav_min,wav_max,R,S,T,P,P0_wav,P0_flux,sigma_w,I,N)
        bootstrap_data[f"{sigma_w}"][i] = data

    #m = v_maxs[v_maxs >= 20].size
    #FAP = m/M

f = open("run_fap.pkl","wb")
pickle.dump(bootstrap_data, f)
f.close()
