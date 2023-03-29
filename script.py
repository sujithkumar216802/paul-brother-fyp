import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

wavelength = 'Wavelength(nm)'
intensity = 'Intensity(a.u.)'

names_of_files = os.listdir('data')
values = []
for filename in names_of_files:
    raw_values = pd.read_csv('data\\' + filename, delim_whitespace=True, header=0)
    wavelength_400_to_900 = raw_values[raw_values[wavelength].between(400, 900)]
    plt.figure(0)
    plt.plot(wavelength, intensity, data=wavelength_400_to_900)
    plt.figure(1)
    plt.plot(wavelength, intensity, data=raw_values)

    log_intensity_400_900 = -np.log10(wavelength_400_to_900[intensity].to_numpy())
    values.append({
        wavelength: wavelength_400_to_900[wavelength].to_numpy(),
        intensity: log_intensity_400_900,
        'mean': np.mean(log_intensity_400_900),
        'std': np.std(log_intensity_400_900)
    })

plt.figure(0)
plt.title('400-900nm')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (a.u.)')
plt.figure(1)
plt.title('All')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (a.u.)')
