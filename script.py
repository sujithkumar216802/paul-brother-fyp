import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

wavelength = 'Wavelength(nm)'
intensity = 'Intensity(a.u.)'

names_of_files = os.listdir('data')
values_400_to_900 = []
for filename in names_of_files:
    raw_values = pd.read_csv('data\\' + filename, delim_whitespace=True, header=0)
    wavelength_400_to_900 = raw_values[raw_values[wavelength].between(400, 900)]
    plt.figure(0)
    plt.plot(wavelength, intensity, data=wavelength_400_to_900, label=filename)
    plt.figure(1)
    plt.plot(wavelength, intensity, data=raw_values, label=filename)

    # with np.errstate(divide='raise', invalid='raise'):
    #     try:
    #         log_intensity_400_to_900 = -np.log10(wavelength_400_to_900[intensity].to_numpy())
    #     except FloatingPointError:
    #         print(filename)
    log_intensity_400_to_900 = -np.log10(wavelength_400_to_900[intensity].to_numpy())
    values_400_to_900.append({
        wavelength: wavelength_400_to_900[wavelength].to_numpy(),
        intensity: log_intensity_400_to_900,
        'mean': np.mean(log_intensity_400_to_900),
        'std': np.std(log_intensity_400_to_900),
        'filename': filename,
    })

plt.figure(0)
plt.title('400-900nm')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (a.u.)')
plt.legend(bbox_to_anchor=(1.5, 1), loc="upper right")
plt.figure(1)
plt.title('All')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (a.u.)')
plt.legend(bbox_to_anchor=(1.5, 1), loc="upper right")

Xij = np.zeros((len(values_400_to_900), len(values_400_to_900[0][wavelength])))
for i, value in enumerate(values_400_to_900):
    for j, au in enumerate(value[intensity]):
        Xij[i][j] = (value['mean'] - au)/value['std']

for i, row in enumerate(Xij):
    plt.figure(3)
    plt.plot(values_400_to_900[i][wavelength], row, label=values_400_to_900[i]['filename'])

plt.figure(3)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Pre processed(technique mentioned in the glucose paper)')
plt.legend(bbox_to_anchor=(1.5, 1), loc="upper right")
