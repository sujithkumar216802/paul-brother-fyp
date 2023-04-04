import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

wavelength = 'Wavelength(nm)'
intensity = 'Intensity(a.u.)'

def read_and_preprocess(data_dir='data'):
    values_400_to_900_df = pd.DataFrame()
    names_of_files = os.listdir(data_dir)
    for filename in names_of_files:
        raw_values = pd.read_csv(data_dir + '\\' + filename, delim_whitespace=True, header=0)
        wavelength_400_to_900_df = raw_values[raw_values[wavelength].between(400, 900)]
        values_400_to_900_df[wavelength] = wavelength_400_to_900_df[wavelength]

        # with np.errstate(divide='raise', invalid='raise'):
        #     try:
        #         log_intensity_400_to_900 = -np.log10(wavelength_400_to_900_df[intensity].to_numpy())
        #     except FloatingPointError:
        #         print(filename)
        log_intensity_400_to_900 = -np.log10(wavelength_400_to_900_df[intensity].to_numpy())
        mean = np.mean(log_intensity_400_to_900)
        std = np.std(log_intensity_400_to_900)
        values_400_to_900_df[filename] = (mean - log_intensity_400_to_900) / std

        plt.figure(0)
        plt.plot(wavelength, intensity, data=raw_values, label=filename)
        plt.figure(1)
        plt.plot(wavelength, intensity, data=wavelength_400_to_900_df, label=filename)
        plt.figure(2)
        plt.plot(values_400_to_900_df[wavelength], values_400_to_900_df[filename], label=filename)

    plt.figure(0)
    plt.title('All')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity (a.u.)')
    plt.legend(bbox_to_anchor=(1.5, 1), loc="upper right")
    plt.figure(1)
    plt.title('400-900nm')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity (a.u.)')
    plt.legend(bbox_to_anchor=(1.5, 1), loc="upper right")
    plt.figure(2)
    plt.title('Values after preprocessing')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Pre processed(technique mentioned in the glucose paper)')
    plt.legend(bbox_to_anchor=(1.5, 1), loc="upper right")

    values_400_to_900_df = values_400_to_900_df.set_index(wavelength)
    return values_400_to_900_df

values = read_and_preprocess()