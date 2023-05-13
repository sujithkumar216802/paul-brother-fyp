import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import cross_val_predict

wavelength = 'Wavelength(nm)'
intensity = 'Intensity(a.u.)'
all_target_df = pd.read_csv('target.txt', delim_whitespace=True, header=0)

def read_and_preprocess(data_dir='forehead data'):
    values_400_to_900_df = pd.DataFrame()
    target_df = pd.DataFrame(columns=['name', 'target'])
    names_of_files = os.listdir(data_dir)
    for filename in names_of_files:
        raw_values = pd.read_csv(
            data_dir + '\\' + filename, delim_whitespace=True, header=0)
        raw_values_400_to_900_df = raw_values[raw_values[wavelength].between(
            400, 900)]
        values_400_to_900_df[wavelength] = raw_values_400_to_900_df[wavelength]

        # with np.errstate(divide='raise', invalid='raise'):
        #     try:
        #         log_intensity_400_to_900 = -np.log10(raw_values_400_to_900_df[intensity].to_numpy())
        #     except FloatingPointError:
        #         print(f'Invalid values in {filename}')
        #         continue

        # try:
        #     all_target_df[all_target_df['name'] == filename]['target'].to_numpy()[0]
        # except IndexError:
        #     print(f'No target value for {filename}')
        #     continue

        target_df.loc[len(target_df)] = {
            "name": filename,
            "target": all_target_df[all_target_df['name'] == filename]['target'].to_numpy()[0]
        }

        # SNV
        log_intensity_400_to_900 = - \
            np.log10(raw_values_400_to_900_df[intensity].to_numpy())
        mean = np.mean(log_intensity_400_to_900)
        std = np.std(log_intensity_400_to_900)
        values_400_to_900_df[filename] = (
            mean - log_intensity_400_to_900) / std

        plt.figure(0)
        plt.plot(wavelength, intensity, data=raw_values, label=filename)
        plt.figure(1)
        plt.plot(wavelength, intensity,
                 data=raw_values_400_to_900_df, label=filename)
        plt.figure(2)
        plt.plot(values_400_to_900_df[wavelength],
                 values_400_to_900_df[filename], label=filename)

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
    return values_400_to_900_df.T, target_df

x_train, target_df = read_and_preprocess(data_dir='ffs')
y_train = target_df['target'].to_numpy()


# PCR related stuff
pca_explained_variance = []
for i in range(1, x_train.shape[0] + 1):
    pca = PCA(n_components=i)
    pca.fit(x_train)
    pca_explained_variance.append(pca.explained_variance_ratio_.sum())
    
plt.figure(3)
plt.plot(range(1, x_train.shape[0] + 1), pca_explained_variance, 'r-o', label='PCR', markerfacecolor='none')

# PLS related stuff
pls_explained_variance = []
for i in range(1, x_train.shape[0] + 1):
    pls = PLSRegression(n_components=i)
    pls.fit(x_train, y_train)
    pls_explained_variance.append(explained_variance_score(y_train, pls.predict(x_train)))

plt.figure(3)
plt.plot(range(1, x_train.shape[0] + 1), pls_explained_variance, 'b-^', label='PLS', markerfacecolor='none')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained variance')
plt.legend(loc="lower right")


comp = 8

pcr = make_pipeline(PCA(n_components=comp), LinearRegression())
pcr.fit(x_train, y_train)
plt.figure(5)
plt.scatter(y_train, pcr.predict(x_train), facecolor='none', color='red',  label=f'PCR with {comp} components')

pls = PLSRegression(n_components=comp)
pls.fit(x_train, y_train)
plt.figure(5)
plt.scatter(y_train, pls.predict(x_train), facecolor='none', color='blue', label=f'PLSR with {comp} components')

plt.figure(5)
plt.xlabel('Observed Response')
plt.ylabel('Fitted Response')
plt.legend(loc="upper left")

comp_in_loadings = 1 # 0-indexing.. 0 is 1, 1 is 2, etc
plt.figure(7)
plt.plot(x_train.columns, (pls.x_loadings_.T[comp_in_loadings]).T, label=f'PLSR Component no.{comp_in_loadings + 1}')
plt.xlabel(wavelength)
plt.ylabel('PLSR Loadings')
plt.legend(loc="upper left")

plt.figure(8)
plt.plot(x_train.columns, pca.components_[comp_in_loadings].T, label=f'PCR Component no. {comp_in_loadings + 1}')
plt.xlabel(wavelength)
plt.ylabel('PCA Loadings')
plt.legend(loc="upper left")

pls_mspe_array = []
pcr_mspe_array = []
for i in range(1, x_train.shape[0] + 1):
    pls = PLSRegression(n_components=i)
    pls.fit(x_train, y_train)
    pls_pred = pls.predict(x_train)
    pls_mspe = mean_squared_error(y_train, pls_pred)
    pls_mspe_array.append(pls_mspe)

    pcr = make_pipeline(PCA(n_components=i), LinearRegression())
    pcr.fit(x_train, y_train)
    pcr_pred = pcr.predict(x_train)
    pcr_mspe = mean_squared_error(y_train, pcr_pred)
    pcr_mspe_array.append(pcr_mspe)

plt.figure(6)
plt.plot(range(1, x_train.shape[0] + 1), pcr_mspe_array, 'r-o', label='PCR', markerfacecolor='none')
plt.plot(range(1, x_train.shape[0] + 1), pls_mspe_array, 'b-^', label='PLS', markerfacecolor='none')
plt.xlabel('Number Of Components')
plt.ylabel('Estimated Mean Squared Prediction Error')
plt.legend(loc="lower right")
