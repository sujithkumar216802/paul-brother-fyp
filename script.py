import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Lasso, Ridge

wavelength = 'Wavelength(nm)'
intensity = 'Intensity(a.u.)'
all_target_df = pd.read_csv('target.txt', delim_whitespace=True, header=0)

def read_and_preprocess(data_dir='forehead data'):
    values_400_to_900_df = pd.DataFrame()
    target_df = pd.DataFrame(columns=['name', 'target'])
    names_of_files = os.listdir(data_dir)
    for filename in names_of_files:
        raw_values = pd.read_csv(data_dir + '\\' + filename, delim_whitespace=True, header=0)
        raw_values_400_to_900_df = raw_values[raw_values[wavelength].between(400, 900)]
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
        log_intensity_400_to_900 = -np.log10(raw_values_400_to_900_df[intensity].to_numpy())
        mean = np.mean(log_intensity_400_to_900)
        std = np.std(log_intensity_400_to_900)
        values_400_to_900_df[filename] = (mean - log_intensity_400_to_900) / std

        plt.figure(0)
        plt.plot(wavelength, intensity, data=raw_values, label=filename)
        plt.figure(1)
        plt.plot(wavelength, intensity, data=raw_values_400_to_900_df, label=filename)
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
    return values_400_to_900_df.T, target_df

values_df, target_df = read_and_preprocess()
target = target_df['target'].to_numpy()
x_train, x_test, y_train, y_test = train_test_split(values_df, target, random_state=42)


# PCR related stuff
pca_explained_variance = []
for i in range(1, x_train.shape[0] + 1):
    pca = PCA(n_components=i)
    pca.fit(x_train)
    pca_explained_variance.append(pca.explained_variance_ratio_.sum())

plt.figure(3)
plt.plot(range(1, x_train.shape[0] + 1), pca_explained_variance, '-ro')

# # PLS related stuff
# pls_explained_variance = []
# for i in range(1, values.shape[0] + 1):
#     pls = PLSRegression(n_components=i)
#     pls.fit(values, target)    

#     variance_in_x = np.var(pls.x_scores_, axis = 0) 
#     fractions_of_explained_variance = variance_in_x / np.sum(variance_in_x)
#     pls_explained_variance.append(fractions_of_explained_variance)

# plt.figure(4)
# plt.plot(range(1, values.shape[0] + 1), pls_explained_variance, '-ro')


for i in range(1, x_train.shape[0] + 1):
    pcr = make_pipeline(PCA(n_components=i), LinearRegression())
    pcr.fit(x_train, y_train)
    plt.figure(4)
    plt.scatter(y_test, pcr.predict(x_test), label=f'{i} components')

    pcr = make_pipeline(PCA(n_components=i), Lasso())
    pcr.fit(x_train, y_train)
    plt.figure(5)
    plt.scatter(y_test, pcr.predict(x_test), label=f'{i} components')

    pcr = make_pipeline(PCA(n_components=i), Ridge())
    pcr.fit(x_train, y_train)
    plt.figure(6)
    plt.scatter(y_test, pcr.predict(x_test), label=f'{i} components')

    pls = PLSRegression(n_components=i)
    pls.fit(x_train, y_train)
    plt.figure(7)
    plt.scatter(y_test, pls.predict(x_test), label=f'{i} components')

plt.figure(4)
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('PCR using linear')
plt.legend(loc="upper right")
plt.figure(5)
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('PCR using lasso')
plt.legend(loc="upper right")
plt.figure(6)
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('PCR using ridge')
plt.legend(loc="upper right")
plt.figure(7)    
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('PLS')
plt.legend(loc="upper right")

linear = LinearRegression()
linear.fit(x_train, y_train)
plt.figure(8)
plt.scatter(y_test, linear.predict(x_test))
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('linear regression')

lasso = Lasso()
lasso.fit(x_train, y_train)
plt.figure(9)
plt.scatter(y_test, lasso.predict(x_test))
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('lasso regression')

ridge = Ridge()
ridge.fit(x_train, y_train)
plt.figure(10)
plt.scatter(y_test, ridge.predict(x_test))
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('ridge regression')
