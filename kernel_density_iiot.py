#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:07:03 2024

@author: william.sommers
"""

# William Sommers
# HiveMQ Technical Account Manager (TAM)

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.stats import kstest



pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 100)      # Set the display width


# function to calcuate Interquartile Range (IQR)

def IQR(dist):
    return np.percentile(dist, 75) - np.percentile(dist, 25)

# calcualte the Freedman-Diaconis rule for histogram nubmer of bins
def FD(dist):
    return 2 * IQR(dist) * pow(length(dist), 1/3)
    
def signal_plot(df, title, x_axis, y_axis, start, end ):
    # plot a sample
    plt.figure(figsize=(20, 6), dpi=80)
    df.loc[start:end, y_axis].plot.line(x='time', y=y_axis)
    df.loc[start:end,'Min'].plot(x=x_axis, y='Min', linestyle = ':')
    df.loc[start:end,'Max'].plot(x=x_axis, y='Max', linestyle = ':')
    plt.title(title)
    plt.show()
    
def signal_multiplot(df, title, x_axis, y_axis, start, end, *other_y ):
    # plot a sample
    plt.figure(figsize=(20, 6), dpi=80)
    df.loc[start:end, y_axis].plot.line(x='time', y=y_axis)
    df.loc[start:end,'Min'].plot(x=x_axis, y='Min', linestyle = ':')
    df.loc[start:end,'Max'].plot(x=x_axis, y='Max', linestyle = ':')
    plt.title(title)
    
    num_args = len(other_y)
    print("Number of arguments:", num_args)
    for axis in other_y:
        print(axis)
        df.loc[start:end, y_axis].plot.line(x='time', y=axis)
    plt.show()


def histogram_plot(df, title, y_axis, start, end):
    df.loc[samp_start:samp_end, y_axis].plot.hist(column=y_axis)
    plt.title(title)
    plt.show()


def kde_plot(df, title, y_axis, start, end):
    Signal=np.array(list(df.loc[samp_start:samp_end, y_axis])).reshape(-1, 1)

    df.plot.hist(column=y_axis, density=True)
    plt.title(title)
    plt.show()

    # perform the kernel density estimations
    print('Kernel Density Estimation {0}'.format(title))
    kde = KernelDensity(kernel='gaussian', bandwidth=1.0)
    kde.fit(Signal)
    x_grid = np.linspace(Signal.min() -1 , Signal.max() +1, len(Signal)).reshape(-1, 1)
    density_Signal = kde.score_samples(x_grid)

    print(density_Signal)
    plt.plot(x_grid, np.exp(density_Signal))
    plt.ylabel('Signal KDE estimate')
    plt.title(title)
    plt.show()
    
    return kde



# Sample ranges
samp_start = 3000
samp_end = 3120

# file
data_file = 'kde_motor_data.csv'

# read the data - replace this with MQTT data
df = pd.read_csv(data_file)
print(df.describe())
length = len(df)
print('Length = {}'.format(length))


S=np.array(list(df.loc[samp_start:samp_end,'signal'])).reshape(-1,1)


# plot the histogram of the signal
# use Freedman-Diaconis rule for number of bins

bin_edges = np.histogram_bin_edges(S, bins='fd')
print('Bin edges = {}'.format(bin_edges))
plt.hist(S, bins=bin_edges, density=True)
plt.title('Histogram of signal (with Noise)')
plt.show()

# perform the kernel density estimations
print('Kernel Density Estimation (signal with noise)')
kde = KernelDensity(kernel='gaussian', bandwidth=1.0)
kde.fit(S)
density_S = kde.score_samples(S)



###########
# histogram and KDE of ideal signal

kde_ideal = kde_plot(df, 'S1', 'S1', samp_start, samp_end)
kde_noise = kde_plot(df, 'Actual (Noisy) Signal', 'signal', samp_start, samp_end)


S1=np.array(list(df.loc[samp_start:samp_end, 'S1'])).reshape(-1, 1)

df.plot.hist(column='S1', density=True)
plt.title('Histogram of signal (ideal)')
plt.show()

# perform the kernel density estimations
print('Kernel Density Estimation (ideal signal)')
kde2 = KernelDensity(kernel='gaussian', bandwidth=1.0)
kde2.fit(S1)
density_S1 = kde.score_samples(S1)


signal_plot(df, 'Motor Current Signal (representative)', 
            'time', 'signal', 5000, 6500)

signal_plot(df, 'Motor Current Signal (with noise)', 
            'time', 'signal', samp_start, samp_end)

signal_plot(df, 'Motor Current Signal2 (dampened with noise)', 
            'time', 'signal2', samp_start, samp_end)


signal_plot(df, 'Motor Current Signal (missing data with noise)', 
            'time', 'signal3', samp_start, samp_end)

signal_plot(df, 'Motor Current Signal (degraded signal with multi-spectrum noise)', 
            'time', 'signal4', samp_start, samp_end)

signal_plot(df, 'Motor Current Signal (degraded signal multi-spectrum)', 
            'time', 'S4', samp_start, samp_end)

signal_multiplot(df, 'Motor Current Signal (degraded signal multi-spectrum)', 
            'time', 'S4', samp_start, samp_end, 'S4_0', 'S4_1', 'S4_2', 'S4_3')

histogram_plot(df, 'Histogram of Primary Noise', 'N', samp_start, samp_end)
histogram_plot(df, 'Histogram of Secondary Noise', 'N2', samp_start, samp_end)

histogram_plot(df, 'Histogram of degraded signal with multi-spectrum noise',
               'signal4', samp_start, samp_end)

histogram_plot(df, 'Histogram of degraded signal multi-spectrum',
               'S4', samp_start, samp_end)

# plot histogram of dampened, noisy signal
df.loc[samp_start:samp_end, 'signal2'].plot.hist(column='signal2')
plt.title('Histogram of signal2 (dampened with noise)')
plt.show()


# plot histogram of signal3 missing data with noise
df.loc[samp_start:samp_end, 'signal3'].plot.hist(column='signal3')
plt.title('Histogram of signal3 (missing data with noise)')
plt.show()


# plot the noise (sample)
# plt.figure(figsize=(20, 6), dpi=80)
# df.loc[samp_start:samp_end,'N'].plot.line(x='time', y='signal')
# plt.title('Noise')
# plt.show()

signal_plot(df, 'Noise', 
            'time', 'N', samp_start, samp_end)



# plt.figure(figsize=(20, 6), dpi=80)
# df.loc[0:10000,'S0'].plot.line(x='time', y='S0')
# plt.title('Logrithmic decay component S0')
# plt.show()

signal_plot(df, 'Logrithmic decay component S0', 
            'time', 'S0', samp_start, samp_end)



signal_plot(df, 'Pure sinusoidal signal S1', 
            'time', 'S1', samp_start, samp_end)


# plot pure sine wave with actual (overlay, not combined)
plt.figure(figsize=(20, 6), dpi=80)
df.loc[samp_start:samp_end,'signal'].plot.line(x='time', y='signal', label='Actual signal')
df.loc[samp_start:samp_end,'S1'].plot.line(x='time', y='S1', label='Pure sine wave')
df.loc[samp_start:samp_end,'Min'].plot(x='time', y='Min', linestyle = ':', label='Min')
df.loc[samp_start:samp_end,'Max'].plot(x='time', y='Max', linestyle = ':', label='Max')
plt.title('Motor Current Signal Sample (with Noise)')
plt.show()


# K-S tests

sample1 = df.loc[samp_start:samp_end, 'signal']
sample2 = df.loc[samp_start:samp_end, 'signal2']
sample3 = df.loc[samp_start:samp_end, 'S1']

kde_ideal = kde_plot(df, 'S1', 'signal', samp_start, samp_end)
kde_actual = kde_plot(df, 'signal', 'signal', samp_start, samp_end)
kde_noise = kde_plot(df, 'Actual (Noisy) Signal', 'signal2', samp_start, samp_end)
 
x = np.linspace(-5, 5, 1000)

# Calculate the cumulative distribution function (CDF) for each KDE
cdf1 = np.exp(kde_ideal.score_samples(x[:, None])).cumsum()
cdf1 /= cdf1[-1]  # Normalize to ensure CDF ranges from 0 to 1

cdf2 = np.exp(kde_noise.score_samples(x[:, None])).cumsum()
cdf2 /= cdf2[-1]  # Normalize to ensure CDF ranges from 0 to 1

cdf3 = np.exp(kde_actual.score_samples(x[:, None])).cumsum()
cdf3 /= cdf3[-1]  # Normalize to ensure CDF ranges from 0 to 1

# Perform the K-S test
ks_statistic, p_value = kstest(cdf1, cdf3)
print(f"K-S statistic: {ks_statistic:.3f}")
print(f"P-value: {p_value:.3f}")


# Perform the K-S test
ks_statistic, p_value = kstest(cdf1, cdf2)
print(f"K-S statistic: {ks_statistic:.3f}")
print(f"P-value: {p_value:.3f}")



# plt.figure(figsize=(20, 6), dpi=80)
# df.loc[samp_start:3060,'S1'].plot.line(x='time', y='S1')
# plt.title('Pure sinusoidal signal S1')
# plt.show()

# # Generate points for plotting the KDEs
# x_range = np.linspace(min(sample1.min(), sample2.min()), max(sample1.max(), sample2.max()), 200)

# # Calculate the CDFs for both KDEs
# cdf1 = np.array([kde1.integrate_box_1d(-np.inf, x) for x in x_range])
# cdf2 = np.array([kde2.integrate_box_1d(-np.inf, x) for x in x_range])

# # Perform the K-S test
# ks_statistic, p_value = ks_2samp(cdf1, cdf2)

# # Perform the K-S test
# ks_statistic, p_value = ks_2samp(cdf1, cdf2)

# # Print the results
# print("K-S statistic:", ks_statistic)
# print("P-value:", p_value)


# provide an array of ones (1) for the single-dimension trasformation
# ones = []
# for i in range(len(S)):
#     ones.append(1)



#sys.exit(0)

# print(density_S)
# plt.plot(np.exp(density_S))
# plt.ylabel('Signal S KDE estimate')
# plt.title('KDE (ideal signal)')
# plt.show()



# plt.figure(figsize=(20, 6), dpi=80)
# df.loc[5000:6500,'signal'].plot.line(x='time', y='signal')
# plt.title('Motor Current Signal (representative)')
# plt.show()

# plot a sample of the motor current signal
# plt.figure(figsize=(20, 6), dpi=80)
# df.loc[samp_start:samp_end,'signal'].plot.line(x='time', y='signal')
# df.loc[samp_start:samp_end,'Min'].plot(x='time', y='Min', linestyle = ':')
# df.loc[samp_start:samp_end,'Max'].plot(x='time', y='Max', linestyle = ':')
# plt.title('Motor Current Signal Sample (with noise)')
# plt.show()

# plot a sample of the motor current signal2 (dampend with noise)
# plt.figure(figsize=(20, 6), dpi=80)
# df.loc[samp_start:samp_end,'signal2'].plot.line(x='time', y='signal')
# df.loc[samp_start:samp_end,'Min'].plot(x='time', y='Min', linestyle = ':')
# df.loc[samp_start:samp_end,'Max'].plot(x='time', y='Max', linestyle = ':')
# plt.title('Motor Current Signal2 Sample (dampened with noise)')
# plt.show()

# plot a sample of the motor current signal3 (missing data w/noise)
# plt.figure(figsize=(20, 6), dpi=80)
# df.loc[samp_start:samp_end,'signal3'].plot.line(x='time', y='signal')
# df.loc[samp_start:samp_end,'Min'].plot(x='time', y='Min', linestyle = ':')
# df.loc[samp_start:samp_end,'Max'].plot(x='time', y='Max', linestyle = ':')
# plt.title('Motor Current Signal3 Sample (missing data with noise)')
# plt.show()



# print(density_S1)
# plt.plot(np.exp(density_S1))
# plt.ylabel('Signal S KDE estimate')
# plt.title('KDE (ideal signal)')
# plt.show()

#plt.fill_between(S1, np.exp(density_S1), alpha=0.5)
#plt.plot(x, np.full_like(S1, -0.01), '|k', markeredgewidth=1)
#plt.ylim(-0.02, 0.22)
#plt.show()




# sys.exit(0)