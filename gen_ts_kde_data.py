#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:45:50 2024

@author: williamsommers
"""


import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(444)
np.set_printoptions(precision=2)  # Output decimal fmt.
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 100)      # Set the display width


# Synthesized simulated current over the motor life
# Amplitude -
# Frequency - hertz or cycles/s (Hz)
# Dampen = dampening facotr
# time= linear spaced time series
# N= standard gaussian noise
# A= nominal amperage range of mixer motor at 20 HP (sinusoidal) ~28A
#
# Components of simulated motor performance and decay:
#
#   S0= base exponential decay carrier over the simulated motor life
#   S1= base sinusoidal operating current nominal
#   FI1= fault injection - impulse at failure time 1
#   FI2= fault injection - impulse train at failure time 2
#   TC = total current:  S0 + S1 + S2 + N

time_span = 100         # number of seconds
Amplitude = 28
Noise_scale = Amplitude / 2.1
Noise2_scale = Noise_scale * 0.14
Frequency = 60
Nyquist = Frequency * 2
Gen_scale = Nyquist * 16
Dampen = 0.98
numsamples = time_span * Gen_scale
# time = np.linspace(1, numsamples, num=int(Frequency * time_span))
time = np.arange(0, time_span, 1/Gen_scale)


N = np.random.uniform(low=-Noise_scale, high=Noise_scale,
                      size=(numsamples,))  # noise

N2 = np.random.uniform(low= -Noise2_scale, high=Noise_scale,
                       size=(numsamples,))

# A = np.random.weibull(10., numsamples)  # amperage

S0 = np.exp(-Dampen * time)
# S1 = np.sin(2 * np.pi * Frequency * time)
S1 = Amplitude * np.sin(2 * np.pi * Frequency * time)
S2 = Amplitude * np.sin(2 * np.pi * Frequency * time)
S3 = Amplitude * np.sin(2 * np.pi * Frequency * time)
# S4 = Amplitude * np.sin(2 * np.pi * Frequency * time) \
#     + 0.3 * Amplitude * np.sin(5 * np.pi * Frequency * time) \
#     + 0.11 * Amplitude * np.sin(13 * np.pi * Frequency * time) \
#     + 0.08 * Amplitude * np.sin(21 * np.pi * Frequency * time)
    
S4_0 = Amplitude * np.sin(2 * np.pi * Frequency * time)
S4_1 = 0.3 * Amplitude * np.sin(5 * np.pi * Frequency * time)
S4_2 = 0.11 * Amplitude * np.sin(13 * np.pi * Frequency * time)
S4_3 = 0.08 * Amplitude * np.sin(21 * np.pi * Frequency * time)
S4 = S4_0 + S4_1 + S4_2 + S4_3


S5 = Amplitude * np.sin(2 * np.pi * Frequency * time)
S6 = Amplitude * np.sin(2 * np.pi * Frequency * time)

FI1 = np.zeros_like(time)
FI2 = np.zeros_like(time)
Max = np.full(numsamples, Amplitude)
Min = np.full(numsamples, -Amplitude)


# print(time.shape)
# print(N.shape)
# #print(A.shape)
# print(S0.shape)
# print(S1.shape)
# print(FI1.shape)
# print(FI2.shape)
# print(time)
# print(S0)

# build the data frame
dfC = pd.DataFrame()
dfC['time'] = time
dfC['N'] = N
dfC['N2'] = N2
# dfC['A'] = A
dfC['S0'] = S0
dfC['S1'] = S1
dfC['signal2'] = S0 * S2 + N + N2  # dampened noisy signal

dfC['signal3'] = S3 + N  + N2      # inject missing data at 3000:4000
dfC.loc[3030:3070, 'signal3'] = 0
dfC.loc[3100:3140, 'signal3'] = 0
dfC.loc[3520:3700, 'signal3'] = 0

dfC['signal4'] = S4 + N + N2       #
dfC['signal5'] = S5 + N + N2       #
dfC['signal6'] = S6 + N + N2       #
dfC['FI1'] = FI1
dfC['FI2'] = FI2
dfC['Min'] = Min
dfC['Max'] = Max
dfC['S1'] = S1
dfC['S2'] = S2
dfC['S3'] = S3
dfC['S4'] = S4
dfC['S4_0'] = S4_0
dfC['S4_1'] = S4_1
dfC['S4_2'] = S4_2
dfC['S4_3'] = S4_3
dfC['S5'] = S5
dfC['S6'] = S6


# dfC['signal'] = Amplitude * S0 * S1 + FI1 + FI2 + N

dfC['signal'] = S1 + N

print(dfC.head(50))
print(dfC.describe(include='all'))


print('writing output kde_motor_data.csv')

filepath = Path('kde_motor_data.csv')
dfC.to_csv(filepath, index=False)
