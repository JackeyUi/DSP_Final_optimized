import soundfile as sf
import numpy as np
from scipy.signal import welch, csd
from scipy.signal import firwin2, freqz, filtfilt
import matplotlib.pyplot as plt

input_signal, input_sr = sf.read('input_white_noise.wav')
output_signal, output_sr = sf.read('output_white_noise.wav')

fs = input_sr  # sample rate = 44.1k
nperseg = 8192  

# Using csd and welch to estimate the FRF H(f)
f, Pxx = welch(input_signal, fs, nperseg=nperseg)
_, Pyy = welch(output_signal, fs, nperseg=nperseg)
_, Pxy = csd(input_signal, output_signal, fs, nperseg=nperseg)

H = Pxy / Pxx

# read desired FRF M(f)
desired_FRF = np.load('desired FRF.npy')  
N = len(desired_FRF)                      
fs = 44100                                 

frequency_axis = np.linspace(0, fs / 2, N)  

M_magnitude = np.abs(desired_FRF)  
H_magnitude = np.abs(H)

W = M_magnitude / (H_magnitude + 1e-4)

# FIR

frequency_axis = np.linspace(0, fs / 2, 4097) 
freq_normalized = np.linspace(0, 1, 4097)
num_taps = 4097  

## adjust the magnitude of W(f)
W_db = 20 * np.log10(np.abs(W))
W_db = np.clip(W_db, -50, 50)
W = W * (0.1** (W_db/40))
##

fir_coeff = firwin2(num_taps, freq_normalized, np.abs(W))
fir_freq, fir_frf = freqz(fir_coeff, worN=4097, fs = 44100)


# calculate FRF 
zerophase_frf = fir_frf * fir_frf.conj()
H_total = H * zerophase_frf

# filter the output_music.wav
output_music, output_rate = sf.read('output_music.wav')
filtered_music = filtfilt(fir_coeff, 1.0, output_music)
sf.write("output_music_compensated_112061559_optimized.wav", filtered_music, output_rate)


plt.figure(figsize=(12, 6))

# plot magnitude
plt.subplot(2, 1, 1)
plt.plot(frequency_axis, 20 * np.log10(np.abs(desired_FRF)), label="Desired Response")
plt.plot(frequency_axis, 20 * np.log10(np.abs(H_total)), label="Compensated System Response")
plt.title("Magnitude Response")
plt.xscale('log')  
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True, which='both', linestyle='--')
plt.legend()

# plot phase
plt.subplot(2, 1, 2)
plt.plot(frequency_axis, np.angle(desired_FRF, deg=True), label="Desired Phase")
plt.plot(frequency_axis, np.angle(H_total, deg=True), label="Compensated Phase")
plt.title("Phase Response")
plt.xscale('log')  
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (degrees)")
plt.grid(True, which='both', linestyle='--')
plt.legend()

plt.tight_layout()
plt.show()

