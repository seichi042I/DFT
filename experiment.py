import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# complex tests
print("複素数計算")
x = 1 + 1j
y = 1 - 1j

print(x + y)


# mixed sin
f1 = 200
f2 = 300
f3 = 500
y = np.array([np.sin(2*np.pi*x*(f1/sr)) + 0.5*np.sin(2*np.pi*x *
             (f2/sr)) + 0.1*np.sin(2*np.pi*x*(f3/sr)) for x in range(sr)])
# y = np.array([np.sin(2*np.pi*x*(f1/sr)) for x in range(sr)])
plt.plot(y)
plt.show()

# eular test
theata = np.pi / 4
c = np.exp(1j * theata)
print(c)
grad = np.imag(c) / np.real(c)
print((np.arctan(grad))/2/np.pi*360)
print(np.angle(c)/2/np.pi*360)

# signal 0 padding
n_dft = 1024

print(y.shape[0]//n_dft)
padding = n_dft - (y.shape[0] % n_dft)
y = np.concatenate([y, np.asarray([0]*padding)])
print(y.shape[0]/n_dft)

# split
n_split = y.shape[0]//n_dft
splited = np.asarray(np.split(y, n_split))
print(splited.shape)

# hann window


def hann(T: int) -> np.ndarray:
    _2pi = 2 * np.pi
    return np.array([0.5 - 0.5 * np.cos(_2pi*t/T) for t in range(T)])


window = hann(n_dft)
print(window.shape)

# plt.plot(window)
# plt.show()

# window apply
stft = []
hop = 128
for i in tqdm(range((len(y)-n_dft)//hop)):
    fx = y[i*hop:i*hop+n_dft] * window
    dft = []
    for f in range(256):
        freq = 2**(f*0.0625)*(n_dft/sr)
        def eular(n): return np.exp(-1j*(2*np.pi*freq)*(n/n_dft))
        sigma = 0j
        for n, x in enumerate(fx):
            sigma += x*eular(n)
        dft.append(np.log(np.abs(sigma)))
    stft.append(dft)

stft = np.array(stft)
plt.imshow(stft.T, origin='lower')
plt.show()
