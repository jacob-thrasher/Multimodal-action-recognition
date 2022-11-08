from utils import  VA_Dataset, collate_fn
from torch.utils.data import DataLoader
from functools import partial
from scipy import signal
from scipy.io import wavfile
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

dst = '/mnt/sdc/jacob/spec'
root = '/mnt/sdc/jacob/audio'
converted = '/mnt/sdc/jacob/converted_audio'


for file in tqdm(os.listdir(root), desc='Converting wavs'):
    try:
        path = os.path.join(root, file)
        sample_rate, samples = wavfile.read(path)
        # samples = np.mean(samples, axis=1)
        freq, times, spec = signal.spectrogram(samples[:, 0], sample_rate)

        os.rename(os.path.join(root, file), os.path.join(converted, file))

        filename = file.split('.')[0]

        fig, ax = plt.subplots(1)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis('off')
        # max_ = spec.max() if spec.max() != 0 else 1
        print(file)
        plt.pcolormesh(1000*times, freq/1000, 10*np.log10(spec/spec.max()), vmin=-120, vmax=0, cmap='inferno')
        ax.axis('off')
        plt.savefig(os.path.join(dst, f'{filename}.png'), frameon=False)
        plt.close()
    except Exception as e:
        print(file)


