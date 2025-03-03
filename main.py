import torchaudio
import torch
import numpy as np
from datasets import load_dataset
import warnings

import seaborn as sns
import matplotlib.pyplot as plt
from util import calculate_wer, load_model

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')

SAMPLE_LIMIT = 1000
dataset = load_dataset("hf-audio/esb-datasets-test-only-sorted", "ami", split="test").select(range(SAMPLE_LIMIT))

model = load_model(ff=False)

path = 'audio_143.wav'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
wers = []
hyps = []
refs = []

for i in range(SAMPLE_LIMIT//10):
    path = 'ami_data/' + dataset[i]['audio']['path']
    hyp = model.transcribe(path, verbose=True)['text']
    ref = dataset[i]['text']
    wer = calculate_wer(hyp, ref)
    refs.append(ref)
    hyps.append(hyp)
    wers.append(wer)

al = []
for i in range(SAMPLE_LIMIT):
    path = 'ami_data/' + dataset[i]['audio']['path']
    audio_len_sec = torchaudio.info(path).num_frames / torchaudio.info(path).sample_rate
    al.append(audio_len_sec)

plt.figure(figsize=(10, 5))
plt.ylabel('WER')
plt.xlabel('Audio length (s)')
plt.scatter(al, wers)
# plt.hist(al, bins=20)