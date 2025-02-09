import pickle
import os
import numpy as np
import torch

# read one-subject data(EEG ,fNIRS and label)
subject_root = 'TMVED3/S1.pkl'

with open(subject_root, 'rb') as f:
    # data: list
    # data contains 108 samples, [(EEG), (fNIRS), label]
    # samples.shape = [(60, 1000), (40, 40), (1,)]
    data = pickle.load(f)

eeg = torch.tensor(np.array([i[0] for i in data]), dtype=torch.float32)
nirs = torch.tensor(np.array([i[1] for i in data]), dtype=torch.float32)
labels = torch.tensor([i[2] for i in data], dtype=torch.long)

# eeg.shape = (108, 60, 1000), nirs.shape = (108, 40, 40), labels.shape = (108, )
print(eeg.shape, nirs.shape, labels.shape)


# read multi-sujects data(EEG, fNIRS and labels)
subject_num = 3
samples = 108

eeg = torch.zeros((subject_num * samples, 60, 1000), dtype=torch.float32)
nirs = torch.zeros((subject_num * samples, 40, 40), dtype=torch.float32)
labels = torch.zeros((subject_num * samples,), dtype=torch.long)

subject_list = os.listdir('TMVED3')
subject_root = [os.path.join(os.getcwd(), 'TMVED3', i) for i in subject_list]

# stack multi-subjects data
for index, sub in enumerate(subject_root):
    with open(sub, 'rb') as f:
        data = pickle.load(f)
    
    eeg[index * samples: (index + 1) * samples] = torch.tensor(np.array([i[0] for i in data]), dtype=torch.float32)
    nirs[index * samples: (index + 1) * samples] = torch.tensor(np.array([i[1] for i in data]), dtype=torch.float32)
    labels[index * samples: (index + 1) * samples] = torch.tensor(np.array([i[2] for i in data]), dtype=torch.long)

print(eeg.shape, nirs.shape, labels.shape)
