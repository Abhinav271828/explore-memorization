import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class LMData(Dataset):
    def __init__(self, filename, size, split):
        self.split = split

        self.ctoi = {c: i for i, c in enumerate('0123456789()-+*/BEP')}

        self.bos = self.ctoi['B']
        self.eos = self.ctoi['E']
        self.data = []
        max_len = 0

        match(split):
            case 'train': dataset_size = int(0.8 * size)
            case 'dev': dataset_size = int(0.1 * size)
            case 'test': dataset_size = int(0.1 * size)

        n = 0
        with open(filename, 'r') as f:
            for line in tqdm(f, desc=f'Loading {split} data'):
                self.data.append([self.bos] + [self.ctoi[c] for c in line.strip().split()] + [self.eos])
                max_len = max(max_len, len(self.data[-1]))
                n += 1
                if n >= dataset_size: break
        
        self.pad = self.ctoi['P']
        for i in range(len(self.data)):
            self.data[i] += [self.pad] * (max_len - len(self.data[i]))
        
        self.data = torch.tensor(self.data, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, :-1], self.data[idx, 1:]