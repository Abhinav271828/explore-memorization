import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class LMData(Dataset):
    def __init__(self, filename, split):
        self.split = split

        self.ctoi = {c: i for i, c in enumerate('0123456789()-+*/BEP')}

        bos = self.ctoi['B']
        eos = self.ctoi['E']
        self.data = []
        max_len = 0

        match(split):
            case 'train': dataset_size = 800000
            case 'dev': dataset_size = 100000
            case 'test': dataset_size = 100000

        n = 0
        with open(filename, 'r') as f:
            for line in tqdm(f, desc=f'Loading {split} data'):
                self.data.append([bos] + [self.ctoi[c] for c in line.strip().split()] + [eos])
                max_len = max(max_len, len(self.data[-1]))
                n += 1
                if n >= dataset_size: break
        
        pad = self.ctoi['P']
        for i in range(len(self.data)):
            self.data[i] += [pad] * (max_len - len(self.data[i]))
        
        self.data = torch.tensor(self.data, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, :-1], self.data[idx, 1:]