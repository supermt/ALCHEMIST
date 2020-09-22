from torch.nn.utils.rnn import pack_sequence

import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5])
c = torch.tensor([6])

pack_sequence([a,b,c])