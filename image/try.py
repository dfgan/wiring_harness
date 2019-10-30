# -*- coding: utf-8 -*-
# author: dengfan
# datetime:2018-11-09 11:16

import numpy as np
arr = np.random.randn(2,6)
print(np.multiply(*arr.shape))


import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable

a = torch.LongTensor([[1,2,4,6],[7,9,5,3]])

bed = nn.Embedding(60,15)

b = bed(a)
print(bed.weight.data.numpy())
