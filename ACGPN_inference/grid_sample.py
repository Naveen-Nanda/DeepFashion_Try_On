# encoding: utf-8

import torch.nn.functional as F
from torch.autograd import Variable

def grid_sample(input, grid, canvas = None):
    output = F.grid_sample(input, grid)
    if canvas is None:
        return output
    else:
        input_mask = Variable(input.data.new(input.size()).fill_(1))
        output = F.grid_sample(input_mask, grid)
        padded_output = output * output + canvas * (1 - output)
        return padded_output
