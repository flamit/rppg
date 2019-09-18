import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
import math

random.seed(123)


class DiagonalLSTM(nn.Module):

    def __init__(self, input_dim, base_out=32):
        super(DiagonalLSTM, self).__init__()
        self.input_dim = input_dim
        self.conv2d = nn.Conv2d(input_dim, 4 * base_out, 1)
        self.conv1d = nn.Conv1d(base_out, 4 * base_out, 2)
        self.base_out = base_out

    def skew(self, inputs, height, width):
        """
        input.shape: (batch size, dim, height, width)
        """
        buffer = torch.zeros(
            [inputs.shape[0], inputs.shape[1], height, 2 * width - 1],
            dtype=torch.float32
        )

        for i in range(height):
            buffer[:, :, i, i:i + width] = buffer[:, :, i, i:i + width] + inputs[:, :, i, :]

        return buffer

    def unskew(self, x, h, w):
        return torch.stack(x, dim=-1)

    def gaussian(self, window_size, sigma):
        gauss = np.array([math.exp(-(x - window_size / 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        gauss = torch.from_numpy(gauss)
        return gauss / gauss.sum()

    def step_fn(self, x, prev_c, prev_h):
        b, c, h = x.shape

        prev_h_padded = torch.zeros([b, self.base_out, 1 + h], dtype=torch.float32)
        prev_h_padded[:, :, 1:] += prev_h

        state_to_state = self.conv1d(prev_h_padded)

        gates = x + state_to_state

        o_f_i = F.sigmoid(gates[:, :3 * self.base_out, :])
        o = o_f_i[:, 0 * self.base_out:1 * self.base_out, :]
        f = o_f_i[:, 1 * self.base_out:2 * self.base_out, :]
        i = o_f_i[:, 2 * self.base_out:3 * self.base_out, :]
        g = F.tanh(gates[:, 3 * self.base_out:4 * self.base_out, :])

        new_c = (f * prev_c) + (i * g)
        new_h = o * F.tanh(new_c)

        return new_c, new_h

    def forward(self, x):
        batch, dim, h, w = x.shape
        x = self.skew(x, h, w)
        input_to_state = self.conv2d(x).permute(3, 0, 1, 2)

        c0 = torch.zeros([batch, self.base_out, h])
        h0 = torch.zeros([batch, self.base_out, h])

        all_c, all_h = [], []
        for i in range(w):
            c0, h0 = self.step_fn(input_to_state[i], c0, h0)
            all_c.append(c0)
            all_h.append(h0)

        return self.unskew(all_h, h, w)


class DiagonalBiLSTM(nn.Module):

    def __init__(self, input_dim, base_out=32):
        super(DiagonalBiLSTM, self).__init__()
        self.forw = DiagonalLSTM(input_dim, base_out)
        self.backw = DiagonalLSTM(input_dim, base_out)

    def forward(self, x):
        f = self.forw(x)
        b = torch.flip(self.backw(torch.flip(x, dims=[3])), dims=[3])
        f[:, :, 1:, :] += b[:, :, :-1, :]
        return f
