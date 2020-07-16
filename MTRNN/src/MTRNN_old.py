#!/usr/bin/env python3
# coding: utf-8
import torch
from torch.nn import Module, Linear
import numpy as np


class MTRNNCell(torch.nn.Module):
    def __init__(
        self,
        input_output_size=2,
        io_hidden_size=10,
        fast_hidden_size=10,
        slow_hidden_size=5,
        tau_input_output=2,
        tau_fast_hidden=5,
        tau_slow_hidden=70,
    ):
        super().__init__()

        # hidden layer sizes
        self.input_output_size = input_output_size
        self.io_hidden_size = io_hidden_size
        self.fast_hidden_size = fast_hidden_size
        self.slow_hidden_size = slow_hidden_size

        # decay rates
        self.tau_input_output = tau_input_output
        self.tau_fast_hidden = tau_fast_hidden
        self.tau_slow_hidden = tau_slow_hidden

        # linear mappings
        self.input_to_io_mapping = Linear(self.input_output_size, self.io_hidden_size)
        self.io_to_fast_mapping = Linear(self.io_hidden_size, self.fast_hidden_size)
        self.fast_to_fast_mapping = Linear(self.fast_hidden_size, self.fast_hidden_size)
        self.fast_to_slow_mapping = Linear(self.fast_hidden_size, self.slow_hidden_size)
        self.slow_to_slow_mapping = Linear(self.slow_hidden_size, self.slow_hidden_size)
        self.slow_to_fast_mapping = Linear(self.slow_hidden_size, self.fast_hidden_size)
        self.fast_to_io_mapping = Linear(self.fast_hidden_size, self.io_hidden_size)
        self.io_to_io_mapping = Linear(self.io_hidden_size, self.io_hidden_size)
        self.io_to_output_mapping = Linear(self.io_hidden_size, self.input_output_size)

        self.tanh = torch.nn.Tanh()
        # self.slow_hidden0 = None
        # batch_size = 1
        # slow_hidden_state = torch.FloatTensor(batch_size, self.slow_hidden_size)
        # slow_hidden_state.uniform_(
        #     -np.sqrt(1 / self.slow_hidden_size), np.sqrt(1 / self.slow_hidden_size)
        # )
        # slow_hidden_state.requires_grad = True
        # self.slow_hidden0 = torch.nn.Parameter(slow_hidden_state)

    def forward(self, input, hidden_states):

        # if hidden_states is None:
        #     hidden_states = self._get_initial_hidden_states(input.size(0))
        io_hidden_state, fast_hidden_state, slow_hidden_state = hidden_states
        # io_hidden_state = io_hidden_state.detach()

        new_fast_hidden_state = self._update_state(
            previous=fast_hidden_state,
            new=[
                self.fast_to_fast_mapping(fast_hidden_state),
                self.slow_to_fast_mapping(slow_hidden_state),
                self.io_to_fast_mapping(io_hidden_state),
            ],
            tau=self.tau_fast_hidden,
        )

        new_slow_hidden_state = self._update_state(
            previous=slow_hidden_state,
            new=[
                self.slow_to_slow_mapping(slow_hidden_state),
                self.fast_to_slow_mapping(fast_hidden_state),
            ],
            tau=self.tau_slow_hidden,
        )

        new_io_hidden_state = self._update_state(
            previous=io_hidden_state,
            new=[
                self.io_to_io_mapping(io_hidden_state),
                self.fast_to_io_mapping(fast_hidden_state),
                self.input_to_io_mapping(input),
            ],
            tau=self.tau_input_output,
        )

        output = self.tanh(self.io_to_output_mapping(io_hidden_state))
        new_hidden_states = (
            self.tanh(new_io_hidden_state),
            self.tanh(new_fast_hidden_state),
            self.tanh(new_slow_hidden_state),
        )

        return output, new_hidden_states

    def _update_state(self, previous, new, tau):
        connected = torch.stack(new)
        new_summed = connected.sum(dim=0)
        return (1 - 1 / tau) * previous + new_summed / tau

    def _get_initial_hidden_states(self, batch_size):
        # Allocate memory
        # input_output_hidden_state = torch.FloatTensor(
        #     batch_size, self.io_hidden_size
        # )
        # fast_hidden_state = torch.FloatTensor(batch_size, self.fast_hidden_size)

        # # Initialize by sampling uniformly from (-sqrt(1/hidden_size), sqrt(1/hidden_size))
        # input_output_hidden_state.uniform_(
        #     -np.sqrt(1 / self.io_hidden_size),
        #     np.sqrt(1 / self.io_hidden_size),
        # )
        # fast_hidden_state.uniform_(
        #     -np.sqrt(1 / self.fast_hidden_size), np.sqrt(1 / self.fast_hidden_size)
        # )

        io_hidden_state = torch.zeros(size=(batch_size, self.io_hidden_size))
        fast_hidden_state = torch.zeros(size=(batch_size, self.fast_hidden_size))
        slow_hidden_size = torch.zeros(size=(batch_size, self.slow_hidden_size))
        return io_hidden_state, fast_hidden_state, slow_hidden_size


class CustomRNN(Module):
    def __init__(self, cell):
        super().__init__()
        self.rnn = cell

    def forward(self, input, hidden_state=None):
        output = torch.empty_like(input)
        for t in range(input.size(0)):
            output[t], hidden_state = self.rnn(input[t], hidden_state)
            # print(t, input[t], output[t])
        return output, hidden_state
