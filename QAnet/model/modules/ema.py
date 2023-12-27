"""
Exponential Moving Average for model parameters.
"""


import paddle

class EMA:

    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}
        self.original = {}

    def register(self, name, val):
        self.shadow[name] = val.numpy().copy()

    def __call__(self, model, num_updates):
        decay = min(self.mu, (1.0 + num_updates) / (10.0 + num_updates))
        for name, param in model.named_parameters():
            if param.stop_gradient is False:
                assert name in self.shadow
                new_average = (1.0 - decay) * param.numpy() + decay * self.shadow[name]
                self.shadow[name] = new_average

    def assign(self, model):
        for name, param in model.named_parameters():
            if param.stop_gradient is False:
                assert name in self.shadow
                self.original[name] = param.numpy().copy()
                param.set_value(self.shadow[name])

    def resume(self, model):
        for name, param in model.named_parameters():
            if param.stop_gradient is False:
                assert name in self.shadow
                param.set_value(self.original[name])
