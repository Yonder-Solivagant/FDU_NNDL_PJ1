from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.grads[key] is None:
                        continue
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] -= self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu=0.9):
        super().__init__(init_lr, model)
        self.mu = mu
        self.velocity = {}
        for layer_idx, layer in enumerate(self.model.layers):
            if layer.optimizable:
                self.velocity[layer_idx] = {
                    key: np.zeros_like(value)
                    for key, value in layer.params.items()
                }
    
    def step(self):
        for layer_idx, layer in enumerate(self.model.layers):
            if layer.optimizable != True:
                continue
            if layer_idx not in self.velocity:
                self.velocity[layer_idx] = {
                    key: np.zeros_like(value)
                    for key, value in layer.params.items()
                }
            for key in layer.params.keys():
                if layer.grads[key] is None:
                    continue
                grad = layer.grads[key]
                if layer.weight_decay:
                    grad = grad + layer.weight_decay_lambda * layer.params[key]
                self.velocity[layer_idx][key] = self.mu * self.velocity[layer_idx][key] - self.init_lr * grad
                layer.params[key] += self.velocity[layer_idx][key]
