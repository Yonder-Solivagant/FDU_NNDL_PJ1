from .op import *
import pickle
import numpy as np


def _he_initializer(fan_in):
    return lambda size: np.random.normal(0.0, np.sqrt(2.0 / fan_in), size=size)

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        super().__init__()
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(
                    in_dim=size_list[i],
                    out_dim=size_list[i + 1],
                    initialize_method=_he_initializer(size_list[i])
                )
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        self.layers = []
        for i in range(len(self.size_list) - 1):
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.W = param_list[i + 2]['W']
            layer.b = param_list[i + 2]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[i + 2]['weight_decay']
            layer.weight_decay_lambda = param_list[i + 2]['lambda']
            if self.act_func == 'Logistic':
                raise NotImplementedError
            elif self.act_func == 'ReLU':
                layer_f = ReLU()
            self.layers.append(layer)
            if i < len(self.size_list) - 2:
                self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, input_shape=(1, 28, 28), num_classes=10, conv_channels=8,
                 kernel_size=3, stride=1, padding=1, lambda_list=None):
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.num_classes = num_classes
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.lambda_list = lambda_list
        self.layers = []
        self._build_layers()

    def _build_layers(self):
        channels, height, width = self.input_shape
        if isinstance(self.kernel_size, int):
            kh = kw = self.kernel_size
        else:
            kh, kw = self.kernel_size
        if isinstance(self.stride, int):
            sh = sw = self.stride
        else:
            sh, sw = self.stride
        if isinstance(self.padding, int):
            ph = pw = self.padding
        else:
            ph, pw = self.padding

        conv_out_h = (height + 2 * ph - kh) // sh + 1
        conv_out_w = (width + 2 * pw - kw) // sw + 1
        flatten_dim = self.conv_channels * conv_out_h * conv_out_w
        lambda_list = self.lambda_list or [0.0, 0.0]

        conv = conv2D(
            in_channels=channels,
            out_channels=self.conv_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            weight_decay=lambda_list[0] > 0,
            weight_decay_lambda=lambda_list[0],
        )
        classifier = Linear(
            in_dim=flatten_dim,
            out_dim=self.num_classes,
            initialize_method=_he_initializer(flatten_dim),
            weight_decay=lambda_list[1] > 0,
            weight_decay_lambda=lambda_list[1],
        )
        self.layers = [conv, ReLU(), Flatten(), classifier]

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if X.ndim == 2:
            X = X.reshape((X.shape[0],) + self.input_shape)
        elif X.ndim == 3:
            X = X[:, np.newaxis, :, :]
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            state = pickle.load(f)
        if not isinstance(state, dict) or state.get('model_type') != 'Model_CNN':
            raise ValueError('The saved file is not a Model_CNN checkpoint.')

        config = state['config']
        self.input_shape = tuple(config['input_shape'])
        self.num_classes = config['num_classes']
        self.conv_channels = config['conv_channels']
        self.kernel_size = config['kernel_size']
        self.stride = config['stride']
        self.padding = config['padding']
        self.lambda_list = config['lambda_list']
        self._build_layers()

        optimizable_layers = [layer for layer in self.layers if layer.optimizable]
        for layer, saved in zip(optimizable_layers, state['params']):
            for key, value in saved['params'].items():
                layer.params[key] = value
                setattr(layer, key, value)
            layer.weight_decay = saved['weight_decay']
            layer.weight_decay_lambda = saved['lambda']
        
    def save_model(self, save_path):
        state = {
            'model_type': 'Model_CNN',
            'config': {
                'input_shape': self.input_shape,
                'num_classes': self.num_classes,
                'conv_channels': self.conv_channels,
                'kernel_size': self.kernel_size,
                'stride': self.stride,
                'padding': self.padding,
                'lambda_list': self.lambda_list,
            },
            'params': [],
        }
        for layer in self.layers:
            if layer.optimizable:
                state['params'].append({
                    'params': {key: value for key, value in layer.params.items()},
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda,
                })

        with open(save_path, 'wb') as f:
            pickle.dump(state, f)
