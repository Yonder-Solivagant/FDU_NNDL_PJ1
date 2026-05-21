from abc import abstractmethod
import numpy as np


def _call_initializer(initialize_method, size, fan_in=None):
    if initialize_method is np.random.normal and fan_in is not None:
        return np.random.normal(0.0, np.sqrt(2.0 / fan_in), size=size)
    try:
        return initialize_method(size=size)
    except TypeError:
        return initialize_method(size)

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = _call_initializer(initialize_method, (in_dim, out_dim), fan_in=in_dim)
        self.b = np.zeros((1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return X @ self.params['W'] + self.params['b']

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        assert self.input is not None, 'forward must be called before backward.'
        self.grads['W'] = self.input.T @ grad
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.params['W'].T
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

        kh, kw = self.kernel_size
        fan_in = in_channels * kh * kw
        self.W = _call_initializer(initialize_method, (out_channels, in_channels, kh, kw), fan_in=fan_in)
        self.b = np.zeros((1, out_channels, 1, 1))
        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}
        self.input_shape = None
        self.input_padded_shape = None
        self.cols = None

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        assert X.ndim == 4, 'conv2D expects input shape [batch, channels, H, W].'
        batch, channels, height, width = X.shape
        assert channels == self.in_channels

        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        X_padded = np.pad(X, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode='constant')
        out_h = (X_padded.shape[2] - kh) // sh + 1
        out_w = (X_padded.shape[3] - kw) // sw + 1

        cols = np.zeros((batch * out_h * out_w, channels * kh * kw), dtype=X.dtype)
        row = 0
        for i in range(out_h):
            for j in range(out_w):
                patch = X_padded[:, :, i * sh:i * sh + kh, j * sw:j * sw + kw]
                cols[row:row + batch] = patch.reshape(batch, -1)
                row += batch
        weight_cols = self.params['W'].reshape(self.out_channels, -1)
        out = cols @ weight_cols.T + self.params['b'].reshape(1, self.out_channels)

        self.input_shape = X.shape
        self.input_padded_shape = X_padded.shape
        self.cols = cols
        return out.reshape(out_h, out_w, batch, self.out_channels).transpose(2, 3, 0, 1)

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        assert self.cols is not None, 'forward must be called before backward.'
        batch, _, out_h, out_w = grads.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        grad_cols = grads.transpose(2, 3, 0, 1).reshape(-1, self.out_channels)
        weight_cols = self.params['W'].reshape(self.out_channels, -1)

        self.grads['W'] = (grad_cols.T @ self.cols).reshape(self.params['W'].shape)
        self.grads['b'] = np.sum(grads, axis=(0, 2, 3), keepdims=True)

        dcols = grad_cols @ weight_cols
        dcols = dcols.reshape(out_h, out_w, batch, self.in_channels, kh, kw)
        dcols = dcols.transpose(2, 3, 0, 1, 4, 5)

        dX_padded = np.zeros(self.input_padded_shape, dtype=grads.dtype)
        for i in range(kh):
            for j in range(kw):
                dX_padded[:, :, i:i + out_h * sh:sh, j:j + out_w * sw:sw] += dcols[:, :, :, :, i, j]

        if ph == 0 and pw == 0:
            return dX_padded
        h_end = dX_padded.shape[2] - ph if ph > 0 else dX_padded.shape[2]
        w_end = dX_padded.shape[3] - pw if pw > 0 else dX_padded.shape[3]
        return dX_padded[:, :, ph:h_end, pw:w_end]
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}


class Flatten(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.optimizable = False
        self.input_shape = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grads):
        return grads.reshape(self.input_shape)
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.optimizable = False
        self.probs = None
        self.labels = None
        self.one_hot = None
        self.grads = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        labels = np.asarray(labels).astype(np.int64).reshape(-1)
        assert predicts.shape[0] == labels.shape[0]
        if predicts.shape[0] == 0:
            raise ValueError('empty batch is not supported.')

        probs = softmax(predicts) if self.has_softmax else predicts
        probs = np.clip(probs, 1e-12, 1.0)

        one_hot = np.zeros_like(probs)
        one_hot[np.arange(labels.shape[0]), labels] = 1.0

        self.probs = probs
        self.labels = labels
        self.one_hot = one_hot
        return float(-np.mean(np.log(probs[np.arange(labels.shape[0]), labels])))
    
    def backward(self):
        # first compute the grads from the loss to the input
        assert self.probs is not None and self.one_hot is not None, 'forward must be called before backward.'
        batch_size = self.labels.shape[0]
        if self.has_softmax:
            self.grads = (self.probs - self.one_hot) / batch_size
        else:
            self.grads = -self.one_hot / self.probs / batch_size
        # Then send the grads to model for back propagation
        if self.model is not None:
            return self.model.backward(self.grads)
        return self.grads

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    def __init__(self, model=None, weight_decay_lambda=1e-4) -> None:
        super().__init__()
        self.model = model
        self.weight_decay_lambda = weight_decay_lambda
        self.optimizable = False

    def forward(self):
        if self.model is None:
            return 0.0
        penalty = 0.0
        for layer in self.model.layers:
            if layer.optimizable and 'W' in layer.params:
                penalty += np.sum(layer.params['W'] ** 2)
        return 0.5 * self.weight_decay_lambda * penalty
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition
