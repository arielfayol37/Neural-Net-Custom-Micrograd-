import random
class Value:
    def __init__(self, value, parents = [None], name = "NaN", operation = "NaN"):
        self.data = value
        self.grad = 0.0
        self.backward = lambda: None
        self.name = name
        self.parents = set(parents)
        self.operation = operation
    def __repr__(self):
        return f"Value(data = {self.data}, name = {self.name}, op = {self.operation})"

    def __add__(self, other):
        new_data = self.data + other.data if isinstance(other, Value) else self.data + other
        output = Value(new_data, parents = [self, other] if isinstance(other, Value) else [self], operation = "+")
        # defining the back value(gradient) of the parents for addition
        def _backward():
            self.grad += output.grad
            if isinstance(other, Value):
                other.grad += output.grad
                other.backward()
            self.backward()
            
        output.backward = _backward
            
        return output

    def __mul__(self, other):
        new_data = self.data * other.data if isinstance(other, Value) else self.data * other
        output = Value(new_data, parents = [self, other] if isinstance(other, Value) else [self], operation = "*")
        # defining the back value(gradient) of the parents for subtraction
        
        def _backward():
            self.grad += other.data * output.grad
            if isinstance(other, Value):
                other.grad += self.data * output.grad
                other.backward()
            self.backward()
            
            
        output.backward = _backward
        return output
    def __rmul__(self, other):
        return self.__mul__(other)
    def __radd__(self, other):
        return self.__add__(other)
    def __rsub__(self, other):
        return self.__sub__(other)
    
    def __neg__(self):
        return Value(-1 * self.data, parents= [self], operation = "-")
    
    def __sub__(self, other):
        new_data = self.data - other.data if isinstance(other, Value) else self.data - other
        output = Value(new_data, parents = [self, other] if isinstance(other, Value) else [self], operation = "-")
        # defining the back value(gradient) of the parents for addition
        def _backward():
            self.grad += output.grad
            if isinstance(other, Value):
                other.grad += -output.grad
                other.backward()
            self.backward()
            
        output.backward = _backward
        return output
        
    def zero_grad(self):
        self.grad = 0
        for parent in self.parents:
            if isinstance(parent, Value):
                parent.zero_grad()

class Neuron:
    def __init__(self, nins):
        self.weights = [Value(random.uniform(-1,1)) for i in range(nins)]
        self.bias = Value(random.uniform(-1,1))
        self.parameters = self.weights + [self.bias]
    def __call__(self, x):
        #assert len(x) == len(self.weights)
        return sum([i * j for i, j in zip(x, self.weights)], self.bias)
    def get_parameters(self):
        return self.parameters

class Layer:
    def __init__(self, nins, nouts):
        self.neurons = [Neuron(nins) for i in range(nouts)]
        self.parameters = []

    def __call__(self, x):
        out = []
        for neuron in self.neurons:
            out.append(neuron(x))
        return out
    def get_parameters(self):
        
        for neuron in self.neurons:
            self.parameters += neuron.get_parameters()
        return self.parameters
class NN:
    def __init__(self, nins, list_nouts):
        layer_sizes = [nins] + list_nouts
        self.layers = [Layer(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]
        self.out = 0.0
        self.parameters = []

    def __call__(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        self.out = out
        return out
    def get_parameters(self):

        for layer in self.layers:
            self.parameters += layer.get_parameters()
        return self.parameters
    def update_parameters(self, lrate = 0.001):
        for out in self.out:
            out.grad = 1.0 
            out.backward()
        
    
