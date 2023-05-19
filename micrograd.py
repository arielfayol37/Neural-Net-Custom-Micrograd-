import random, math
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
        # defining the back value(gradient) of the parents for multiplication
        
        def _backward():
            self.grad += other.data * output.grad
            if isinstance(other, Value):
                other.grad += self.data * output.grad
                other.backward()
            self.backward()   
        output.backward = _backward
        return output
    
    def __pow__(self, other):
        new_data = (self.data ** other.data) if isinstance(other, Value) else (self.data ** other)
        output = Value(new_data, parents = [self, other] if isinstance(other, Value) else [self], operation = f"**{other}")
        # defining the back value(gradient) of the parents for power
        
        def _backward():
            self.grad += other.data * (self.data ** (other.data - 1)) if isinstance(other, Value) else \
                                other * (self.data ** (other -1)) * output.grad
            if isinstance(other, Value):
                other.grad += output * math.log(self.data) * output.grad
                other.backward()
            self.backward()    
        output.backward = _backward
        return output
    
    def exp(self, constant = 1):
        new_data = math.exp(constant * self.data)
        output = Value(new_data, parents = [self])
        def _backward():
            self.grad += output
        output.backward = _backward
        return output
    
    def __truediv__(self, other):
        return (self * other ** -1) if isinstance(other, Value) else Value(self.data/other)
    def __rtruediv__(self, other):
        return other * (self ** -1)
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
        # defining the back value(gradient) of the parents for subtraction
        def _backward():
            self.grad += output.grad
            if isinstance(other, Value):
                other.grad += -output.grad
                other.backward()
            self.backward()
            
        output.backward = _backward
        return output
        
    def zero_grad(self):
        self.grad = 0.0

#------------------------------------------------------------
## Some Non-Linearities
    
    def tanh(self):
        e = self.exp(2)
        output = (e + Value(1))/(e -Value(1)) 
        def _backward():
            self.grad += (Value(1) - output ** 2) * output.grad
        output.backward = _backward
        return output

    def relu(self):
        new_data = 0 if self.data < 0 else self.data
        output = Value(new_data)
        def _backward():
            self.grad += 0 if self.data < 0 else output.grad
        output.backward = _backward
        return output
    
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
    """
    def __init__(self, nins, nouts, activation = None):
        self.neurons = [Neuron(nins) for i in range(nouts)]
        self.parameters = []
        self.activation = activation
        

    def __call__(self, x):
        out = []
        if self.activation is None:
            for neuron in self.neurons:
                out.append(neuron(x))
        else:
            for neuron in self.neurons:
                out.append(self.activation(neuron(x))
        return out
 
    """
    def __init__(self, nins, nouts, activation = None):
        self.neurons = [Neuron(nins) for i in range(nouts)]
        self.parameters = []
        self.activation = activation
        

    def __call__(self, x):
        out = []
        if self.activation is None:
            for neuron in self.neurons:
                out.append(neuron(x))
    
        elif self.activation == "relu":
            for neuron in self.neurons:
                out.append(neuron(x).relu())
                           
        elif self.activation == "tanh":
            for neuron in self.neurons:
                out.append(neuron(x).tanh())
        return out
   
    def get_parameters(self):
        for neuron in self.neurons:
            self.parameters += neuron.get_parameters()
        return self.parameters
class MLP:
    def __init__(self, nins, list_nouts, activations = [], layers_to_be_activated = []):
        assert len(activations) == len(layers_to_be_activated)
        layer_sizes = [nins] + list_nouts
        self.layers = [Layer(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]
        for i, index_of_layer in enumerate(layers_to_be_activated):
            self.layers[index_of_layer].activation = activations[i]
        self.out = 0.0
        self.parameters = self.get_parameters()

    def __call__(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        self.out = out
        return out
    
    def get_parameters(self):
        params = []
        for layer in self.layers:
            params += layer.get_parameters()
        return params
            
    def zero_grad(self):
        for param in self.parameters:
            param.grad = 0.0
        
class Loss:
    def __init__(self):
        pass
    
    def MSE(self, ypred, ytrue):
        assert len(ypred) ==  len(ytrue), "len of ypred and ytrue don't match"
        loss = Value(0)
        for yp, yt in zip(ypred, ytrue):
            loss += (yp - yt) ** 2
        return loss
        
        
class NN:
    def __init__(self):
        self.MLP = None
        self.L = Loss()
    def build_model(self, in_out_shape, intermediary_layers, activations = [], activation_locations = []):
        self.MLP = MLP(in_out_shape[0], intermediary_layers + [in_out_shape[1]], activations, activation_locations)
    
    def train(self, inputs, outputs, epochs, lr = 0.01):
        
        self.parameters = self.MLP.get_parameters()
        
        for e in range(epochs):
                ypreds = [self.MLP(input_) for input_ in inputs]
                self.MLP.zero_grad()
                mse = sum([self.L.MSE(ypreds[i], outputs[i]) for i in range(len(ypreds))])/Value(len(ypreds))
                mse.grad = 1.0
                mse.backward()
                print("Weight: ",self.MLP.layers[0].neurons[0].weights[0].data)
                print("Grad: ",self.MLP.layers[0].neurons[0].weights[0].grad)
                for i in range(len(self.parameters)):
                    self.parameters[i].data -= lr * self.parameters[i].grad
    
                #lr = lr*math.exp(-e)
                print(f"MSE: {mse.data}\n")
    def predict(self, input_):
        return self.MLP(input_)
                
