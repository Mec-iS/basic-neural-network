"""
A neuron with a signmoid activation function
"""
import numpy as np

from AA_a_gate import forward_multiply_gate
from AC_multiple_gates import forward_add_gate


class Unit:
    """
    A simple object to store units' states.

    quote: 'every Unit corresponds to a wire in the diagrams'
    Each edge on the diagram is a Unit.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     
     Example:

                 +------+     q
        x  +---->|      |           *-----------*
                 |  +=  |---------->|           |
        y  +---->|      |           |           |
                 +------+           |           |
                                    |    *=     |------->  f
                                    |           |
        z  +----------------------->|           |
                                    |           |
                                    *-----------*

     x, y, z, q, f are Units
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    def __init__(self, value, grad=0.0):
        # value computed in the forward pass
        self.value = value 
        # the derivative of circuit output w.r.t this unit, computed in backward pass
        self.grad = grad


class Gate:
    def __init__(self, operation, *args):
        # store the function that characterize the gate
        self.operation = operation
        # store the tuple with the numerical values of the inputs
        self.inputs = tuple(a.value for a in args)

        # check that all inputs are Units or Gates
        for a in args:
            if not (isinstance(a, Unit) or isinstance(a, Gate)):
                raise TypeError() 
        
        # init output
        self.output = Unit(self.value, 0.0)

    @property
    def value(self):
        return self.forward()

    def forward():
        raise NotImplemented()

    def backward():
        raise NotImplemented()


class multiplyGate(Gate):
    """
    A gate for the binary operation multiplication
    """
    def __init__(self, *args):
        # initialize super-class to be a multiplying gate
        super(multiplyGate, self).__init__(
                forward_multiply_gate, *args
        )

        self.multipliers = args

    def forward(self):
        return self.operation(
            *tuple(m for m in self.inputs)
        )

    def backward(self):
        # quote: 'take the gradient in output unit and chain it with the
        # local gradients, which we derived for multiply gate before
        # then write those gradients to those Units.'
        list_ = list(self.multipliers)
        for m in list_:
            # update the gradient for each unit 
            m.grad += m.value * self.output.grad

        self.multipliers = tuple(list_)
    
class addGate(Gate):
    """
    A gate for the binary operation addition
    """
    def __init__(self, *args):
        # initialize super-class to be an adding gate
        super(addGate, self).__init__(
                forward_add_gate, *args
        )
        # store the adding Units
        self.addends = args
        
        # set the gradient to the fixed amount 1.0
        self.output.grad = 1.0

    def forward(self):
        return self.operation(
            *tuple(a for a in self.inputs)
        )

    def backward(self):
        # quote: 'take the gradient in output unit and chain it with the
        # local gradients, which we derived for multiply gate before
        # then write those gradients to those Units.'
        list_ = list(self.addends)
        for m in list_:
            # update the gradient for each unit 
            m.grad += m.value * self.output.grad

        self.addends = tuple(list_)

class sigmoidGate(Gate):
    """
    A gate for the sigmoid activation function
    """
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    sig_deriv = lambda x: x * (1 - x)

    def __init__(self, *args):
        # initialize super-class to be an adding gate
        super(addGate, self).__init__(
                self.sigmoid, *args
        )
        self.input_ = args[0]

    def forward(self):
        return self.operation(
            self.inputs[0]
        )

    def backward(self):
        # quote: 'take the gradient in output unit and chain it with the
        # local gradients, which we derived for multiply gate before
        # then write those gradients to those Units.'
        self.output.grad = self.sig_deriv(self.value) 
