from AE_single_neuron import *


def test1():
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     
     Example:

                 +------+     a
       u1  +---->|      |           *-----------*
                 |  +=  |---------->|           |
       u2  +---->|      |           |           |
                 +------+           |           |
                                    |    *=     |------->  b
                                    |           |
       u3  +----------------------->|           |
                                    |           |
                                    *-----------*

     u1, u2, u3, a, b are Units
     += and *= are Gates
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Calculate forward and backward for this two gates circuit with
     a sigmoid activation in the output layer.
    """

    # init inputs
    u1 = Unit(5.0)
    u2 = Unit(-2.0)
    u3 = Unit(-4.0)

    # init output
    circuit_output = None

    assert u1.value == 5 and u2.value == -2 and u3.value == -4
    assert u1.grad == 1.0 and u2.grad == 1.0 and u3.grad == 1.0

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Load inputs:')
    print('Input 1 is {} | input 2 is {} | input 3 is {}'.format(u1.value, u2.value, u3.value))

    a = addGate(u1, u2)
    b = multiplyGate(a, u3)
    assert a.addends[0] is u1 and a.addends[1] is u2
    assert a.inputs == (u1.value, u2.value) and b.inputs == (a.value, u3.value)
    assert a.value == 3 and b.value == -12

    # the final output of the circuit and the starting gradient for the backprop
    c = sigmoidGate(b)

    print('---------------------------------------')
    print('Feed-forward:')
    print('Gate a results in {} | Gate b results in {} ' 
          '| Gate c results in {}'.format(a.value, b.value, c.value))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    circuit_output = c  

    def backprop():
        """Update all the values using the gradient of each function"""
        #print(circuit_output, circuit_output.multipliers)
        
        circuit_output.backward()

        b.backward()
        
        a.backward()

    backprop()

    def descent():
        """Update the forward pass with gradient descent"""
        step_size = 0.01
        u1.value += step_size * -u1.grad
        u2.value += step_size * -u2.grad
        u3.value += step_size * -u3.grad

    descent()

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('New Feed-Forward after first iteration of Gradient Descent:')
    print('Input 1 is {} | input 2 is {} | input 3 is {}'.format(u1.value, u2.value, u3.value))

    a = addGate(u1, u2)
    b = multiplyGate(a, u3)
    c = sigmoidGate(b)

    print('---------------------------------------')
    print('Feed-forward:')
    print('Gate a results in {} | Gate b results in {} ' 
          '| Gate c results in {}'.format(a.value, b.value, c.value))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    # gradient descent on the circuit made it more performant
    assert c.value > circuit_output.value

test1()


def test2():
    """
    Calculate a more complex circuit:
      (a * x) + (b * y) + c
    """
    # create placholders for inputs and gates
    global a, b, c, x, y
    global mulg0, mulg1, addg0, addg1, sg0 
    
    def feed_forward(*args):
        global a, b, c, x, y
        if not args:
            # create input units
            a = Unit(1.0)
            b = Unit(2.0)
            c = Unit(-3.0)
            x = Unit(-1.0)
            y = Unit(3.0)
        else:
            a, b, c, x, y = tuple(a for a in args)
        # feed forward
        global mulg0, mulg1, addg0, addg1, sg0 
        mulg0 = multiplyGate(a, x)
        mulg1 = multiplyGate(b, y)
        addg0 = addGate(mulg0.output, mulg1.output)
        addg1 = addGate(addg0.output, c)
        sg0 = sigmoidGate(addg1.output)

        return sg0.output
        
    def backprop():
        global mulg0, mulg1, addg0, addg1, sg0 
        sg0.backward()
        print('sg0.grad', sg0.grad)
        addg1.backward()
        print('addg1.grad', addg1.grad)
        addg0.backward()
        print('addg0.grad', addg0.grad)
        mulg1.backward()
        print('mulg1.grad', mulg1.grad)
        mulg0.backward()
        print('mulg0.grad', mulg0.grad)

        return

    def descent():
        global a, b, c, x, y
        import numpy as np
        step_size = -0.01
        a.value += step_size * a.grad; # assert np.isclose(np.array([a.value]), np.array([-0.105]))
        b.value += step_size * b.grad # b.grad is 0.315
        c.value += step_size * c.grad # c.grad is 0.105
        x.value += step_size * x.grad # x.grad is 0.105
        y.value += step_size * y.grad # y.grad is 0.210

        return 

    # 1st iteration
    circuit_output = feed_forward()
    print('1', circuit_output.value)
    backprop()
    descent()

    # 2nd iteration
    circuit_output = feed_forward(a, b, c, x, y)
    print('2', circuit_output.value)
    backprop()
    descent()



test2()