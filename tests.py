from AE_single_neuron import *

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

print('---------------------------------------')
print('Feed-forward:')
print('Gate a results in {} | Gate b results in {}'.format(a.value, b.value))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

# the final output of the circuit and the starting gradient for the backprop
circuit_output = b  # value=-12 grad=1.0

print(circuit_output, circuit_output.multipliers)
circuit_output.backward()
print(circuit_output, circuit_output.multipliers)
print(tuple(str(m) for m in circuit_output.multipliers))
#backprop1 = multiplyGate(circuit_output).output.grad 

