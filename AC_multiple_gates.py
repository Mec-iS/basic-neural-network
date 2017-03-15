"""
'A single extra multiplication will turn a single (useless gate)
 into a cog in the complex machine that is an entire neural network.'
"""

from AA_a_gate import forward_multiply_gate

def forward_add_gate(x, y):
    """
    A gate that performs addition of two addends.
    """
    return x + y

def forward_circuit(x, y, z):
    """
    Define a circuit with three inputs, an addition gate and a
     multiplying gate.
    """
    q = forward_add_gate(x, y)
    f = forward_multiply_gate(q, z)
    return f

x, y, z = -2, 5, -4
f = forward_circuit(x, y, z)  # return -12
