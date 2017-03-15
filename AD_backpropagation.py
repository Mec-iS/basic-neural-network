"""
Using gates' derivatives/gradients to calculate adjustments
 to increase performance.
 (in this case the objective is just to get an hugher result)
"""

from AA_a_gate import forward_multiply_gate
from AC_multiple_gates import forward_add_gate

x, y, z = -2, 5, -4
q = forward_add_gate(x, y)  # returns 3
f = forward_multiply_gate(q, z)

### calculate derivatives (from last operation to first)
# the MULTIPLY gate with respect to its inputs
# wrt is short for "with respect to"
derivative_f_wrt_z = q
derivative_f_wrt_q = z   # returns 3 

# derivative of the ADD gate with respect to its inputs is just always 1
# quote: "this makes sense because to make the output of a single addition gate higher, 
# we expect a positive tug on both x and y, regardless of their values."
derivative_q_wrt_x = 1.0
derivative_q_wrt_y = 1.0

# chain rule for derivatives
# the resulting derivatives of f() are just the chained derivates of f() and q()
var derivative_f_wrt_x = derivative_f_wrt_q * derivative_q_wrt_x  # returns -4
var derivative_f_wrt_y = derivative_f_wrt_q * derivative_q_wrt_y  # returns -4

# GRADIENT with respect of x, y, z [-4, -4, 3]
gradient_f_wrt_xyz = [derivative_f_wrt_x, derivative_f_wrt_y, derivative_f_wrt_z]

# quote: "let the inputs respond to the force/tug" (gradient descent)
step_size = 0.01
x = x + step_size * derivative_f_wrt_x   # -2.04
y = y + step_size * derivative_f_wrt_y   # 4.96
z = z + step_size * derivative_f_wrt_z   # -3.97

# Our circuit now gives higher output (given the initial assumption, better performance):
q = forwardAddGate(x, y)   # q becomes 2.92
f = forwardMultiplyGate(q, z)    # output is -11.59