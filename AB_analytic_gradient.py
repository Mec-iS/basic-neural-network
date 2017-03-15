from AA_a_gate import forward_multiply_gate

x, y = -2, 3
out = forward_multiply_gate(x, y)  # returns -6

# a gradient of an operand of a multiplication is,
# by definition of a derivative, the other operand:
# f(x, y) = xy  
x_gradient = y
y_gradient = x

# we want to approach the minimum using a minimal quantity
step_size = 0.01
# the direction of this approaching is given by the gradient
x += step_size * x_gradient
y += step_size * y_gradient
# this results into a 'better' (in this case higher) result
# (in this case we use a simple 'cost function': higher number
# means better performance)
out_new = forward_multiply_gate(x, y)  # returns -5.87
