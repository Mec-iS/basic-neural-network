def forward_multiply_gate(*args):
    r = 1
    for a in args:
        r *= a
    return r

print(forward_multiply_gate(-2, 3))  # returns -6
