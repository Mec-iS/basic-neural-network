import unittest

from AE_single_neuron import *

class TestGate(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.a = a = Unit(1.0)
        self.b = b = Unit(2.0)
        self.c = c = Unit(-3.0)
        self.x = x = Unit(-1.0)
        self.y = y = Unit(3.0)

        self.mulg0 = multiplyGate(a, x)
        self.mulg1 = multiplyGate(b, y)
        self.addg0 = addGate(self.mulg0.output, self.mulg1.output)
        self.addg1 = addGate(self.addg0.output, c)
        self.sg0 = sigmoidGate(self.addg1.output)

        print('Testing Gates')

    def test_init(self):
        # multiply gates store input units as `multipliers`
        assert self.mulg0.multipliers == tuple([self.a, self.x])
        assert self.mulg1.multipliers == tuple([self.b, self.y])


         

if __name__ == '__main__':
    unittest.main()