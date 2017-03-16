import unittest

from AE_single_neuron import *

class TestUnit(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.a = Unit(1.0)
        self.b = Unit(2.0)
        self.c = Unit(-3.0)
        self.x = Unit(-1.0)
        self.y = Unit(3.0)

        print('Testing Units')

    def test_init(self):
        assert self.a.value == 1.0
        assert self.a.grad == 1.0

        assert self.b.value == 2.0
        assert self.b.grad == 1.0        

if __name__ == '__main__':
    unittest.main()