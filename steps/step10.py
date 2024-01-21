import unittest
from calculator import *


class SquareTest(unittest.TestCase):
    '''
    unit test 하는 방법
    '''

    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertAlmostEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)


unittest.main()
