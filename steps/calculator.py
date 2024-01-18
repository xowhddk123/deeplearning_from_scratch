import numpy as np

# step01


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None  # step06
        self.creator = None

    def set_creator(self, func):
        self.creator = func  # step07

    def backward(self):
        funcs = [self.creator]  # 1.함수를 가져온다.
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output  # 함수의 입력과 출력을 가져온다.
            x.grad = f.backward(y.grad)
            if x.creator is not None:
                funcs.append(x.creator)

# step02


class Function:
    def __call__(self, input):
        x = input.data  # 데이터를 꺼낸다.
        y = self.forward(x)  # 구체적인 계산은 forward method에서 한다.
        output = Variable(y)  # variable 형태로 되돌린다.
        output.set_creator(self)  # 출력 변수에 창조자를 설정한다.
        self.input = input  # 입력변수를 저장
        self.output = output  # 출력도 저장한다.
        return output

    def forward(self, x):
        raise NotImplementedError()  # 구현 안되어 있으면 에러 표시

    def backward(self, gy):  # step06
        raise NotImplementedError()  # 구현 안되어 있으면 에러 표시

# step03


class Square(Function):
    def forward(self, x):
        y = x**2  # step06
        return y

    # step06
    def backward(self, gy):
        x = self.input.data
        gx = 2*x*gy
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

# step04


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data-y0.data)/(2*eps)


def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))
