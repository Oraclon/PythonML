from matplotlib import pyplot as plt;
import random as r;
import math as m;

data = [r.random() for _ in range(32)];
targets = [ 1 if x > .5 else 0 for x in data];
epochs: int = 1000;

w = r.random() - .4;
b = r.random() - .4;

ac_w = 0;
ac_b = 0;
e: float = m.pow(10, -8);
lr: float = 0.4;
error = 0;

xx= []
yy= []
aa= []
isAdagrad: bool = True;
def Predict(x:float)->float:
    return w * x + b;

def Activate(x:float)->(float,float):
    act: float = 1 / (1 + m.exp(-x));
    der: float = act * (1 - act);
    return (act,der);

for epoch in range(epochs):
    predictions: list[float] = [Predict(x) for x in data];
    acts: list[(float, float)] = [Activate(x) for x in predictions];
    activations: list[float] = [x[0] for x in acts];
    derivatives: list[float] = [x[1] for x in acts];

    error = sum([m.pow(a-t,2) for a,t in zip(activations, targets)]) / len(targets);
    g_error = [2*(a-t) for a,t in zip(activations, targets)];

    g_bias: list[float] = [ ge * d for ge,d in zip(g_error, derivatives) ]
    g_slope: list[float] = [ i * g for i,g in zip(data, g_bias)]

    sgrad = sum(g_slope) / len(g_slope);
    ac_w += m.pow(sgrad, 2);
    cwlr: float = lr / m.sqrt(ac_w + e);

    bgrad = sum(g_bias) / len(g_bias);
    ac_b += m.pow(bgrad,2);
    cblr: float = lr / m.sqrt(ac_b + e);

    w -= cwlr * sgrad;
    b -= cblr * bgrad;

    pass
pass 
plt.plot(xx)
plt.plot(yy)
plt.plot(aa)
plt.show();
