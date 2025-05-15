from matplotlib import pyplot as plt;
import random as r;
import math as m;

data = [r.random() for _ in range(32)];
targets = [x * 2 for x in data];
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

for epoch in range(epochs):
    predictions: list[float] = [Predict(x) for x in data];
    error = sum([m.pow(p-t,2) for p,t in zip(predictions, targets)])/len(targets);
    error_i = [2*(p-t) for p,t in zip(predictions, targets)]
    wi = [i * g for i,g in zip(data, error_i)];

    wgrad: float = sum(wi)
    ac_w+= m.pow(wgrad,2);
    adjlrw = lr / m.sqrt(ac_w + e)

    bgrad: float = sum(error_i) 
    ac_b+= m.pow(bgrad,2);
    adjlrb = lr / m.sqrt(ac_b + e);
    
    if isAdagrad:
        w -= adjlrw * wgrad;
        b -= adjlrb * bgrad;
    else:
        w -= lr * wgrad;
        b -= lr * bgrad;

    xx.append(ac_w);
    yy.append(ac_b);
    aa.append(error);
    print(error);
pass 
plt.plot(xx)
plt.plot(yy)
plt.plot(aa)
plt.show();
