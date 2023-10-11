from libs.common.Models import MODEL, SLOPE;
from libs.common.DatasetGenerator import DemoDataset;

import numpy as np;
import random as r;
import math;
import os;

os.system("clear");

class BCD:
    def Predict(self, model: MODEL, input: float):
        z1 = model.s1.w * input + model.s1.b;
        a1 = math.tanh(z1);
        z2 = model.s2.w * a1 + model.s2.b;
        a2 = math.tanh(z2);
        z3 = model.s3.w * a2 + model.s3.b;
        return round(1 / (1 + math.exp(-z3)));

    def Train(self, model: MODEL, batches: list, optimizer: bool):

        model.s1 = SLOPE(activation= 'tanh');
        model.s2 = SLOPE(activation= 'tanh');
        model.s3 = SLOPE(activation= 'sigmoid');

        for model.epoch in range(1, model.epochs):
            for model.bid, batch in enumerate(batches):
                model.d = len(batch);
                inputs  = [x.input for x in batch];
                targets = [x.target for x in batch];
                
                model.s1.Predict(inputs);
                model.s2.Predict(model.s1.predictions);
                model.s3.Predict(model.s2.predictions);

                model.errors= [-math.log(p) if t == 1 else -math.log(1 - p) for p,t in zip(model.s3.predictions, targets)];
                model.error= sum(model.errors) / model.d;

                er_ders = [ -1 / p if t == 1 else 1 / (1 - p) for p,t in zip(model.s3.predictions, targets) ];

                dz3     = [ ed * pd for ed, pd in zip(er_ders, model.s3.pred_derivs) ];
                dw3     = [ e * p for e,p in zip(dz3, model.s2.predictions) ];

                dz2     = [ e * pd for e, pd in zip(dz3, model.s2.pred_derivs) ];
                dw2     = [ e * a for e,a in zip(dz2, model.s1.predictions) ];

                dz1     = [ e * pd for e,pd in zip(dz2, model.s1.pred_derivs) ];
                dw1     = [ e * i for e,i in zip(dz1, inputs) ];

                model.s3.UpdateW(model, dw3, optimizer);
                model.s2.UpdateW(model, dw2, optimizer);
                model.s1.UpdateW(model, dw1, optimizer);

                model.s3.UpdateB(model, dz3, optimizer);
                model.s2.UpdateB(model, dz2, optimizer);
                model.s1.UpdateB(model, dz1, optimizer);
            
                if(model.error <= pow(10, -3)):
                    break;
            
            print(model.error);
            pass

            if(model.error <= pow(10, -3)):
                break;
            
            

model    = MODEL();
model.SetEpochs(10000);
model.SetLearning(.2);

data     = DemoDataset(140000, 64, 4, False);
batches  = data.batches;
to_train = batches[:-10];
to_eval  = batches[-10:];

bcd      = BCD();
bcd.Train(model= model, batches= to_train, optimizer= True);

correct, wrong, acc= 0,0,0;
for batch in to_eval:
    for item in batch:
        pred = bcd.Predict(model, item.input[0]);
        targ = item.target;
        if pred == targ:
            correct += 1;
        else:
            wrong += 1;

print(f'{model.epoch} : {model.bid} | {correct} : {wrong} | {((correct * 100))/(correct + wrong)}%')
pass