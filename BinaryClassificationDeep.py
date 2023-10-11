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

    def Train(self, model: MODEL, batches: list):

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
                pr_ders = [ p * (1 - p) for p in model.s3.predictions ];

                dz3     = [ ed * pd for ed, pd in zip(er_ders, pr_ders) ];
                dw3     = sum([ e * p for e,p in zip(dz3, model.s2.predictions) ]) / model.d;
                db3     = sum(dz3) / model.d;

                dz2     = [ e * (1 - pow(a,2)) for e,a in zip(dz3, model.s2.predictions) ];
                dw2     = sum([ e * a for e,a in zip(dz2, model.s1.predictions) ]) / model.d;
                db2     = sum(dz2) / model.d;

                dz1     = [ e * (1 - pow(a,2)) for e,a in zip(dz2, model.s1.predictions) ];
                dw1     = sum([ e * i for e,i in zip(dz1, inputs) ]) / model.d;
                db1     = sum(dz1) / model.d

                tmp_w3     = model.s3.w - model.a * dw3;
                model.s3.w = tmp_w3;
                tmp_w2     = model.s2.w - model.a * dw2;
                model.s2.w = tmp_w2;
                tmp_w1     = model.s1.w - model.a * dw1;
                model.s1.w = tmp_w1;

                tmp_b3     = model.s3.b - model.a * db3;
                model.s3.b = tmp_b3;
                tmp_b2     = model.s2.b - model.a * db2;
                model.s2.b = tmp_b2;
                tmp_b1     = model.s1.b - model.a * db1;
                model.s1.b = tmp_b1;
            
                if(model.error <= pow(10, -2)):
                    break;
            if(model.error <= pow(10, -2)):
                break;
            
            print(model.error);
            pass

model    = MODEL();
model.SetEpochs(10000);
model.SetLearning(.4);

data     = DemoDataset(140000, 256, 2, True);
batches  = data.batches;
to_train = batches[:-10];
to_eval  = batches[-10:];

bcd      = BCD();
bcd.Train(model= model, batches= to_train);

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