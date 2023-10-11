from libs.common.Models import MODEL, SLOPE;
from libs.common.DatasetGenerator import DemoDataset;

import numpy as np;
import random as r;
import math;
import os;

os.system("clear");

class BCD:
    def Predict(self, model: MODEL, input: float):
        model.s1.Predict(input);
        model.s2.Predict(model.s1.acts);
        model.s3.Predict(model.s2.acts);
        return round(model.s3.acts[0]);

    def Errors(self, model: MODEL, activations: list, targets: list):
        errors           = [ -math.log(p) if t == 1 else -math.log(1 - p) for p,t in zip(activations, targets) ];
        model.error_ders = [ -1 / p if t == 1 else 1 / (1 - p)  for p,t in zip(activations, targets)];
        model.error      = sum(errors) / model.d
        return errors;

    def Train(self, model: MODEL, batches: list, optimizer: bool):

        model.s1 = SLOPE(activation= 'tanh');
        model.s2 = SLOPE(activation= 'tanh');
        model.s3 = SLOPE(activation= 'sigmoid');

        for model.epoch in range(1, model.epochs):
            for model.bid, batch in enumerate(batches):
                model.d = len(batch);
                inputs  = [ x.input[0] for x in batch ];
                targets = [ x.target for x in batch ];
                
                model.s1.Predict(inputs);
                model.s2.Predict(model.s1.acts);
                model.s3.Predict(model.s2.acts);

                model.errors = self.Errors(model, model.s3.acts, targets);

                model.s3.CalcDerivs(model.error_ders, model.s2.acts)
                model.s2.CalcDerivs(model.s3.derivs , model.s1.acts)
                model.s1.CalcDerivs(model.s2.derivs, inputs);

                model.s3.UpdateW(model, optimizer);
                model.s2.UpdateW(model, optimizer);
                model.s1.UpdateW(model, optimizer);

                model.s3.UpdateB(model, optimizer);
                model.s2.UpdateB(model, optimizer);
                model.s1.UpdateB(model, optimizer);
            
                if(model.error <= pow(10, -3)):
                    break;
            
            print(model.error);
            pass

            if(model.error <= pow(10, -3)):
                break;

model    = MODEL();
model.SetEpochs(10000);
model.SetLearning(.4);

data     = DemoDataset(400000, 512, 2, True);
batches  = data.batches;
to_train = batches[:-10];
to_eval  = batches[-10:];

bcd      = BCD();
bcd.Train(model= model, batches= to_train, optimizer= False);

correct, wrong, acc= 0,0,0;
for batch in to_eval:
    for item in batch:
        pred = bcd.Predict(model, item.input);
        targ = item.target;
        if pred == targ:
            correct += 1;
        else:
            wrong += 1;

print(f'{model.epoch} : {model.bid} | {correct} : {wrong} | {round(((correct * 100))/(correct + wrong), 4)}%')
pass