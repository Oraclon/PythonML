from libs.common.Models import DVAL, SCALER;
import random as r;
import numpy as np;
import math;

class DemoDataset:
    def __init__(self, datasetsize: int, batchsize: int, commonvar: float, isbinary: bool):

        self.dataset = [];
        self.scalers = [];
        self.batches = [];

        for i in range(1, datasetsize):
            dval = DVAL();
            dval.input  = [ i, -(i*100) ];
            dval.target = i * commonvar if not isbinary else 0 if i <= datasetsize/commonvar else 1;
            self.dataset.append(dval);
        r.shuffle(self.dataset);

        inputs= [x.input for x in self.dataset];
        inputsT= np.transpose(inputs);
        scaled= []
        for inp in inputsT:

            scaler= SCALER();
            scaler.m   = np.average(inp);
            scaler.s   = math.sqrt( sum( [ pow(x - scaler.m,2) / (len(inp) -1) for x in inp ]) );
            scaler.min = min(inp);
            scaler.max = max(inp);
            self.scalers.append(scaler);

            sc = [ (x - scaler.m) / scaler.s for x in inp ];

            scaled.append(sc);
        scT= np.transpose(scaled);

        for i, inp in enumerate(scT):
            self.dataset[i].input= inp;
        
        self.batches= [self.dataset[z : z + batchsize] for z in [x for x in range(0, len(self.dataset), batchsize)]];