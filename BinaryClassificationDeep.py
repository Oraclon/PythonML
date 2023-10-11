from libs.common.Models import MODEL, SLOPE, DVAL;
import random as r;
import math;

class BCD:
    def Train(model: MODEL, batches: list):

        model.s1 = SLOPE(activation= 'tahn');
        model.s2 = SLOPE(activation= 'tahn');
        model.s3 = SLOPE(activation= 'sigmoid');

        for model.epoch in range(model.epochs):
            for model.bid, batch in enumerate(batches):
                model.d = len(batch);
                inputs  = [x.input[0] for x in batch];
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
            print(model.error);
            pass

class DemoDataset:
    def __init__(self, datasetsize: int, batchsize: int, commonvar: float, isbinary: bool):

        self.dataset = [];

        for i in range(1, datasetsize):
            dval = DVAL();
            dval.input  = [ i ];
            dval.target = i * commonvar if not isbinary else 0 if i <= datasetsize/commonvar else 1;
            self.dataset.append(dval);
        r.shuffle(self.dataset);