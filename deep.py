import os;
import math;
import numpy as np;
import random as r;
from libs.common.DatasetGenerator import DemoDataset;

os.system("clear");

class Model:
    def __init__(self,  loss_function: int):

        #Loss Function
        # 0 : Mean
        # 1 : LogLoss

        self.Learning         = 0.4;
        self.B1               = 0.9;
        self.B2               = 0.999;
        self.D                : int;
        self.E                = pow(10, -8);
        self.Loss             = loss_function;
        self.Vdw              = 0;
        self.Sdw              = 0;
        self.Vdb              = 0;
        self.Sdb              = 0;

        self.Error            : float;
        self.Errors           : list;
        self.ErrorDerivatives : list;

        self.BatchID          : int;
        self.EpochNO          : int;

    def GetError(self, latest_activation: list, targets: list):
        
        def UpdateError():
            if self.Loss == 0:
                self.Error = sum(self.Errors) / (2 * len(self.Errors));
            elif self.Loss == 1:
                self.Error = sum(self.Errors) / len(self.Errors);

        if self.Loss == 0:
            self.Errors           = [ pow( a-t ) for a,t in zip(latest_activation, targets) ];
            self.ErrorDerivatives = [ 2*( a-t ) for a,t in zip(latest_activation, targets) ];
        
        elif self.Loss == 1:
            self.Errors           = [ -math.log(a) if t == 1 else -math.log(1-a) for a,t in zip(latest_activation, targets) ];
            self.ErrorDerivatives = [ -1 / a if t == 1 else 1 / (1 - a) for a,t in zip(latest_activation, targets) ];

        UpdateError();

class Node:
    def __init__(self, activation: int, layer_id: int):

        # Activation
        # 0 : Tahn
        # 1 : Sigmoid

        self.NodeID              = layer_id;
        self.ActivationDerivs    : list;
        self.Activations         : list;
        self.W                   = r.random() - .4;
        self.B                   = 0;
        self.Js                  : list;
        self.JWs                 : list;
        self.SelectedActivation  = activation;
        self.Vdw                 = 0;
        self.Sdw                 = 0;
        self.Vdb                 = 0;
        self.Sdb                 = 0;
    
    def NodeEval(self, input: float) -> float:
        prediction = self.W * input + self.B;
        activation = 0;

        if self.SelectedActivation == 0:
            activation = math.tanh(prediction);
        elif self.SelectedActivation == 1:
            return round(1 / (1 + math.exp(-prediction)));

        return activation;

    def NodePredict(self, inputs: list):

        acts     = [];
        act_ders = [];

        for input in inputs:
            prediction = self.W * input + self.B;
            
            if self.SelectedActivation == 0:
                activation = math.tanh(prediction)
                acts.append(activation);
                act_ders.append(1 - pow(activation, 2));
            elif self.SelectedActivation == 1:
                activation = 1/ (1 + math.exp(-prediction));
                acts.append(activation);
                act_ders.append(activation * (1- activation));
        
        self.Activations      = acts;
        self.ActivationDerivs = act_ders;
    
    def GetNodeDerivs(self,  previous_derivative: list, respect_to : list):
        self.Js  = [ pd * ad for pd, ad in zip(previous_derivative, self.ActivationDerivs) ];
        self.JWs = [ j * rt for j, rt in zip(self.Js, respect_to) ];

    def NodeUpdate(self, model: Model):
        self.OptimizeW(model, self.JWs);
        self.OptimizeB(model, self.Js);

    def OptimizeW(self, model: Model, derivs: list):
        jw  = sum(derivs) / model.D;
        jw2 = sum([pow(x,2) for x in derivs]) / model.D;

        old_vdw  = model.B1 * self.Vdw + (1 - model.B1) * jw;
        self.Vdw = old_vdw;
        old_sdw  = model.B2 * self.Sdw + (1 - model.B2) * jw2;
        self.Sdw = old_sdw;

        vdw_c     = self.Vdw / (1- pow(model.B1, model.D));
        sdw_c     = self.Sdw / (1- pow(model.B2, model.D));

        tmpw      = self.W - model.Learning * vdw_c / (math.sqrt(sdw_c)+ model.E);
        self.W    = tmpw;

    def OptimizeB(self, model: Model, derivs: list):
        jb  = sum(derivs) / model.D;
        jb2 = sum([pow(x, 2) for x in derivs]) / model.D;

        old_vdb   = model.B1 * self.Vdb + (1 - model.B1) * jb;
        self.Vdb  = old_vdb;
        old_sdb   = model.B2 * self.Sdb + (1 - model.B2) * jb2;
        self.Sdb  = old_sdb;

        vdb_c     = self.Vdb / (1 - pow(model.B1, model.D));
        sdb_c     = self.Sdb / (1 - pow(model.B2, model.D));

        tmpb      = self.B - model.Learning * vdb_c / (math.sqrt(sdb_c) + model.E);
        self.b  = tmpb;

Dataset = DemoDataset(300000, 1024, 2, True);
to_train = Dataset.batches[: -8]
to_eval  = Dataset.batches[-8 :]

model   = Model(1);

node1   = Node(0, 1);
node2   = Node(0, 2);
node3   = Node(1, 3);


for model.EpochNO in range(10000):
    for model.BatchID, batch in enumerate(to_train):

        inputs = [x.input for x in batch];
        targets = [x.target for x in batch];
        model.D = len(batch);

        node1.NodePredict(inputs);
        node2.NodePredict(node1.Activations);
        node3.NodePredict(node2.Activations);

        model.GetError(node3.Activations, targets);

        node3.GetNodeDerivs(model.ErrorDerivatives, node2.Activations)
        node2.GetNodeDerivs(node3.Js, node1.Activations);
        node1.GetNodeDerivs(node2.Js, inputs);

        node3.NodeUpdate(model);
        node2.NodeUpdate(model);
        node1.NodeUpdate(model);

        if(model.Error <= pow(10, -4)):
            break;
    if(model.Error <= pow(10, -4)):
        print(model.EpochNO, model.BatchID);
        break;

def Eval(batches: list):
    c,w,a= 0,0, 0;
    for batch in batches:
        for item in batch:
            pred1 = node1.NodeEval(item.input);
            pred2 = node2.NodeEval(pred1)
            pred  = node3.NodeEval(pred2);
            if(pred == item.target):
                c+= 1;
            else:
                w+= 1;
    print(f"Correct: [{c}] Wrong: [{w}] Acc: [{ round((c*100)/(c+w), 4) }%]");

Eval(to_eval);