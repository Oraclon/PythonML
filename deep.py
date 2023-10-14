import os;
import math;
import numpy as np;
import random as r;
from libs.common.DatasetGenerator import DemoDataset;

os.system("clear");

class RetModel:
    def __init__(self):
        self.Activation      : float;
        self.ActivationDeriv : float;

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

        self.w1               = r.random();
        self.b1               = 0

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

        self.NodeID           = layer_id;
        self.ActivationDerivs : list;
        self.Activations      : list;
        self.Activation       = activation;
        self.W                = r.random() - .4;
        self.B                = 0;
        self.J                : list;
        self.JW               : list;
    
    def NodePredict(self, input: float) -> RetModel:
        
        Prediction = self.W * input + self.B;
        Ret        = RetModel();

        if self.Activation == 0:
            Ret.Activation      = math.tanh(Prediction);
            Ret.ActivationDeriv = 1 - pow(Ret.Activation,2);
        if self.Activation == 1:
            Ret.Activation      = 1 / (1 + math.exp(- Prediction));
            Ret.ActivationDeriv = Ret.Activation * (1 - Ret.Activation);
        
        return Ret;

    def NodeSetActivationDerivs(self, activation_derivs: list):
        self.ActivationDerivs = activation_derivs;

    def NodeSetActivations(self, activations: list):
        self.Activations = activations;

    def NodeCalcDerivatives(self, prev_derivs: list, resp_to: list):
        self.J = [ d * a for d,a in zip(prev_derivs, self.ActivationDerivs) ];
        self.Jw = [ d * i for d, i in zip(self.J, resp_to) ];
    
    def NodeUpdate(self, model: Model):

        tmp_w = self.W - model.Learning * sum(self.Jw) / model.D;
        self.W = tmp_w;

        tmp_b= self.B - model.Learning * sum(self.J) / model.D;
        self.B= tmp_b;

class Layer:
    def __init__(self, nodes_size: int, node_activation: int):
        '''
            Node Activations:
            - 0 : Tanh
            - 1 : Sigmoid
        '''
        
        self.NodeActivationType   = node_activation
        self.TotalNodes           = nodes_size;
        self.Nodes                = [];
        self.NodeActivations      : list;
        self.NodeActivationDerivs : list;
        self.NodeJS               : list;
    
        self.BuildNodes();
    
    def BuildNodes(self):
        for i in range(self.TotalNodes):
            node= Node(self.NodeActivationType, i+1);
            self.Nodes.append(node);
        pass
    
    def TrainNodes(self, inputs: list):
        node: Node;

        node_activations       = []
        node_activation_derivs = []

        for input in inputs:
            for node_id, node in enumerate(self.Nodes):
                node_result = node.NodePredict(input)
                node_activations.append(node_result.Activation);
                node_activation_derivs.append(node_result.ActivationDeriv);
        
        self.NodeActivations      = node_activations;
        self.NodeActivationDerivs = node_activation_derivs;
        pass

    def GetNodeDerivatives(self, previous_layer_derivs: list, respect_to: list):
        node_derivatives = [];
        node: Node;
        for node in self.Nodes:
            node.NodeCalcDerivatives(previous_layer_derivs, respect_to);
            node_derivatives.append(node.J);
        
        self.JS = node_derivatives;

Dataset = DemoDataset(300000, 256, 2, True);
to_train = Dataset.batches[: -8]
to_eval  = Dataset.batches[-8 :]

model   = Model(1);

Layer1 = Layer(1, 0);
Layer2 = Layer(1, 1);

w1, w2 = 0.4, 0.6;
b1, b2 = 0  ,   0,

def Predict(w, b, i):
    return w * i + b;
def Sigmoid(prediction):
    return 1 / (1 + math.exp(-prediction));

def OptimizeW(model: Model, derivs: list):
    jw  = sum(derivs) / model.D;
    jw2 = sum([pow(x,2) for x in derivs]) / model.D;

    old_vdw   = model.B1 * model.Vdw + (1 - model.B1) * jw;
    model.Vdw = old_vdw;
    old_sdw   = model.B2 * model.Sdw + (1 - model.B2) * jw2;
    model.Sdw = old_sdw;

    vdw_c     = model.Vdw / (1- pow(model.B1, model.D));
    sdw_c     = model.Sdw / (1- pow(model.B2, model.D));

    tmpw      = model.w1 - model.Learning * vdw_c / (math.sqrt(sdw_c)+ model.E);
    model.w1  = tmpw;

def OptimizeB(model: Model, derivs: list):
    jb  = sum(derivs) / model.D;
    jb2 = sum([pow(x, 2) for x in derivs]) / model.D;

    old_vdb   = model.B1 * model.Vdb + (1 - model.B1) * jb;
    model.Vdb = old_vdb;
    old_sdb   = model.B2 * model.Sdb + (1 - model.B2) * jb2;
    model.Sdb = old_sdb;

    vdb_c     = model.Vdb / (1 - pow(model.B1, model.D));
    sdb_c     = model.Sdb / (1 - pow(model.B2, model.D));

    tmpb      = model.b1 - model.Learning * vdb_c / (math.sqrt(sdb_c) + model.E);
    model.b1  = tmpb;

for _ in range(10000):
    for batch in to_train:

        inputs = [x.input for x in batch];
        targets = [x.target for x in batch];
        model.D = len(batch);

        pred = [Predict(model.w1, model.b1, input) for input in inputs]
        act  = [Sigmoid(p) for p in pred]
        actD = [a * (1-a) for a in act];
    
        # errors= [-math.log(a) if t == 1 else -math.log(1-a) for a,t in zip(act, targets)];
        # errorsD = [-1 / a if t == 1 else 1/ (1-a) for a,t in zip(act, targets)];
        # model.Error = sum(errors) / model.D;
        model.GetError(act, targets)
    
        j2  = [ e * a for e,a in zip(model.ErrorDerivatives, actD) ];
        j2w = [ j * float(i) for j,i in zip(j2,  inputs) ];

        OptimizeW(model, j2w);
        OptimizeB(model, j2);

        if(model.Error <= pow(10, -4)):
            break;
    if(model.Error <= pow(10, -4)):
        break;
    print(_+1, model.Error, end='\r')

c,w,a= 0,0, 0;

for batch in to_eval:
    for item in batch:
        pred = round( Sigmoid(Predict(model.w1, model.b1, item.input)) )
        if(pred == item.target):
            c+= 1;
        else:
            w+= 1;
print(f"Correct: [{c}] Wrong: [{w}] Acc: [{ round((c*100)/(c+w), 4) }%]")