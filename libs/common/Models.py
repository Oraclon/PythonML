import random as r;
import math;

class Tanh:
    def Get( prediction: float):
        return math.tanh(prediction);

class ReLU:
    def Get( prediction: float ):
        return max(0, prediction);

class Sigmoid:
    def Get( prediction: float ):
        return 1 / (1 + math.exp(- prediction));

class DVAL:
    input: list;
    target: float;

class SCALER:
    def __init__(self):
        self.m   : float;
        self.s   : float;
        self.min : float;
        self.max : float;

class SLOPE:
    def __init__(self, activation: str):

        self.activation = activation;

        self. w = r.random() - .5;
        self.vdw = 0;
        self.sdw = 0;

        self.b = 0;
        self.vdb = 0;
        self.sdb = 0;
    
        self.predictions: list;
        self.pred_derivs: list;
        self.derivs: list;

    def Predict(self, inputs: list):
        predictions = [ (self.w * input) + self.b  for input in inputs];
        activations = None;

        if(self.activation == "tanh"):
            activations= [ Tanh.Get(p) for p in predictions ];
        elif(self.activation == "relu"):
            activations= [ ReLU.Get(p) for p in predictions ];
        elif(self.activation == "sigmoid"):
            activations = [ Sigmoid.Get(p) for p in predictions ];
        else:
            self.predictions= predictions;
        self.predictions = activations;

class MODEL:
    def __init__(self):
        self.bid: int;
        self.fid: int;
        self.epoch: int;
        self.epochs: int;

        self.a = 0.4;
        self.slopes = [];
    
        self.b1 = 0.9;
        self.b2 = 0.999;
        self.e= pow(10, -8);
        self.d: int;
    
    def SetEpochs(self, epochs: int):
        self.epochs = epochs;
    def SetLearning(self, learning: float):
        self.a = learning;