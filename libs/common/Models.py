import random as r;
import math;

class MODEL:
    def __init__(self):
        self.bid    : int;
        self.fid    : int;

        self.epoch  : int;
        self.epochs : int;

        self.error  : float;
        self.errors : list;
        self.error_ders: list;

        self.a = 0.4;
        self.slopes = [];
    
        self.b1 = 0.9;
        self.b2 = 0.999;
        self.e= pow(10, -8);
        self.d      : int;
    
    def SetEpochs(self, epochs: int):
        self.epochs = epochs;
    def SetLearning(self, learning: float):
        self.a = learning;

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
        self.vdw      = 0;
        self.sdw      = 0;
        self.vdw_c    : float;
        self.sdw_c    : float;

        self.b        = 0;
        self.vdb      = 0;
        self.sdb      = 0;
        self.vdb_c    : float;
        self.sdb_c    : float;
    
        self.acts     : list;
        self.act_ders : list;

        self.derivs   : list;
        self.derivs_w : list

    def Predict(self, inputs: list):
        predictions = [ (self.w * input) + self.b  for input in inputs];
        activations = None;

        if(self.activation == "tanh"):
            activations= [ Tanh.Get(p) for p in predictions ];
            self.act_ders= [ 1 - pow(p,2) for p in activations];
        elif(self.activation == "relu"):
            activations= [ ReLU.Get(p) for p in predictions ];
            self.act_ders = [ 0 if p <= 0 else 1 for p in activations];
        elif(self.activation == "sigmoid"):
            activations = [ Sigmoid.Get(p) for p in predictions ];
            self.act_ders = [ p * (1- p) for p in activations ];
        else:
            self.acts= predictions;
        self.acts = activations;
    
    def OptimizeVars(self, ders: list, model: MODEL, isbias: bool):
        j= sum(ders) / model.d;
        j_pow= sum([pow(x, 2) for x in ders]) / model.d;

        if not isbias:
            old_vdw    = model.b1 * self.vdw + (1 - model.b1) * j;
            self.vdw   = old_vdw;
            vdw_c      = self.vdw / (1 - pow(model.b1, model.d));

            old_sdw    = model.b2 * self.sdw + (1 - model.b2) * j_pow;
            self.sdw   = old_sdw;
            sdw_c      = self.sdw / (1 - pow(model.b2, model.d));

            tmp_w      = self.w - model.a * vdw_c / (math.sqrt(sdw_c) + model.e);
            self.w     = tmp_w;

        else:
            old_vdb    = model.b1 * self.vdb + (1 - model.b1) * j;
            self.vdb   = old_vdb;
            vdb_c      = self.vdb / (1 - pow(model.b1, model.d));
    
            old_sdb    = model.b2 * self.sdb + (1 - model.b2) * j_pow;
            self.sdb   = old_sdb;
            sdb_c      = self.sdb / (1 - pow(model.b2, model.d));
    
            tmp_b      = self.w - model.a * vdb_c / (math.sqrt(sdb_c) + model.e);
            self.b     = tmp_b;
    
    def UpdateW(self, model : MODEL, active_optimizer: bool):
        tmp_w = 0;
        if (active_optimizer):
            pass;
        else:
            tmp_w  = self.w - model.a * sum(self.derivs_w) / model.d;
            self.w = tmp_w;
    
    def UpdateB(self, model: MODEL, active_optimizer: bool):
        tmp_b = 0;
        if(active_optimizer):
            pass;
        else:
            tmp_b  = self.b - model.a * sum(self.derivs) / model.d;
            self.b = tmp_b;

    def CalcDerivs(self, error_deriv_before: list, respect_to_input: list):
        self.derivs     = [ ed * ad for ed, ad in zip(error_deriv_before, self.act_ders)];
        self.derivs_w   = [ d * i for d,i in zip(self.derivs, respect_to_input) ];