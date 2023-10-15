import math;
import numpy as np;
import random as r;
from libs.Model import Model;

class NodeRetModel:
    Activation      : float;
    ActivationDeriv : float;

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
        self.NodeIsUpdated       = False;
    
    def __PrepareNode(self, feature_len: int):
        w   = [];
        vdw = [];
        sdw = [];

        for _ in range(feature_len):
            w.append(r.random() - .5);
            vdw.append(0);
            sdw.append(0);
        
        self.W   = w;
        self.Vdw = vdw;
        self.Sdw = sdw;
    
        self.NodeIsUpdated = True;

    def NodeLayerPredict(self, input: float)-> NodeRetModel:

        if not self.NodeIsUpdated:
            self.__PrepareNode(len(input));

        prediction = np.dot(self.W , input) + self.B;
        ret_model  = NodeRetModel();
        if self.SelectedActivation  == 0:
            ret_model.Activation      = math.tanh(prediction);
            ret_model.ActivationDeriv = 1 - pow(ret_model.Activation, 2);
        elif self.SelectedActivation == 1:
            ret_model.Activation      = 1 / (1 + math.exp(-prediction));
            ret_model.ActivationDeriv = ret_model.Activation * (1 - ret_model.Activation);
        return ret_model;

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
