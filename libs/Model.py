import math;

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