import os;
from libs.common.DatasetGenerator import DemoDataset;

from libs.Nodes import Node;
from libs.Model import Model;
from libs.Layer import Layer;

os.system("cls");

Dataset  = DemoDataset(100000, 128, 2, True);
to_train = Dataset.batches[: -8]
to_eval  = Dataset.batches[-8 :]

model   = Model(1);
model.Error = 10;

Layer1 = Layer(1, 2, 0);
Layer2 = Layer(2, 1, 1);
for model.EpochNO in range(10000):
    for model.BatchID, batch in enumerate(to_train):
        inputs = [x.input for x in batch];
        targets = [x.target for x in batch];
        model.D = len(batch);

        Layer1.NodesTrain(inputs)
        Layer2.NodesTrain(Layer1.NodeActivations);

        model.GetError(Layer2.NodeActivations, targets)

        Layer2.NodesGetJs(model.ErrorDerivatives, Layer1.NodeActivations);
        Layer1.NodesGetJs(Layer2.NodeJs, inputs);
        
        Layer2.NodesUpdate(model);
        Layer1.NodesUpdate(model);

        print(f"{model.EpochNO+1, model.Error}", end="\r")
        pass
