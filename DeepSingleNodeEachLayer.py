import os;
from libs.common.DatasetGenerator import DemoDataset;

from libs.Nodes import Node;
from libs.Model import Model;
from libs.Layer import Layer;

os.system("cls");

Dataset = DemoDataset(1000000, 1024, 2, True);
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
        
        
        print(f'[Training] Epoch: {model.EpochNO} | Batch: {model.BatchID} | Error: {model.Error}', end="\r")

        if(model.Error <= pow(10, -2)):
            break;
    if(model.Error <= pow(10, -2)):
        os.system("cls")
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