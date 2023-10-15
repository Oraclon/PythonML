from libs.Nodes import Node;
import numpy as np;

class Layer:
    def __init__(self, layer_id: int, total_nodes: int, node_activation: int):
        """
            Layer Activation:
            - 0) Tahn
            - 1) Sigmoid
        """

        self.TotalNodes      = total_nodes;
        self.Nodes           : list;
        self.LayerId         = layer_id;
        self.NodeActivations : list;
        self.NodeJs          : list;

        self.__BuildNodes(total_nodes, node_activation);

    def __BuildNodes(self, total_nodes: int, node_activation: int):
        nodes = [];
        for i in range(total_nodes):
            node = Node(node_activation, self.LayerId);
            nodes.append(node);
        self.Nodes = nodes;

    def NodesTrain(self, inputs: list):
        """
        NodeRetModel:
            Activation      : float;
            ActivationDeriv : float;
        """
        data= []
        node: Node;
        for node in self.Nodes:
            node_activations= []
            for input in inputs:
                node_answer = node.NodeLayerPredict(input);
                node_activations.append(node_answer.Activation);
            data.append(node_activations);
        self.NodeActivations= list(zip(*data));
    
    def NodesGetJs(self, previous_activations :list, respect_to :list):
        pp = [sum(x) for x in previous_activations]
        node: Node;
        for node in self.Nodes:
            node.GetNodeDerivs(previous_activations, respect_to);
            pass
    

# -(w1) 
#      \
#       (Node) --> (P) --> (A)
#      /
# -(W2) 