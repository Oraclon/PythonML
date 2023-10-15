from libs.Nodes import Node;
from libs.Model import Model;
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
        layer_id    = self.LayerId;
        nodes_total = self.TotalNodes;
        pass
    
    def NodesGetJs(self, previous_activations :list, respect_to :list):
        pass;

    def NodesUpdate(self, model: Model):
        node: Node;
        for node in self.Nodes:
            node.NodeUpdate(model);

    

# -(w1) 
#      \
#       (Node) --> (P) --> (A)
#      /
# -(W2) 