# Routing in CV-networks

This code is used to generate covariance matrices of Gaussian cluster states associated with graphs of CZ gates and simulate an entanglement routing protocol. The complex networks topologies are simulated in python through the python module NetworkX that can be downloaded here https://pypi.org/project/networkx/.
In the code you can already find the topologies of the Erdos-Renyi (ER), Barabasi-Albert (BA), Watts-Strogatz (WS), Internet Autonomous System (AS) and Protein-protein interaction (PP) models. However other structures can be easily implemented with networkx https://networkx.org/documentation/stable/reference/generators.html. Besides the networks' parameters, one can tune the number of nodes in the network, the squeezing parameters and strength of the CZ-gates. Since the latter always provide some squeezing, the squeezing parameters are normally set to 0.
After generating the resource CV quantum state, we simulate three entanglement routing protocols in which every node measures its mode in the P or Q quadratures except the two nodes, Alice A and Bob B, who want to communicate. The first protocol, called Routing, seeks for the shortest paths between A and B in the network, and it checks one by one the paths that should be measured in P to increase the entanglement and measures in Q all the rest. The second protocol, Shortest, takes only one of the shortest paths and measures it in P, while the rest is measured in Q. Finally, the protocol AllP, measures all the terminal nodes (namely the nodes with degree 1) in Q and the rest in P.

# Output

For each network realisation, A is chosen to be the node with the largest degree and all the three protocols are applied to each node of the network in order to compare their performances. The first figure produced shows the negativity produced by the three different protocols applied to each node of the network.  The nodes are labeled in order of distance and of number of paths connecting to Alice.  The blue, orange and green stems represent the negativity of the final pair after the Routing,Shortest and AllP protocols respectively, while the dashed lines represent the mean value for all the nodes.  The color of the marker indicates the distance of the node from A and the grey columns represent the ratio of paths that improved the entanglement in Routing.
The second figure shows the graph of the network, where the  nodes  are  again  sorted  by  distance  and  number  of parallel paths and the size of each node is proportional to its degree.   The node with highest negativity and all the paths that improved its entanglement are highlighted with red thick lines. The node that has the highest difference in the negativity produced by the Routing and Shortest protocols and all the paths that improved its entanglement are highlighted with green thick lines. The last figure shows the subgraph of these two last nodes and all the paths connecting them to A.

# Python

This code was written to run with python3.

# Associated research

The code lies at the basis of the numerical results in the following publication: arXiv:2108.08176
