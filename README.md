# Approximated Betwenness Centrality AxBC
Multi-heuristics Optimization for approximating BC on large-scale graphs on Multi-GPUs

## Short description and usage

AxBC provide an fully-distributed approximated algorithm for computing Betwenness Centrality on large-scale graphs.

The algorithm is based sampling technique which consist on selecting nodes (pivots) to run Brandes' algorthm. 
The number of pivots can be statically assigned (e.g., 10% of the nodes of the graph) or determined according to a stop criterion based on the approximated score determied by the probability estimator (adaptive sampling). 

### Description of the heuristics. 
The algorithm provides two kind of heuristics
- static: based on the property of the graph (e.g., degree distribution)
- dynamic: based on the computation of the Brandes of the previous itarations  

1. uniform sampling (Bader'implementation)
2. Close to High-degree vertex
3. Betwenness-based
3. Lazy approach based on delta of the previous step
4. LCC based

### Example of usage for shared-memory system

./<axbc> -p 1x1 

