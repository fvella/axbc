# Approximated Betwenness Centrality AxBC
Multi-heuristics Optimization for approximating BC on large-scale graphs on Multi-GPUs

## Short description and usage

AxBC provide an fully-distributed approximated algorithm for computing Betwenness Centrality on large-scale graphs.

The algorithm is based sampling technique which consist on selecting nodes (pivots) to run Brandes' algorthm[^1]. 
The number of pivots can be statically assigned (e.g., 10% of the nodes of the graph) or determined according to a stop criterion based on the approximated score determied by the probability estimator (adaptive sampling[^2]). 

### Description of the heuristics. 
The algorithm provides two kind of heuristics
- static: based on the property of the graph (e.g., degree distribution)
- dynamic: based on the computation of the Brandes of the previous itarations  

1. uniform sampling (adaptive sampling)
2. Close to High-degree vertex
3. Betwenness-based
4. Lazy approach based on delta of the previous step
5. LCC based

## Example of usage
Description of the most important options 
```
./<axbc> -p <row procs X column procs> -f <path to the graph.txt> -n <number of vertices of the graphs> -c <adaptative sempling ON/OFF> -x <number of vertices to check for the approximation for the adaptative sampling> -z <heuristic> -N <static number of rounds>
```
-c is seleted the value -N represent the maximum number of Brandes iterations. 

### Herustics 
Select a strategy between 0 and 6.


0. Uniform sampling 
1. High-Degree Neighborhood: first randomly select an vertex with an high outdegree. Then it randomly selects a neighbours of a such vertex. 
2. Local Clustering Coefficient: compute a distribution based on LCC score. This heuristics is particularly expesive on large graphs.   
3. BC-based sampling strategy: randomly select the next vertex based on current BC score distribution (dynamic long-memory heuristic)
4. Delta sampling strategy: randomly select the next vertex based on current delta distribution (dynamic short-memory heuristic). 
5. BC-based inverse sampling strategy: 1-BC based sampling strategy. 
6. Roulette wheel: randomly select one of the heustistics for each iteration of the algorithm. 

Other options are:

-H 1: implements 1-degree reduction. Pre-process the graph by removing vertices with 1 neighbour (1-degree reduction).

-m: shared-memory implementation. 

-U: make the graph undirected.

-a: analyze the degree.

-d: dump the graph.

[^1]:Brandes, U., 2001. A faster algorithm for betweenness centrality. Journal of
mathematical sociology, 25(2), pp.163-177.
[^2]:Bader, D.A., Kintali, S., Madduri, K. and Mihail, M., 2007, December.
Approximating betweenness centrality. In International Workshop on Algorithms and Models
for the Web-Graph (pp. 124-137). Springer, Berlin, Heidelberg.
