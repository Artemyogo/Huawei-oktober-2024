Formal statementYou are given a roadmap data as a planar graph $G$. Each node of the graph is given with its geographical coordinates (latitude and longitude). It is guaranteed that no pair of given nodes is on a distance more than 100 kilometers from each other while all nodes have pairwise distinct coordinates.Each road is given as a sequence of node IDs. It is a polyline on a sphere without self-intersections or self-loops. It is guaranteed that if two road polylines share a common point on a sphere, this point is necessary one of the given nodes of the roadmap.The edges of graph $G$ are formed by union of edges (segments between consecutive nodes) of all roads.The given roadmap graph $G$ is not guaranteed to be connected.We call a geographical point internal point of $G$ if:
It does not coincide with any node of the roadmap;
    It does not belong to any road of the roadmap;
    It does not belong to the outer side of the roadmap planar graph, that is, it belongs to an area enclosed by a finite set of roads.
 Additionally, you are given a set of points called users, each one described by a its geographical coordinates. For every user it is guaranteed that it is an internal point of $G$. You should find $k$ sequences of node IDs of $G$ satisfying the following:
 Each sequence represents a simple cycle of $G$. This implies that it represents a polygon on a 3D-sphere, or a spherical polygon.
     Every internal point of $G$ should be an internal point of exactly one of $k$ polygons.
     Every polygon should contain at least 512 and at most 1024 users.
 
	
		
Note that $k$ is not given in the input, but is part of your output and can vary. The score of your solution does not explicitly depend on the value of $k$.
