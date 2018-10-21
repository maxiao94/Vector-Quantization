# Vector-Quantization / Cluster Analysis
K-Means &amp; K-Medians Clustering using Euclidean Distance &amp; Manhattan Distance

[K-Means](https://en.wikipedia.org/wiki/K-means_clustering) and [K-Medians](https://en.wikipedia.org/wiki/K-medians_clustering) have many applications in data science and cluster analysis (under the correct assumptions). 

In the code, an initial centroid guess is input such that a dictionary of centroid information is updated iteratively; the loop breaks when the centroids from the previous iteration remain unchanged in the current iteration. In the event that the centroids do not converge, the loop will not break. The points belonging to the determined respective clusters are color-coordinated to tell them apart. 

The image files correspond to the 3 examples at the bottom of the script. For each example, there is an initial picture and final picture. The initial picture shows the initial centroid guesses (each uniquely colored) against all of the data points (colored black). The final picture color-coordinates the points of each cluster; the data points are semi-transparent and the centroids are opaque. 

For the coloring scheme, one must input an iterable of colors to be used; this can be a string (like 'rbg') or normalized instances from a colormap. 

Python 3.5.2

Module Versions used:

--> numpy==1.15.0

--> matplotlib==2.2.2

