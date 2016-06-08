# first compute a distance matrix
distance = dist(as.matrix(data))

# now perform the clustering
hc = hclust(distance)

# finally, plot the dendrogram
plot(hc)
  
data_scaled = scale(data)

# Cluster into k=5 clusters:
myClusters = kmeans(data, 5)

# Summary of the clusters
summary(myClusters)

# Centers (mean values) of the clusters
myClusters$centers

# Cluster assignments
myClusters$cluster

# Within-cluster sum of squares and total sum of squares across clusters
myClusters$withinss
myClusters$tot.withinss


# Plotting a visual representation of k-means clusters
library(cluster)
clusplot(data, myClusters$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)

