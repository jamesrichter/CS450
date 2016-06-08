
library(igraph)
library(sna)

# Reading an edge list
#mommyEdgesTwitter = read.csv("MommyMentionMap.csv", header = FALSE)
twitterGraph = graph.data.frame(mommyEdgesTwitter)

# Plot a simple graph using Fruchterman-Reingold layout
# Assuming a graph "g"
layout1 = layout.fruchterman.reingold(g)
layout2 = layout.auto(g)
#plot(g, layout=layout2)

# Assuming a graph "g"

# Compute density
graph.density(g)

# Compute centralization measures for all nodes in the graph
dc = centralization.degree(g)
cc = centralization.closeness(g)
bc = centralization.betweenness(g)
V(g)[which.max(bc$res)]
V(g)[which.max(cc$res)]
V(g)[which.max(dc$res)]
max(bc$res)
max(cc$res)
max(dc$res)

# For each of the centralization measures, the value for every
# node is stored in a list $res (e.g., dc$res)
