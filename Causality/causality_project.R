# Load the libraries 
library(vars)
library(urca)
library(pcalg)
# To install pcalg library you may first need to execute the following commands:
# source("https://bioconductor.org/biocLite.R")
# biocLite("graph")
# biocLite("RBGL")

# Read the input data 
input = read.csv('data.csv')
# Build a VAR model 
# Select the lag order using the Schwarz Information Criterion with a maximum lag of 10
# see ?VARSelect to find the optimal number of lags and use it as input to VAR()
lags = VARselect(input, type = "const")$selection[3]
varModel = VAR(input, p = lags)

# Extract the residuals from the VAR model 
# see ?residuals
res = residuals(varModel)

# Check for stationarity using the Augmented Dickey-Fuller test 
# see ?ur.df
summary(ur.df(res[,'Move']))
summary(ur.df(res[,'RPRICE']))
summary(ur.df(res[,'MPRICE']))

# For all three variables
# Significant test-statistic (more negative than the critical values). So, reject Null  hypothesis
# hence, the residuals are stationary

# Check whether the variables follow a Gaussian distribution  
# see ?ks.test
ks.test(res[,'Move'], 'pnorm')
ks.test(res[,'RPRICE'], 'pnorm')
ks.test(res[,'MPRICE'], 'pnorm')

# Low p values - significant. Hence reject NULL Hypothesis. Therefore, the residuals are not normally distributed

# Write the residuals to a csv file to build causal graphs using Tetrad software
write.csv(res, file = 'residuals.csv', row.names = FALSE)

# OR Run the PC and LiNGAM algorithm in R as follows,
# see ?pc and ?LINGAM 

# PC Algorithm
n = nrow(res)
V = colnames(res)
pc.fit = pc(suffStat = list(C=cor(res), n=n), indepTest = gaussCItest, alpha=0.1, labels = V)
require(Rgraphviz)
plot(pc.fit, main="CPDAG from PC algorithm")
# LiNGAM Algorithm
lingam.fit = lingam(res)

d = dim(lingam.fit$Bpruned)

require(igraph)

# transpose because the lingam.fit returns the adjacency matrix 
# such that [i,j] is edge from j to i
edL = t(lingam.fit$Bpruned)
colnames(edL) <- V
rownames(edL) <- V

g = graph.adjacency(edL, add.rownames = TRUE)
plot(g)
# Reference: for graph.adjacency() function, following link was used for understanding
# https://www.r-bloggers.com/graph-from-sparse-adjacency-matrix/
