set.seed(25)

n = 20
prior.mean = 4
prior.sd = 0.8

likelihood.mean = 6
likelihood.sd = 1.5

post.mean = 5.70099668
post.sd = sqrt(0.09568106)

prior.X = sort(rnorm (n, prior.mean, prior.sd))
prior.dist = dnorm(prior.X, prior.mean, prior.sd)

likelihood.X = sort(rnorm (n, likelihood.mean, likelihood.sd))
likelihood.dist = dnorm(likelihood.X, likelihood.mean, likelihood.sd)

post.X = sort(rnorm(n, post.mean, post.sd))
post.dist = dnorm(post.X, post.mean, post.sd)

x1 = min(c(min(prior.X), min(likelihood.X), min(post.X))) - 1
x2 = max(c(max(prior.X), max(likelihood.X), max(post.X))) + 1

y1 = min(c(min(prior.dist), min(likelihood.dist), min(post.dist))) - 1
y2 = max(c(max(prior.dist), max(likelihood.dist), max(post.dist))) + 1

plot(prior.X, prior.dist, type='l', xlim=c(x1,x2), ylim=c(y1,y2), col='green', xlab="X", ylab="Density")
par(new=TRUE)
plot(likelihood.X, likelihood.dist, type='l', xlim=c(x1,x2), ylim=c(y1,y2), col='blue', xlab="X", ylab="Density")
par(new=TRUE)
plot(post.X, post.dist, type='l', xlim=c(x1,x2), ylim=c(y1,y2), col='red', xlab="X", ylab="Density")

legend(x2-2.5,y2,c('Prior','Likelihood', 'Posterior'), lty=c(1,1,1), lwd=c(2.5,2.5, 2.5),col=c('green','blue', 'red'))

