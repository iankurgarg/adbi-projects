require(fpp)
#1. The holts models are as written below:
data(books)
hcp = holt(books[,'Paperback'], h=4)
forecast(hcp)

hch = holt(books[,'Hardcover'], h=4)
forecast(hch)



#2. Holt's method seems to given smaller SSE for both Paperback and Hardcover
# as compared to the SSE of the SES models.
sseHCP = sum(residuals(hcp)*residuals(hcp))
sseHCH = sum(residuals(hch)*residuals(hch))

# The data contains linear trend. So, to use simple exponential smoothing
# we first need to remove the trend from the data.

d1 = ndiffs(books[,'Paperback'])
deTrendPaperback = diff(books[,'Paperback'], differences = d1)
sesp = ses(deTrendPaperback, h=4)

d2 = ndiffs(books[,'Hardcover'])
deTrendHardcover = diff(books[,'Hardcover'], differences = d2)
sesh = ses(deTrendHardcover, h=4)

# To calculate the SSE for comparison with holt's method we need to add the 
# trend back to fitted values to calulcate SSE

fitted1 =  diffinv(sesp$fitted, xi=books[1,'Paperback'])
res1 = fitted1 - books[,'Paperback']
sseSESP = sum(res1^2)

fitted2 =  diffinv(sesh$fitted, xi=books[1,'Hardcover'])
res2 = fitted2 - books[,'Hardcover']
sseSESH = sum(res2^2)

# SSE Comparison shoes that the holt's method gives lower SSE for both Paperback and Hardcover
# data as compared to the simple exponential smoothing

#3. 
# Now for comparing the forecasts of simple exponential smoothing with holt's method
# we need to add the trend back to the ses forecasts as well.


f1 = diffinv(sesp$mean, xi=books[30,'Paperback'])
f2 = diffinv(sesh$mean, xi=books[30,'Hardcover'])

plot(books[,'Paperback'], xlim=c(0,35))
lines(hcp$mean, col='red')
lines(f1, col='blue')

plot(books[,'Hardcover'], xlim=c(0,35))
lines(hch$mean, col='red')
lines(f2, col='blue')

# Comparison between the forecasts of holt's method and simple exponential smoothing
# We see that the forecasts of holt's method are more reasonable as it is closer to the actual mean trend
# whereas the forecasts of the SES method are based on the last value in the data and are hence away 
# from the actual mean trend of the data.