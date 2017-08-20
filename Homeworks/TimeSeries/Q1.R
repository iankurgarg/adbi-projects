require(fpp) 
data(dj)
plot(dj)
acf(dj)
Box.test(dj, lag=10, fitdf=0, type="Lj")

nd = ndiffs(dj)

dj2 = diff(dj, differences = nd)
plot(dj2)
Acf(dj2)
Box.test(dj2, lag=10, fitdf=0, type="Lj")

# Initially, the pval returned by the Ljung test was almost close to zero, 
# meaning the X-sqaured value was greater than the critical value and hence
# the null hypothesis was rejected and it was not white noise

# But after the differencing, it was observed that the pval for Ljung test returned was quite 
# greater than 0.05, and hence in the 95% confidence interval, it would not be 
# significant and hence null hypothesis cant be rejected. So, the data after 
# differencing is white noise.

# Based on Acf plots, we see that the initially the Acf plot was decreasing slowly
# hence signifying that the data wasn't stationary
# But after the differencing the Acf plot is decreasing to zero quite fast
# and hence signifying that the data has now become stationary.

kpss.test(dj)
kpss.test(dj2)

# Stationarity of the data can also be checked using kpss test since the data is non-seasonal
# Intially the pval of kpss.test was almost close to zero. So null hypothesis is rejected.
# Hence, the data was not stationary
# But after differencing, the kpss.test returns pval greater than 0.05. Hence, null hypothesis
# can't be rejected. So, data has been made stationary.