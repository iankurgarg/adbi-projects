# Homework 4
# Submitted by: Ankur Garg, agarg12@ncsu.edu
require(fpp)

#1. The data has seasonal component which changes slowly over years, 
# no specific trend and stable variance.
plot(elecequip)

#2. The data is decomposed below
d = stl(elecequip, s.window = 'periodic')

#3. Yes, the data is seasonal. So, it is seasonaly adjusted.
elecSeasAdj = seasadj(d)
plot(elecSeasAdj)


#4. There is no need to do variance stablization as seen from the data plot. 
# We can also see that by plotting the data after BoxCox
# transformation, it is quite similar to the original seasonal adjusted data
#lambda = BoxCox.lambda(elecSeasAdj)
#elecVar = BoxCox(elecSeasAdj, lambda)
#plot(elecVar)

#5. The data is not stationary as Acf decreases slowly.
# Also, we can check stationarity using adf.test since we already have removed seasonality.
# adf.test returns pval of 0.4699, which suggests that Null Hypothesis can't be rejected.
# So, that means, the data is non stationary
Acf(elecSeasAdj)
adf.test(elecSeasAdj)


#6. Yes, after differencing the non-stationary data has been converted to stationary
# as can be seen from the fact that the Acf now decreases to zero quite fast
# Also, using adf.test, we see that the pval is very low (less than 0.1)
# which suggests, we reject Null hypothesis. So, that means, the data is stationary now.
nd = ndiffs(elecSeasAdj)
elecDiff = diff(elecSeasAdj, differences = nd)
Acf(elecDiff)
adf.test(elecDiff)

#7. The model returns the following values: p = 3, d = 0, q = 1
model = auto.arima(elecDiff)
model$aicc

#8. AICc for: ARIMA(4,0,0) = 981.36, ARIMA(3,0,0) = 981.66, ARIMA(2,0,0) = 998.87
# Best Value is for original auto arima model
model1 = Arima(elecDiff, order=c(4,0,0))
model1$aicc
model2 = Arima(elecDiff, order=c(3,0,0))
model2$aicc
model3 = Arima(elecDiff, order=c(2,0,0))
model3$aicc

bestmodel = model

#9. Acf suggests that the residuals are white noise
# So does the Box.test(), p-val is close to 1 which suggests that the null hypothesis can't be
# rejected. So, the residuals are white noise
errors = residuals(bestmodel)
Acf(errors)
Box.test(errors)

#10. The model is proper. Forecast is plotted below
plot(forecast(bestmodel))
