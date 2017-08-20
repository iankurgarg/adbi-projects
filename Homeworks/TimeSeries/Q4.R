require(fpp)
data(ukcars)
plot(ukcars, ylab = "Production, thousands of cars")

#1. Model Selected was ANA
etsuk = ets(ukcars)
accuracy(etsuk)[,'RMSE']
etsuk$aicc

#2. RMSE of all the three models almost similar which suggests that all the three mdoels
# are quite similar. Even the AICC comparison shows that all three models are similar.
stlFit <- stl(ukcars, s.window = "periodic")
plot(stlFit)
adjusted <- seasadj(stlFit)
fcastHoltDamp = holt(adjusted, damped=TRUE, h = 8)

dampHoltRMSE = sqrt(mean(((fcastHoltDamp$fitted + stlFit$time.series[,"seasonal"]) - ukcars)^2))
dampHoltRMSE
fcastHoltDamp$model$aicc

fcastHolt = holt(adjusted, h = 8)

holtRMSE = sqrt(mean(((fcastHolt$fitted + stlFit$time.series[,"seasonal"]) - ukcars)^2))
holtRMSE
fcastHolt$model$aicc

#3. 
# The following plots are for the forecasts by the three models
# Based on these plots we can se that there isn't much difference between the 
# three forecasts. So, as such selecting a best out of all doesn't make sense.
# Still, damped Holt's forecasts seem more reasonable as instead of just 
# assming single linear trend for the whole data it forecasts based on other factors as well.
# Also, forecasts of ets and damped linear holt's model are quite close

plot(ukcars, xlim = c(1977, 2008), main="damped holt")
lines(fcastHoltDamp$mean + 
        stlFit$time.series[2:9,"seasonal"], 
      col = "red", lwd = 2)

plot(ukcars, xlim = c(1977, 2008), main="linear holt")
lines(fcastHolt$mean + stlFit$time.series[2:9,"seasonal"], 
      col = "red", lwd = 2)

plot(ukcars, xlim = c(1977, 2008), main="ets")
lines(forecast(etsuk)$mean, col = "red", lwd = 2)

