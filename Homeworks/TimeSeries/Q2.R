data = usnetelec
plot(data)
Acf(data)
# Based on the plot of the data, no need for BoxCox transformation
# Also, there is no seasonality in the data as seen from the plots
# and also by using decompose, (it gives an errors message that the
# data is not seasonal)
# decompose(data)
# But based on Acf, we can see the data is not stationary as Acf decreases slowly
# To remove the trend, we can use differencing
nd = ndiffs(data)
if (nd > 0) {
  data2 = diff(data, differences = nd)
}
#This transforms the data to stationary. Running ndiffs again on data2 returns 0.
# We can also check this using plots
plot(data2)
Acf(data2)
# Large pval - null hypothesis not rejected. - stationary and non seasonal
kpss.test(data2)


data = usgdp
plot(data)
Acf(data)
# Based on the plot of the data, no need for BoxCox transformation
# Also, there is no seasonality in the data
# But based on Acf, we can see the data is not stationary as Acf decreases slowly
# To remove the trend, we can use differencing
nd = ndiffs(data)
if (nd > 0) {
  data2 = diff(data, differences = nd)
}
# This transforms the data to stationary. Running ndiffs again on data2 returns 0.
# We can also check this using plots
plot(data2)
Acf(data2)
# Large pval - null hypothesis not rejected. - stationary and non seasonal
kpss.test(data2)


data = mcopper
plot(data)
Acf(data)
# Based on the plot of the data, we need BoxCox transformation for variance stablization
lambda = BoxCox.lambda(data)
data = BoxCox(data, lambda)
plot(data)
plot(decompose(data))
# Also, there is seasonality in the data (seen in the decompsoe() plot), 
# which can be removed by using seasadj
data = seasadj(decompose(data))
Acf(data)
# But based on Acf, we can see the data is not stationary as Acf decreases slowly
# To remove the trend, we can use differencing
nd = ndiffs(data)
if (nd > 0) {
  data2 = diff(data, differences = nd)
}
# This transforms the data to stationary. Running ndiffs again on data2 returns 0.
# We can also check this using plots
plot(data2)
Acf(data2)
# Large pval - null hypothesis not rejected. - stationary and non seasonal
kpss.test(data2)


data = enplanements
plot(data)
Acf(data)
# Based on the plot of the data, we need BoxCox transformation for variance stablization
lambda = BoxCox.lambda(data)
data = BoxCox(data, lambda)
plot(data)
# Also, there is seasonality in the data, which can be removed by using seasadj
data = seasadj(decompose(data))
plot(data)
Acf(data)
# Based on Acf, we can see the data is not stationary as Acf decreases slowly
# To remove the trend, we can use differencing
nd = ndiffs(data)
if (nd > 0) {
  data2 = diff(data, differences = nd)
}
# This transforms the data to stationary. Running ndiffs again on data2 returns 0.
# We can also check this using plots
plot(data2)
Acf(data2)
# Large pval - null hypothesis not rejected. - stationary and non seasonal
kpss.test(data2)


data = visitors
plot(data)
Acf(data)
# Based on the plot of the data, we need BoxCox transformation for variance stablization
lambda = BoxCox.lambda(data)
data = BoxCox(data, lambda)
plot(data)
# Also, there is seasonality in the data, which can be removed by differencing
# To remove seasonality, we can use seasadj() function
data = seasadj(decompose(data))
# Based on Acf, we can see the data is not stationary as Acf decreases slowly
# To remove the trend, we can use differencing
nd = ndiffs(data)
if (nd > 0) {
  data2 = diff(data, differences = nd)
}
# This transforms the data to stationary. Running ndiffs again on data2 returns 0.
# We can also check this using plots
plot(data2)
Acf(data2)
# Large pval - null hypothesis not rejected. - stationary and non seasonal
kpss.test(data2)
