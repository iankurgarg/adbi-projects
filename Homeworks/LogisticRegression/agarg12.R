library(readxl)
library(reshape)
library(dummies)
library(qcc)

set.seed(10)

generate_pivot_table <- function (pivot_table) {
  thresh = 0.05
  pivot_table['merge'] = pivot_table[1]
  len = dim(pivot_table)[1]  
  for (i in 1:(len-1)) {
    for (j in (i+1):len) {
      if ( j <= len) {
        if (abs(pivot_table[i,2] - pivot_table[j,2]) < thresh) {
          pivot_table[j,3] = pivot_table[i,3]
        }
      }
    }
  }
  return (pivot_table)
}


#---------------Input---------------
input_data = read_excel('eBayAuctions.xls')

train_size = 0.60
train_indices = sample(seq_len(nrow(input_data)), size = floor(train_size*nrow(input_data)))

train_data = input_data[train_indices,]
test_data = input_data[-train_indices,]

train_copy = train_data

#---------Generating Pivot Tables---------
input.melt = melt(train_data, id.vars = c(1,2,4,5), measure.vars = 8)
pivot_table1 = cast(input.melt, currency ~ variable, mean)
pivot_table2 = cast(input.melt, Category ~ variable, mean)
pivot_table3 = cast(input.melt, endDay ~ variable, mean)
pivot_table4 = cast(input.melt, Duration ~ variable, mean)

#----------Using Pivot Tables to merge categories---------
pivot_table = generate_pivot_table(pivot_table1)
len = dim(pivot_table)[1]
for (i in 1:len) {
  train_data[train_data['currency']==pivot_table[i,1], 'currency'] = pivot_table[i,3]
  test_data[test_data['currency']==pivot_table[i,1], 'currency'] = pivot_table[i,3]
}

pivot_table = generate_pivot_table(pivot_table2)
len = dim(pivot_table)[1]
for (i in 1:len) {
  train_data[train_data['Category']==pivot_table[i,1], 'Category'] = pivot_table[i,3]
  test_data[test_data['Category']==pivot_table[i,1], 'Category'] = pivot_table[i,3]
}

pivot_table = generate_pivot_table(pivot_table3)
len = dim(pivot_table)[1]
for (i in 1:len) {
  train_data[train_data['endDay']==pivot_table[i,1], 'endDay'] = pivot_table[i,3]
  test_data[test_data['endDay']==pivot_table[i,1], 'endDay'] = pivot_table[i,3]
}

pivot_table = generate_pivot_table(pivot_table4)
len = dim(pivot_table)[1]
for (i in 1:len) {
  train_data[train_data['Duration']==pivot_table[i,1], 'Duration'] = pivot_table[i,3]
  test_data[test_data['Duration']==pivot_table[i,1], 'Duration'] = pivot_table[i,3]
}


#---------Creating Dummy Variables----------------
#cols = c("Category","currency", "endDay", "Duration");
#for (col in cols) {
#  i = 0;
#  for (val in unique(train_data[col])[,1]) {
#    i = i+1;
#  }
#  j = 0;
#  for (val in unique(train_data[col])[,1]) {
#    if (j < i-1) {
#      train_data[paste(col, val, sep="_")] = ifelse(train_data[col] == val, 1, 0)
#      test_data[paste(col, val, sep="_")] = ifelse(test_data[col] == val, 1, 0)
#      j = j+1
#    }
#  }
#}

#cols_indices = match(cols, names(train_data))
#train_data = train_data[-cols_indices]
#test_data = test_data[-cols_indices]

#Commented part includes code to do this manually. It works.

train_data = dummy.data.frame(train_data)

#-----------------Fitting the Models---------------

fit.all <- glm(`Competitive?` ~.,family=binomial(link='logit'),data=train_data, control = list(maxit = 500))
coefs = fit.all$coefficients


#----- Getting the predictor with the highest regression coefficient------------
m = coefs[1]
mi = 1;
for (i in 2:length(coefs)) {
  op1 = abs(as.numeric(coefs[i]))
  if (!is.na(op1) && op1 > abs(as.numeric(m))) {
    m = coefs[i]
    mi = i
  }
}
m_name = m
for (i in names(train_data)) {
  if (grepl(i, names(fit.all$coefficients)[mi]))
    m_name = i
}
print(m_name)
subset = c("Competitive?", m_name)

fit.single = glm(`Competitive?` ~., family=binomial(link='logit'), data=train_data[subset])

#---------Getting Statistically significant predictors-----
significance_level = 0.05

coefs = summary(fit.all)$coefficients

significant_predictors = coefs[coefs[,4] < significance_level,]

#---------------Reduced Model---------------------
m_name = c("Competitive?")
for (i in names(train_data)) {
  for (s in names(significant_predictors[,1])) {
    if (grepl(i, s)) {
      m_name = c(m_name, i)
    }
  }
}
m_name = unique(m_name)

fit.reduced = glm(`Competitive?` ~., family=binomial(link='logit'), data=train_data[m_name])

#-----------Comparing both models------------

anova(fit.reduced, fit.all, test='Chisq')

#-----------Checking for Over Dispersion------------

s=rep(length(train_data$`Competitive?`), length(train_data$`Competitive?`))
qcc.overdispersion.test(train_data$`Competitive?`, size=s, type="binomial")


