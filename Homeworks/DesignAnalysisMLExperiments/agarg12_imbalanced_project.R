# ========================================
# Multiple Hypothesis Testing
# Part 1: K-fold Cross-Validation Paired t-Test
# Part 2: Analysis of Variance (ANOVA) Test
# Part 3: Wilcoxon Signed Rank test
# ========================================

set.seed(10)

# Load the required R packages
require(cvTools)
require(C50)
require(kernlab)
require(e1071)
require(stats)
require(car)
require(gplots)
# **********************************************
# Part 1: K-fold Cross-Validation Paired t-Test
# *****************************************

# Load the iris data set
iris1 = read.csv('datasets/Iris_data.txt', header = FALSE)

# Randomize the data and perform 10-fold Cross-Validation
# See ?sample and ?cvFolds
input = sample(iris1)
folds = cvFolds(nrow(iris1), K = 10, type = 'random')

DTERROR = vector(length=10)
SVMERROR = vector(length=10)

for (i in 1:10) {
  # Use the training set to train a C5.0 decision tree and Support Vector Machine
  dtmodel = C5.0(V5 ~ ., data = input, subset = folds$subsets[folds$which != i])
  svmmodel = ksvm(V5 ~ ., data = input[folds$subsets[folds$which != i],])
  
  # Make predictions on the test set and calculate the error percentages made by both the trained models
  test = input[folds$subsets[folds$which == i], -which(names(input) %in% c("V5"))]
  predictions_dt = predict.C5.0(dtmodel, test)
  predictions_svm = predict(svmmodel, test)
  
  error_dt = 100*sum(input[folds$subsets[folds$which == i], 'V5'] != predictions_dt)/length(predictions_dt)
  error_svm = 100*sum(input[folds$subsets[folds$which == i], 'V5'] != predictions_svm)/length(predictions_svm)
  DTERROR[i] = error_dt
  SVMERROR[i] = error_svm
}

# Perform K-fold Cross-Validation Paired t-Test to compare the means of the two error percentages
t.test(DTERROR, SVMERROR, paired = TRUE)
# No statistical difference betweent he means

# *****************************************
# Part 2: Analysis of Variance (ANOVA) Test
# *****************************************

# Load the Breast Cancer data set 
breastCancer = read.csv('datasets/Wisconsin_Breast_Cancer_data.txt', header=FALSE)
# Randomize the data and perform 10-fold Cross-Validation
# See ?sample and ?cvFolds
input = sample(breastCancer[,-which(names(breastCancer) %in% c("V1"))])
folds = cvFolds(nrow(breastCancer), K=10, type='random')

# Use the training set to train following classifier algorithms
# 	1. C5.0 decision tree (see ?C5.0 in C50 package)
# 	2. Support Vector Machine (see ?ksvm in kernlab package)
# 	3. Naive Bayes	(?naiveBayes in e1071 package) 
# 	4. Logistic Regression (?glm in stats package) 

breastCancer[,'V2'] = factor(breastCancer[,'V2'])
errors_val_dt = c()
errors_val_svm = c()
errors_val_nb = c()
errors_val_lr = c()
errors_name = c()

for (i in 1:10) {
  train = input[folds$subsets[folds$which != i], ]
  test = input[folds$subsets[folds$which == i], ]
  test = test[complete.cases(test),]
  
  dtmodel = C5.0(V2 ~ ., data = train)
  svmmodel = ksvm(V2 ~ ., data = train)
  nbModel = naiveBayes(V2 ~ ., data = train)
  lrModel = glm(V2 ~ ., data = train, family=binomial, control = list(maxit = 50))
  
  # Make predictions on the test set and calculate the error percentages made by the trained models
  predictions_dt = predict.C5.0(dtmodel, test)
  predictions_svm = predict(svmmodel, test)
  predictions_nb = predict(nbModel, test)
  
  predictions_lr = predict.glm(lrModel, test[,-which(names(test) %in% c("V2"))], type='response')
  predictions_lr[is.na(predictions_lr)] = 0
  predictions_lr = replace(predictions_lr, predictions_lr>0.5, 1)
  predictions_lr = replace(predictions_lr, predictions_lr<0.5, 0)
  predictions_lr = replace(predictions_lr, NA, 0)
  predictions_lr = factor(predictions_lr)
  levels(predictions_lr)[levels(predictions_lr) == "0"] <- 'B'
  levels(predictions_lr)[levels(predictions_lr) == "1"] <- 'M'
  
  errors_val_dt = c(errors_val_dt, 100*sum(test[,'V2'] != predictions_dt)/length(predictions_dt))
  errors_val_svm = c(errors_val_svm, 100*sum(test[,'V2'] != predictions_svm)/length(predictions_svm))
  errors_val_nb = c(errors_val_nb, 100*sum(test[,'V2'] != predictions_nb)/length(predictions_nb))
  errors_val_lr = c(errors_val_lr, 100*sum(test[,'V2'] != predictions_lr)/length(predictions_lr))
}
errors_name = c(rep('DT', 10), rep('SVM', 10), rep('NB', 10), rep('LR', 10))
errors_val = c(errors_val_dt, errors_val_svm, errors_val_nb, errors_val_lr)

errors = data.frame(errors_name,errors_val)
colnames(errors) = c('Model', 'Error')
# Compare the performance of the different classifiers using ANOVA test (see ?aov)
fit = aov(Error ~ Model, errors)
summary(fit)

lm.fit = lm(Error ~ Model, data=errors)
qqPlot(lm.fit)
bartlett.test(Error ~ Model, data=errors)
outlierTest(fit)

plotmeans(Error ~ Model, data = errors)

# *****************************************
# Part 3: Wilcoxon Signed Rank test
# *****************************************

# Load the following data sets,
# 1. Iris
iris1 = read.csv('datasets/Iris_data.txt', header = FALSE)

# 2. Ecoli 
ecoli = read.csv('datasets/Ecoli_data.csv', header = FALSE)

# 3. Wisconsin Breast Cancer
breastCancer = read.csv('datasets/Wisconsin_Breast_Cancer_data.txt', header = FALSE)

# 4. Glass
glass = read.csv('datasets/Glass_data.txt', header = FALSE)

# 5. Yeast
yeast = read.csv('datasets/Yeast_data.csv', header = FALSE)

# Randomize the data and perform 10-fold Cross-Validation
# See ?sample and ?cvFolds
input1 = sample(iris1)
input2 = sample(ecoli[,-1])
input3 = sample(breastCancer[,-1])
input4 = sample(glass[,-1])
input4$V11 = factor(input4$V11)
input5 = sample(yeast[,-1])

folds1 = cvFolds(nrow(input1), K=10, type='random')
folds2 = cvFolds(nrow(input2), K=10, type='random')
folds3 = cvFolds(nrow(input3), K=10, type='random')
folds4 = cvFolds(nrow(input4), K=10, type='random')
folds5 = cvFolds(nrow(input5), K=10, type='random')

# Use the training set to train following classifier algorithms
# 	1. C5.0 decision tree (see ?C5.0 in C50 package)
# 	2. Support Vector Machine (see ?ksvm in kernlab package)


DTERROR = vector(length=5)
SVMERROR = vector(length=5)

for (i in 1:10) {
#---- DATASET 1 - iris
  # Use the training set to train a C5.0 decision tree and Support Vector Machine
  dtmodel = C5.0(V5 ~ ., data = input1, subset = folds1$subsets[folds1$which != i])
  svmmodel = ksvm(V5 ~ ., data = input1[folds1$subsets[folds1$which != i],])
  test = input1[folds1$subsets[folds1$which == i], ]
  
  # Make predictions on the test set and calculate the error percentages made by both the trained models
  test = test[, -which(names(test) %in% c("V5"))]
  predictions_dt = predict.C5.0(dtmodel, test)
  predictions_svm = predict(svmmodel, test)
  
  error_dt = 100*sum(input1[folds1$subsets[folds1$which == i], 'V5'] != predictions_dt)/length(predictions_dt)
  error_svm = 100*sum(input1[folds1$subsets[folds1$which == i], 'V5'] != predictions_svm)/length(predictions_svm)
  DTERROR[1] = DTERROR[1] + error_dt
  SVMERROR[1] = SVMERROR[1] + error_svm
  
#-----DATASET 2 - ecoli
  dtmodel = C5.0(V9 ~ ., data = input2, subset = folds2$subsets[folds2$which != i])
  svmmodel = ksvm(V9 ~ ., data = input2[folds2$subsets[folds2$which != i],])
  test = input2[folds2$subsets[folds2$which == i], ]
  
  # Make predictions on the test set and calculate the error percentages made by both the trained models
  test = test[, -which(names(test) %in% c("V9"))]
  predictions_dt = predict.C5.0(dtmodel, test)
  predictions_svm = predict(svmmodel, test)
  
  error_dt = 100*sum(input2[folds2$subsets[folds2$which == i], 'V9'] != predictions_dt)/length(predictions_dt)
  error_svm = 100*sum(input2[folds2$subsets[folds2$which == i], 'V9'] != predictions_svm)/length(predictions_svm)
  DTERROR[2] = DTERROR[2] + error_dt
  SVMERROR[2] = SVMERROR[2] + error_svm
  
#-----DATASET 3 - Breast Cancer
  dtmodel = C5.0(V2 ~ ., data = input3, subset = folds3$subsets[folds3$which != i])
  svmmodel = ksvm(V2 ~ ., data = input3[folds3$subsets[folds3$which != i],])
  test = input3[folds3$subsets[folds3$which == i], ]
  
  # Make predictions on the test set and calculate the error percentages made by both the trained models
  test = test[, -which(names(test) %in% c("V2"))]
  predictions_dt = predict.C5.0(dtmodel, test)
  predictions_svm = predict(svmmodel, test)
  
  error_dt = 100*sum(input3[folds3$subsets[folds3$which == i], 'V2'] != predictions_dt)/length(predictions_dt)
  error_svm = 100*sum(input3[folds3$subsets[folds3$which == i], 'V2'] != predictions_svm)/length(predictions_svm)
  DTERROR[3] = DTERROR[3] + error_dt
  SVMERROR[3] = SVMERROR[3] + error_svm
  
#----DATASET 4 - glass
  dtmodel = C5.0(V11 ~ ., data = input4, subset = folds4$subsets[folds4$which != i])
  svmmodel = ksvm(V11 ~ ., data = input4[folds4$subsets[folds4$which != i],])
  test = input4[folds4$subsets[folds4$which == i], ]
  
  # Make predictions on the test set and calculate the error percentages made by both the trained models
  test = test[, -which(names(test) %in% c("V11"))]
  predictions_dt = predict.C5.0(dtmodel, test)
  predictions_svm = predict(svmmodel, test)
  
  error_dt = 100*sum(input4[folds4$subsets[folds4$which == i], 'V11'] != predictions_dt)/length(predictions_dt)
  error_svm = 100*sum(input4[folds4$subsets[folds4$which == i], 'V11'] != predictions_svm)/length(predictions_svm)
  DTERROR[4] = DTERROR[4] + error_dt
  SVMERROR[4] = SVMERROR[4] + error_svm
  
#-----DATASET 5 - yeast
  dtmodel = C5.0(V10 ~ ., data = input5, subset = folds5$subsets[folds5$which != i])
  svmmodel = ksvm(V10 ~ ., data = input5[folds5$subsets[folds5$which != i],])
  test = input5[folds5$subsets[folds5$which == i], ]
  
  # Make predictions on the test set and calculate the error percentages made by both the trained models
  test = test[, -which(names(test) %in% c("V10"))]
  predictions_dt = predict.C5.0(dtmodel, test)
  predictions_svm = predict(svmmodel, test)
  
  error_dt = 100*sum(input5[folds5$subsets[folds5$which == i], 'V10'] != predictions_dt)/length(predictions_dt)
  error_svm = 100*sum(input5[folds5$subsets[folds5$which == i], 'V10'] != predictions_svm)/length(predictions_svm)
  DTERROR[5] = DTERROR[5] + error_dt
  SVMERROR[5] = SVMERROR[5] + error_svm
}
DTERROR = DTERROR/10
SVMERROR = SVMERROR/10

# Compare the performance of the different classifiers using Wilcoxon Signed Rank test (see ?wilcox.test)
wilcox.test(DTERROR, SVMERROR)
