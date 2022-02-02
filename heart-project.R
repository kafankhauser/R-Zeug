# install packages
if(!require("caret")) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require("tidyverse")) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require("ggpubr")) install.packages("ggpubr", repos = "http://cran.us.r-project.org")
if(!require("knitr")) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require("easypackages")) install.packages("easypackages", repos = "http://cran.us.r-project.org")
if(!require("randomForest")) install.packages("randomForest", repos = "http://cran.us.r-project.org")
library("easypackages")
libraries("caret", "tidyverse", "ggpubr", "knitr", "randomForest")
set.seed(2, sample.kind = "Rounding")

# download data set 
cleveland <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"),
                     header=FALSE, col.names =  
                       c("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang",
                         "oldpeak","slope", "ca", "thal", "num"))

hungary <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data"),
                      header=FALSE, col.names =  
                        c("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang",
                          "oldpeak","slope", "ca", "thal", "num"))  

switzerland <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data"),
                    header=FALSE, col.names =  
                      c("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang",
                        "oldpeak","slope", "ca", "thal", "num"))  

va <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data"),
                        header=FALSE, col.names =  
                          c("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang",
                            "oldpeak","slope", "ca", "thal", "num"))

data <- rbind(cleveland, hungary, switzerland, va)


# change outcome into 0 (no diagnosis) vs 1 (diagnosis) - classification
data$heart_disease <- ifelse(data$num == 0, 0, 1)
table(data$heart_disease)

# NA handeling
data[data == "?"] <- NA
# exclude $slope (NAs=309), $ca (NAs=611) and $thal (NAs=486) from datset
sum(is.na(data$trest))
sum(is.na(data$ca))
sum(is.na(data$thal))
data <- data %>% select(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, heart_disease)


# assign proper data formats
data$trestbps <- as.numeric(data$trestbps)
data$chol <- as.numeric(data$chol)
data$fbs <- as.factor(data$fbs)
data$restecg <- as.factor(data$restecg)
data$thalach <- as.numeric(data$thalach)
data$exang <- as.factor(data$exang)
data$oldpeak <- as.numeric(data$oldpeak)
data$heart_disease <- as.factor(data$heart_disease)


# Exploratory analysis
# estimate effect between continous variables and binary outcome
sapply(data, function(x) if("numeric" %in% class(x) ) {
  wilcox.test (x ~ data$heart_disease)} else { fisher.test(data$heart_disease, x, simulate.p.value = TRUE) } )

# create test set and train set
test_index <- createDataPartition(data$heart_disease, times = 1, p = 0.5, list = FALSE)
test_set <- data[test_index, ]
train_set <- data[-test_index, ]

#baseline prediction
random <- sample(c(0,1), nrow(test_set), replace = TRUE)
mean(random == test_set$heart_disease) # this is the accuracy of the baseline model

########## random forest
fit <- randomForest(heart_disease~. , data = train_set , na.action = na.omit)

# evaluate random forest : accuracy 76%
probabilities.test <- predict(fit, test_set, type='class')
conf_matrix <- confusionMatrix(probabilities.test, test_set$heart_disease, positive = '1')

# plot the confusion matrix
cm <- as.table(conf_matrix)
fourfoldplot(cm)

#evaluate the importance of different variables
varImpPlot(fit)
importance(fit)

