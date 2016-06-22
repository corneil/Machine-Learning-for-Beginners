library(RSQLite)
library(SparseM)
library(NLP)
library(RTextTools)
source('create_matrix_fix.R')
# connect to SQLite database
con <- dbConnect(drv=RSQLite::SQLite(), dbname="amazon-fine-foods/database.sqlite")
# fetch all data
reviews <- dbGetQuery(con, "select Score, Text from Reviews") # , HelpfulnessNumerator, HelpfulnessDenominator
print('loaded data')
data <- reviews[1:20000,]
trainsize <- nrow(data) * 0.75
evalsize <- nrow(data) * 0.25
train <- data[1:trainsize,]
targetStart <- nrow(data)-(evalsize - 1)
targetdata <- data[targetStart:nrow(data),]
print('partitioned data')
print('create matrix')
startTime <- proc.time()
dataMatrix <- create_matrix(train$Text) # , removeSparseTerms=0.99
print('create container')
trainSplit = as.integer(trainsize * 0.75)
container <- create_container(dataMatrix, train$Score, trainSize = 1:trainSplit, testSize = trainSplit:trainsize, virgin=FALSE)
print('train model')
model <- train_model(container, "SVM", kernel="linear", cost=1)
trainTime <- proc.time() - startTime
print('Train time')
print(trainTime)
startTime <- proc.time()
print('create predictions')
predMatrix <- create_matrix(targetdata$Text, originalMatrix=dataMatrix)
predSize = nrow(targetdata);
print('predicting data')
predictionContainer <- create_container(predMatrix, labels=rep(0,predSize), testSize=1:predSize, virgin=FALSE)
results <- classify_model(predictionContainer, model)
predictTime <- proc.time() - startTime
print('Prediction time')
print(predictTime)
print('Combining results and target set')
combined <- data.frame(as.numeric(results$SVM_LABEL), as.numeric(results$SVM_PROB), as.numeric(targetdata$Score))
predictedValues <- combined[,1]
actualValues <- combined[,3]
errors <- predictedValues - actualValues
errorsHist <- table(errors)
accuracy <- as.numeric(errorsHist[names(errorsHist) == 0]) / nrow(targetdata)
print(sprintf('Accuracy:%f%%', accuracy * 100))
