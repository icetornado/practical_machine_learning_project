library(caret)
library(randomForest)
library(doParallel)

registerDoParallel()
#getDoParWorkers()

rawTraining <- read.csv(file.path("data", "pml-training.csv"), stringsAsFactors=FALSE, na.strings = c("NA","#DIV/0!",""))
rawTesting <- read.csv(file.path("data", "pml-testing.csv"), stringsAsFactors=FALSE, na.strings = c("NA","#DIV/0!",""))

## -- split data
set.seed(124567)
inTrain <- createDataPartition(y=rawTraining$classe, p=0.6, list=FALSE)
training <- rawTraining[inTrain, ]
testing <- rawTraining[-inTrain, ]

## counting NAs in each column, and returning a list of columns which have  > 80% NAs
naList <- data.frame(ord = integer(0), name = character(0), cnNA = integer(0))
for(j in 1:length(training)) {
        if(sum(is.na(training[, j])) / length(training[,j]) > 0.8) {
                naList <- rbind(naList, data.frame(ord = j, name = names(training)[j], cnNA = sum(is.na(training[, j]))))
                #print(names(rawTraining)[j])
        }
}

## eliminate columns
removeCols <- c(1:7, naList$ord)

training <- training[-removeCols]
testing <- testing[-removeCols]

## set factor
training$classe <- as.factor(training$classe)
testing$classe <- as.factor(testing$classe)


## remove highly correlated vars
correlationMatrix <- cor(training[-53])
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
training1 = training[-highlyCorrelated]
testing1 = testing[-highlyCorrelated]

## declaring fit control
fitControl <- trainControl(
        method = "repeatedcv",
        number = 10,
        repeats = 3)

## test rpart
fitRPart <- train(classe ~ ., method = "rpart", data = training1)
predictRPart <- predict(fitRPart, testing[, -22])
confusionRPart <- confusionMatrix(testing1[, 22], predictRPart)
## results : Accuracy : 0.5038   -  Kappa : 0.3569   

fitRPartBig <- train(classe ~ ., method = "rpart", data = training)
predictRPartBig <- predict(fitRPartBig, testing[, -53])
confusionRPartBig <- confusionMatrix(testing[, 53], predictRPartBig)
##  Accuracy : 0.4992  - Kappa : 0.3455     

## gbm skinny
fitGBM <- train(classe ~ ., method = "gbm", data = training1, verbose = FALSE, trControl = fitControl)
predictGBM <- predict(fitGBM, testing1[, -22])
confusionGBM <- confusionMatrix(testing1[, 22], predictGBM)

#trellis.par.set(caretTheme())
#plot(fitGBM, metric = "Kappa")
## -- Accuracy : 0.879         -    Kappa : 0.8471                  running time 5 mins

## gbm fat
fitGBMBig <- train(classe ~ ., method = "gbm", data = training, verbose = FALSE, trControl = fitControl)
predictGBMBig <- predict(fitGBMBig, testing[, -53])
confusionGBMBig <- confusionMatrix(testing[, 53], predictGBMBig)

## Accuracy : 0.9601  ---  Kappa : 0.9507    with no grid tuning 

##random forest slim
# fitControl2 <- trainControl(
#         method = "repeatedcv",
#         number = 10,
#         repeats = 3)
fitRF <- train(classe~ ., data=training1, method="rf", trControl=fitControl, verbose = FALSE)
predictRF <- predict(fitRF, newdata = testing1)
confusionRF <- confusionMatrix(predictRF, testing1$classe)

#trellis.par.set(caretTheme())
plot(fitRF, metric = "Kappa")

plot(varImp(fitRF))
transparentTheme(trans = .4)
featurePlot(x = training1[, c("pitch_forearm", "magnet_belt_y", "roll_dumbbell", "roll_forearm")],
            y = training1$classe,
            plot = "pairs",
            ## Add a key at the top
            auto.key = list(columns = 4))

featurePlot(x = training1[, c("pitch_forearm", "magnet_belt_y", "roll_dumbbell", "roll_forearm")],
            y = training1$classe,
            plot = "box",
            ## Pass in options to bwplot() 
            scales = list(y = list(relation="free"),
                          x = list(rot = 90)),
            layout = c(4,1 ),
            auto.key = list(columns = 4))

qplot(pitch_forearm, gyros_arm_y, colour=classe, data=training1)

library("GGally")
ggpairs(training1[, c("pitch_forearm", "magnet_belt_y", "roll_dumbbell", "roll_forearm")], color = "classe", data = training1)
## Accuracy : 0.9806   -  Kappa : 0.9755      Runtime = 5 mins

## random forest all
fitMoreRF <- train(classe~ ., data=training, method="rf", trControl=fitControl, verbose = FALSE)
predictMoreRF <- predict(fitMoreRF, newdata = testing)
confusionMoreRF <- confusionMatrix(predictMoreRF, testing$classe)
##  Accuracy : 0.9922       -   Kappa : 0.9902    Runtime = 8 mins   

## comparing models
modelList <- list(GBM = fitGBM,
                  RF = fitRF,
                  RF53 = fitMoreRF)

resamps <- resamples(modelList )
trellis.par.set(theme1)
bwplot(resamps, layout = c(3, 1))

## test submission
pml_write_files = function(x){
        n = length(x)
        for(i in 1:n){
                filename = paste0("problem_id_",i,".txt")
                write.table(x[i],file=file.path("results",filename),quote=FALSE,row.names=FALSE,col.names=FALSE)
        }
}

testPredict <- predict(modFit, newdata = rawTesting)
#library(pROC)
for(j in 1:length(modelList)) {
        tP <- predict(modelList[j],  newdata = rawTesting)
        #auc <- roc()
        print(tP)
}

pml_write_files(testPredict)
