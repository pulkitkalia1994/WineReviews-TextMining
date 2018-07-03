library(tm)
library(SnowballC)
library(caTools)
library(rpart)
library(rpart.plot)

wine<-read.csv("../input/winemag-data_first150k.csv",stringsAsFactors = FALSE)

# Create corpus

corpus = VCorpus(VectorSource(wine$description)) 
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removePunctuation)

##removing word wine as it won't be much helpful
corpus = tm_map(corpus, removeWords, c("wine", stopwords("english")))

corpus = tm_map(corpus, stemDocument)

frequencies = DocumentTermMatrix(corpus)

sparse = removeSparseTerms(frequencies, 0.97)

wineSparse = as.data.frame(as.matrix(sparse))
colnames(wineSparse) = make.names(colnames(wineSparse))
wineSparse$country = as.factor(wine$country)

set.seed(123)

library(caret)
library(rpart)
library(rpart.plot)
library(e1071)
library("xgboost")
library("Ckmeans.1d.dp")
wineSparse$country<-as.numeric(wineSparse$country)
wineSparse$country<-wineSparse$country-1

train_index <- sample(1:nrow(wineSparse), nrow(wineSparse)*0.75)
# Full data set
data_variables <- as.matrix(wineSparse[,-1])
data_label <- wineSparse[,"country"]
data_matrix <- xgb.DMatrix(data = as.matrix(wineSparse), label = data_label)
# split train data and make xgb.DMatrix
train_data   <- data_variables[train_index,]
train_label  <- data_label[train_index]
train_matrix <- xgb.DMatrix(data = train_data, label = train_label)
# split test data and make xgb.DMatrix
test_data  <- data_variables[-train_index,]
test_label <- data_label[-train_index]
test_matrix <- xgb.DMatrix(data = test_data, label = test_label)

numberOfClasses <- length(unique(wineSparse$country))
xgb_params <- list("objective" = "multi:softmax",
                   "eval_metric" = "mlogloss",
                   "num_class" = numberOfClasses)
nround    <- 5 # number of XGBoost rounds
cv.nfold  <- 5

# Fit cv.nfold * cv.nround XGB models and save OOF predictions
                   
bst_model <- xgb.train(params = xgb_params,
                       data = train_matrix,
                       nrounds = nround)

test_pred <- predict(bst_model, newdata = test_matrix)
#outputdf<-as.matrix(table(test_pred,test_label))

sum<-0
for(i in 1:37733){
if(test_pred[[i]]==test_label[[i]]){
        sum<-sum+1
    }
}

cat("Accuracy is",sum/37733)


##You can use the same to predict type of wine and wineyard.