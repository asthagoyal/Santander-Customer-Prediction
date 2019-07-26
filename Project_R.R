rm(list=ls())

#Set working directory
setwd("E:/MY/Project/Santander Customer Transaction")


x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees')

install.packages(x)
lapply(x, require, character.only = TRUE)
rm(x)

Train= read.csv("Train1.csv", header = T, na.strings = c(" ", "", "NA"))


View(Train)
dim(Train)

Test= read.csv("Test.csv", header = T, na.strings = c(" ", "", "NA"))
 str(Test)
 
 #convert factor
 Train$target =  as.factor(Train$target)
 train$target =  as.factor(train$target)
 class(train$target)

# Grid
 
 require(gridExtra)
 #Count of target classes
 table(Train$target)
 #Percenatge counts of target classes
 table(Train$target)/length(Train$target)*100
 #Bar plot for count of target classes
 plot1<-ggplot(Train,aes(target))+theme_bw()+geom_bar(stat='count',fill='lightgreen')
 #Violin with jitter plots for target classes
 plot2<-ggplot(Train,aes(x=target,y=1:nrow(Train)))+theme_bw()+geom_violin(fill='lightblue')+
   facet_grid(Train$target)+geom_jitter(width=0.02)+labs(y='Index')
 grid.arrange(plot1,plot2, ncol=2)
 
 
 #Distribution of train attributes from 3 to 102
 for (var in names(Train)[c(3:102)]){
   target<-Train$target
   plot<-ggplot(Train, aes(x=Train[[var]],fill=target)) +
     geom_density(kernel='gaussian') + ggtitle(var)+theme_classic()
   print(plot)
 }
 
 
 #Distribution of train attributes from 103 to 202
 for (var in names(Train)[c(103:202)]){
   target<-Train$target
   plot<-ggplot(Train, aes(x=Train[[var]], fill=target)) +
     geom_density(kernel='gaussian') + ggtitle(var)+theme_classic()
   print(plot)
 }
 
 #Applying the function to find mean values per row in train and test data.
 train_mean<-apply(Train[,-c(1,2)],MARGIN=1,FUN=mean)
 test_mean<-apply(Test[,-c(1)],MARGIN=1,FUN=mean)
 ggplot()+
   #Distribution of mean values per row in train data
   geom_density(data=Train[,-c(1,2)],aes(x=train_mean),kernel='gaussian',show.legend=TRUE,color='blue')+theme_classic()+
   #Distribution of mean values per row in test data
   geom_density(data=Test[,-c(1)],aes(x=test_mean),kernel='gaussian',show.legend=TRUE,color='green')+
   labs(x='mean values per row',title="Distribution of mean values per row in train and test dataset")
 #Applying the function to find mean values per column in train and test data.
 train_mean<-apply(Train[,-c(1,2)],MARGIN=2,FUN=mean)
 test_mean<-apply(Test[,-c(1)],MARGIN=2,FUN=mean)
 ggplot()+
#Distribution of mean values per column in train data
geom_density(aes(x=train_mean),kernel='gaussian',show.legend=TRUE,color='blue')+theme_classic()+
   #Distribution of mean values per column in test data
geom_density(aes(x=test_mean),kernel='gaussian',show.legend=TRUE,color='green')+
labs(x='mean values per column',title="Distribution of mean values per row in train and test dataset")

#Applying the function to find standard deviation values per row in train and test data.
 train_sd<-apply(Train[,-c(1,2)],MARGIN=1,FUN=sd)
 test_sd<-apply(Test[,-c(1)],MARGIN=1,FUN=sd)
 ggplot()+
   #Distribution of sd values per row in train data
   geom_density(data=Train_sd[,-c(1,2)],aes(x=train_sd),kernel='gaussian',show.legend=TRUE,color='red')+theme_classic()+
   #Distribution of mean values per row in test data
   geom_density(data=Test_sd[,-c(1)],aes(x=test_sd),kernel='gaussian',show.legend=TRUE,color='blue')+
   labs(x='sd values per row',title="Distribution of sd values per row in train and test dataset")
 
 #Applying the function to find sd values per column in train and test data.
 train_sd<-apply(Train[,-c(1,2)],MARGIN=2,FUN=sd)
 test_sd<-apply(Test[,-c(1)],MARGIN=2,FUN=sd)
 ggplot()+
   #Distribution of sd values per column in train data
   geom_density(aes(x=train_sd),kernel='gaussian',show.legend=TRUE,color='red')+theme_classic()+
   #Distribution of sd values per column in test data
   geom_density(aes(x=test_sd),kernel='gaussian',show.legend=TRUE,color='blue')+
   labs(x='sd values per column',title="Distribution of std values per column in train and test dataset")
 
 
 #Applying the function to find skewness values per row in train and test data.
 train_skew<-apply(Train[,-c(1,2)],MARGIN=1,FUN=skewness)
 test_skew<-apply(Test[,-c(1)],MARGIN=1,FUN=skewness)
 ggplot()+
   #Distribution of skewness values per row in train data
   geom_density(aes(x=train_skew),kernel='gaussian',show.legend=TRUE,color='green')+theme_classic()+
   #Distribution of skewness values per column in test data
   geom_density(aes(x=test_skew),kernel='gaussian',show.legend=TRUE,color='blue')+
   labs(x='skewness values per row',title="Distribution of skewness values per row in train and test dataset")
 
 #Applying the function to find skewness values per column in train and test data.
 train_skew<-apply(Train[,-c(1,2)],MARGIN=2,FUN=skewness)
 test_skew<-apply(Test[,-c(1)],MARGIN=2,FUN=skewness)
 ggplot()+
   #Distribution of skewness values per column in train data
   geom_density(aes(x=train_skew),kernel='gaussian',show.legend=TRUE,color='green')+theme_classic()+
   #Distribution of skewness values per column in test data
   geom_density(aes(x=test_skew),kernel='gaussian',show.legend=TRUE,color='blue')+
   labs(x='skewness values per column',title="Distribution of skewness values per column in train and test dataset")
 
 
 #Applying the function to find kurtosis values per row in train and test data.
 train_kurtosis<-apply(Train[,-c(1,2)],MARGIN=1,FUN=kurtosis)
 test_kurtosis<-apply(Test[,-c(1)],MARGIN=1,FUN=kurtosis)
 ggplot()+
   #Distribution of sd values per column in train data
   geom_density(aes(x=train_kurtosis),kernel='gaussian',show.legend=TRUE,color='blue')+theme_classic()+
   #Distribution of sd values per column in test data
   geom_density(aes(x=test_kurtosis),kernel='gaussian',show.legend=TRUE,color='red')+
   labs(x='kurtosis values per row',title="Distribution of kurtosis values per row in train and test dataset")
 
 #Applying the function to find kurtosis values per column in train and test data.
 train_kurtosis<-apply(Train[,-c(1,2)],MARGIN=2,FUN=kurtosis)
 test_kurtosis<-apply(Test[,-c(1)],MARGIN=2,FUN=kurtosis)
 ggplot()+
   #Distribution of sd values per column in train data
   geom_density(aes(x=train_kurtosis),kernel='gaussian',show.legend=TRUE,color='blue')+theme_classic()+
   #Distribution of sd values per column in test data
   geom_density(aes(x=test_kurtosis),kernel='gaussian',show.legend=TRUE,color='red')+
   labs(x='kurtosis values per column',title="Distribution of kurtosis values per column in train and test dataset")
 
 #Calculte missing values
missing_val = data.frame(apply(Train,2,function(x){sum(is.na(x))}))
sum(missing_val)

missing_val1 = data.frame(apply(Test,2,function(x){sum(is.na(x))}))
sum(missing_val1)

#Correlations in train data
#convert factor to int
Train$target<-as.numeric(Train$target)
train_correlations<-cor(Train[,c(2:202)])
train_correlations
#Plot
corrgram(Train[,numeric_index], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")



# Outlier Analysis
numeric_index = sapply(Train,is.numeric) #selecting only numeric

numeric_data = Train[ ,numeric_index - 2]

cnames = colnames(numeric_data)
cnames


#remove outlier 
for(i in cnames){
     print(i)
     val = Train[,i][Train[,i] %in% boxplot.stats(Train[,i])$out]
     #print(length(val))
     Train = Train[which(!Train[,i] %in% val),]
}


qqnorm(Train$var_0)
hist(Train$var_1)

#Standardisation
 for(i in cnames){
   print(i)
   Train[,i] = (Train[,i] - mean(Train[,i]))/
                                  sd(Train[,i])
}

for(i in 1:ncol(Train)){
   
   if(class(Train[,i]) == 'factor'){
      
      Train[,i] = factor(Train[,i], labels=(1:length(levels(factor(Train[,i])))))
      
   }
}

# Logistic Regression
#for check
#train.index = createDataPartition(Train$target, p = .8, list = FALSE)
#train = Train[ train.index , ]
#test  = Train[-train.index,]
logit_model = glm(target ~ ., data = Train, family = binomial ("logit"), maxit = 1000)
#summary of the model
summary(logit_model)
#predict using logistic regression
logit_Predictions = predict(logit_model, newdata = Test, type = "response")
#convert prob
logit_Predictions = ifelse(logit_Predictions > 0.5, 1, 0)
##Evaluate the performance of classification model
#for check
#ConfMatrix_RF = table(test$target, logit_Predictions)
#ConfMatrix_RF

#False Negative rate
#check Precision
#(TP/(TP+FP))
#Check Recall
#(TP/(TP+FN))
#False Negative rate
#FNR = FN/FN+TP 

#Precision =14.44
#Recall = 71.09
#Accuracy = 54.74
#FNR = 28.91

 
##Decision tree for classification
#Develop Model on training data
C50_model = C5.0(target ~., Train, trials = 100, rules = TRUE)

#Summary of DT model
summary(C50_model)

#write rules into disk
write(capture.output(summary(C50_model)), "c50Rules.txt")

#Lets predict for test cases
C50_Predictions = predict(C50_model, Test, type = "class")

##Evaluate the performance of classification model
#for check
#ConfMatrix_C50 = table(test$target, C50_Predictions)
#confusionMatrix(ConfMatrix_C50)

#check Precision
#(TP/(TP+FP))
#Check Recall
#(TP/(TP+FN))
#False Negative rate
#FNR = FN/FN+TP 

#Recall= 19.90
#Precision = 19.59
#Accuracy = 89.95
#FNR = 80.73


###Random Forest
RF_model = randomForest(target ~ ., Train, importance = TRUE, ntree = 500)
#Presdict test data using random forest model
RF_Predictions = predict(RF_model, Test)
##Evaluate the performance of classification model
#for check
#ConfMatrix_RF = table(test$target, RF_Predictions)
#confusionMatrix(ConfMatrix_RF)

#check Precision
#(TP/(TP+FP))

#Check Recall
#(TP/(TP+FN))

#accuracy_score(y_test, y_pred)*100
#((TP+TN)*100)/(TP+TN+FP+FN)

#False Negative rate
#FNR = FN/FN+TP 

#Accuracy = 89.951
#Precision = 72.45
#Recall = 0.6
#FNR = 99.45
