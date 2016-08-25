# Configure environment ---------------------------------------------------
Sys.sleep(30)
# Github
# system('git remote add origin https://github.com/scottbradshaw/Red-Hat-Business-Value.git')
# system('git config remote.origin.url git@github.com:scottbradshaw/Red-Hat-Business-Value.git')
# system('git push -u origin master')

library(data.table)
library(caret)
library(dplyr)
library(tidyr)
library(lubridate)
library(Metrics)
library(pROC)
library(randomForest)
library(rpart)

# Get data ----------------------------------------------------------------

submit <- data.table(read.csv("act_test.csv"))
submit[,date:=as.Date(date)]
submit[,outcome:=factor(outcome)]

train <- data.table(read.csv("act_train.csv"))
train[,date:=as.Date(date)]
train[,outcome:=factor(outcome)]

people <- data.table(read.csv("people.csv"))

# Create variables --------------------------------------------------------
train[,month:=month(date,label=T)]
train[,year:=year(date)]
train[,day:=day(date)]
train[,wday:=wday(date,label=T)]

submit[,month:=month(date,label=T)]
submit[,year:=year(date)]
submit[,day:=day(date)]
submit[,wday:=wday(date,label=T)]

# Split into validation and test ------------------------------------------

# Because there are multiple activities/rows per person, let's split the list of unique people into three groups and merge back on to the original dataset
set.seed(2)
temp <- train[!duplicated(people_id),.(people_id)] %>% sample_frac(size=0.8)
test <- anti_join(x=train,y=temp,by="people_id")
test <- data.table(test)
train <- inner_join(x=train,y=temp,by="people_id")
train <- data.table(train)

rm(temp)

# Explore! ----------------------------------------------------------------

head(train)

train %>% group_by(activity_category) %>% summarise(count=n()) %>% mutate(percent=count/sum(count))
temp <- train %>% count(activity_category,outcome) %>% mutate(percent=n/sum(n))
ggplot(temp) + geom_bar(aes(x=activity_category,y=percent,fill=factor(outcome)),stat="identity")

temp <- train %>% count(year,outcome) %>% mutate(percent=n/sum(n))
ggplot(temp) + geom_bar(aes(x=year,y=percent,fill=factor(outcome)),stat="identity")

temp <- train %>% count(month,outcome) %>% mutate(percent=n/sum(n))
ggplot(temp) + geom_bar(aes(x=month,y=percent,fill=factor(outcome)),stat="identity")

temp <- train %>% count(day,outcome) %>% mutate(percent=n/sum(n))
ggplot(temp) + geom_bar(aes(x=day,y=percent,fill=factor(outcome)),stat="identity")

temp <- train %>% count(wday,outcome) %>% mutate(percent=n/sum(n))
ggplot(temp) + geom_bar(aes(x=wday,y=percent,fill=factor(outcome)),stat="identity")

rm(temp)

# First model -------------------------------------------------------------

now <- Sys.time()

set.seed(2)
glm_1 <- glm(outcome ~ activity_category + month + year + wday,
             data=train,
             family="binomial")

difftime(Sys.time(),now)

qplot(x=predict(glm,train,type="prob")[,2],fill=train$outcome,geom="density",alpha=0.5)

auc(train$outcome,predict(glm_1,train,type="response"))
# Area under the curve: 0.6144

# Create features ---------------------------------------------------------

# TRAIN

# Features that are time/history based

## Because there are multiple activities per day for some people, and because we can't assume any order of the activities that happened on the same day, we have to assume they are independent. Hence calculate variables at the day level rather than activity level

temp <- train[,list(number_of_activities=.N),.(people_id,date)][order(people_id,date)]
temp[,number_of_activities_before:=shift(cumsum(number_of_activities), fill=0),people_id]
temp[,number_of_activities:=NULL]
train <- merge(train,temp,by=c("people_id","date"))

train[,first_activity:=ifelse(number_of_activities_before==0,1,0)]

train[,number_of_activities_on_day:=.N,.(people_id,date)]

temp <- train[!duplicated(train,by=c("people_id","date")),.(people_id,date)]
setorder(x=temp,people_id,date)
temp[,days_since_last_activity:=c(NA,diff(date)),people_id]

train <- merge(train,temp,by=c("people_id","date"))

train[,activity_yesterday:=ifelse(is.na(days_since_last_activity),"new",ifelse(days_since_last_activity==1,1,0))]

# Capped features
train[,number_of_activities_before_capped:=ifelse(number_of_activities_before>150,150,number_of_activities_before)]
train[,number_of_activities_on_day_capped:=ifelse(number_of_activities_on_day_capped>30,30,number_of_activities_on_day_capped)]

# TEST

extra_variables <- function(x) {
  # number_of_activities_before
  x[,number_of_activities_before:=NULL]
  temp <- x[,list(number_of_activities=.N),.(people_id,date)][order(people_id,date)]
  temp[,number_of_activities_before:=shift(cumsum(number_of_activities), fill=0),people_id]
  temp[,number_of_activities:=NULL]
  x <- merge(x,temp,by=c("people_id","date"))
  
  # first_activity
  x[,first_activity:=ifelse(number_of_activities_before==0,1,0)]
  
  # number_of_activities_on_day
  x[,number_of_activities_on_day:=.N,.(people_id,date)]
  
  # days_since_last_activity
  x[,days_since_last_activity:=NULL]
  temp <- x[!duplicated(x,by=c("people_id","date")),.(people_id,date)]
  setorder(x=temp,people_id,date)
  temp[,days_since_last_activity:=c(NA,diff(date)),people_id]
  x <- merge(x,temp,by=c("people_id","date"))
  
  # activity_yesterday
  x[,activity_yesterday:=as.factor(ifelse(is.na(days_since_last_activity),"new",ifelse(days_since_last_activity==1,1,0)))]
  
  # capped variables
  x[,number_of_activities_before_capped:=ifelse(number_of_activities_before>150,150,number_of_activities_before)]
  x[,number_of_activities_on_day_capped:=ifelse(number_of_activities_on_day>30,30,number_of_activities_on_day)]
  
  return(x)
}

validation <- extra_variables(validation)
test <- extra_variables(test)

# Analyse features --------------------------------------------------------

# number_of_activities

## Some people have a large amount of activities. Make a new capped feature above
train[,.N,people_id][order(-N)][1:10]
quantile(train$number_of_activities_before,probs=c(0.9,0.95,0.99))

temp <- train[,list(activities=.N),.(number_of_activities_before_capped,outcome)][,percent:=activities/sum(activities),number_of_activities_before_capped][order(number_of_activities_before_capped,outcome)] # don't know how to do this in dplyr
ggplot(temp) + geom_bar(aes(x=number_of_activities_before_capped,y=percent,fill=factor(outcome)),stat="identity")

# number_of_activities_on_day
temp <- train[,list(activities=.N),.(number_of_activities_on_day_capped,outcome)][,percent:=activities/sum(activities),number_of_activities_on_day_capped][order(number_of_activities_on_day_capped,outcome)] # don't know how to do this in dplyr
ggplot(temp[number_of_activities_on_day_capped<=30]) + geom_bar(aes(x=number_of_activities_on_day_capped,y=percent,fill=factor(outcome)),stat="identity")

# first_activity
temp <- train[,.N,.(first_activity,outcome)][order(first_activity,outcome)]
temp[,perc:=N/sum(N),first_activity]

# activity_yesterday
temp <- train[,.N,.(activity_yesterday,outcome)][order(activity_yesterday,outcome)]
temp[,perc:=N/sum(N),activity_yesterday]

# days_since_last_activity
temp <- train[,list(activities=.N),.(days_since_last_activity,outcome)][order(days_since_last_activity,outcome)]
temp[,percent:=activities/sum(activities),days_since_last_activity]
ggplot(temp[days_since_last_activity<=365]) + geom_bar(aes(x=days_since_last_activity,y=percent,fill=factor(outcome)),stat="identity")

temp <- train[,list(activities=.N),.(floor(days_since_last_activity/28),outcome)][order(floor,outcome)]
temp[,percent:=activities/sum(activities),floor]
ggplot(temp) + geom_bar(aes(x=floor,y=percent,fill=factor(outcome)),stat="identity")

# Models ------------------------------------------------------------------

## Second iteration of glm

now <- Sys.time()

set.seed(2)
glm_2 <- glm(outcome ~ activity_category + month + year + wday + number_of_activities_before_capped + number_of_activities_on_day_capped + + activity_yesterday,
             data=train,
             family="binomial")

difftime(Sys.time(),now)

qplot(x=predict(glm_2,train,type="response"),fill=train$outcome,geom="density",alpha=0.5)

sample <- train %>% sample_frac(size=0.2)
auc(sample$outcome,predict(glm_2,sample,type="response"))
auc(test$outcome,predict(glm_2,test,type="response"))
rm(sample)
# TRAIN AUC: 0.6318
# TEST AUC: 0.6262

## Decision tree
train_control <- trainControl(method="repeatedcv",number=3,repeats=1,classProbs = T,summaryFunction = multiClassSummary,allowParallel = F)

now <- Sys.time()
set.seed(2)
dt_1 <- train(factor(outcome,labels=c("No","Yes")) ~ activity_category + month + year + wday + number_of_activities_before_capped + number_of_activities_on_day_capped + activity_yesterday,
              data=train %>% sample_frac(size=0.01),
              method="rpart",
              trControl=train_control,
              metric="ROC")#data.frame(cp=seq(0.00001,0.0002,0.000025)))
difftime(Sys.time(),now)

sample <- train %>% sample_frac(size=0.2)
sample[,outcome:=factor(outcome,labels=c("No","Yes"))]
auc(sample$outcome,predict(dt_1,sample,type="prob")[,2])
auc(test$outcome,predict(dt_1,test,type="prob")[,2])
rm(sample)

dt_1 <- rpart(outcome ~ activity_category + month + year + wday + number_of_activities_before_capped + number_of_activities_on_day_capped + activity_yesterday,
              data=train,
              control=list(cp=0.00001))
sample <- train %>% sample_frac(size=0.2)
auc(sample$outcome,predict(dt_1,sample,type="prob")[,2])
auc(test$outcome,predict(dt_1,test,type="prob")[,2])
rm(sample)
# TRAIN AUC: 0.6213
# TEST AUC: 0.6168

## Random Forest

now <- Sys.time()

set.seed(2)
rf_1 <- randomForest(outcome ~ activity_category + month + year + wday + number_of_activities_before_capped + number_of_activities_on_day_capped + activity_yesterday,
                     data=train %>% sample_frac(0.4),
                     importance=T,
                     ntree=100)

sample <- train %>% sample_frac(size=0.2)
auc(sample$outcome,predict(rf_1,sample,type="prob")[,2])
auc(test$outcome,predict(rf_1,test,type="prob")[,2])
rm(sample)
# TRAIN AUC: 0.7218
# TEST AUC: 0.6675