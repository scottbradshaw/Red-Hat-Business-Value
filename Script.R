# Configure environment ---------------------------------------------------

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
temp <- train[!duplicated(people_id),.(people_id)] %>% sample_frac(size=0.6)
validation <- anti_join(x=train,y=temp,by="people_id")
validation <- data.table(validation)
train <- inner_join(x=train,y=temp,by="people_id")
train <- data.table(train)

temp <- validation[!duplicated(people_id),.(people_id)] %>% sample_frac(size=0.5)
test <- anti_join(x=validation,y=temp,by="people_id")
test <- data.table(test)
validation <- inner_join(x=validation,y=temp,by="people_id")
validation <- data.table(validation)

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

# Features that are time based
