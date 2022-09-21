#install.packages("gbm")
#install.packages("xtable")
library (gbm)
library(ggplot2)
library(plyr)
library(dplyr)
library(corrplot)
library(caret)
library(gridExtra)
library(scales)
library(Rmisc)
library(ggrepel)
library(randomForest)
library(psych)
library(xgboost)
library(xtable)
library(lubridate)
library(TSstudio)
library(IRdisplay)
library(htmlwidgets)
library(h2o)

###  Dataset ---------------

setwd("~/Desktop/IntegerationCredit/Hourly_Energy_consumption")
df<-read.csv("PJME_hourly.csv", stringsAsFactors = FALSE)
glimpse(df)
df$Datetime<-as.Date(df$Datetime)

df1<-aggregate(PJME_MW~Datetime, df, sum)
glimpse(df1)
df2 <- df1

hist(df2$PJME_MW, col="orange", xlim=c(300000,1200000), main="Energy Consumption", xlab="MW")
boxplot(df2$PJME_MW, col="orange", main="Energy Consumption", ylab="MW")

# df2<-df1
# d2_date = as.POSIXct(df1$Datetime , format = "%Y-%m-%d %H:%M:%S ")
# df2$DATE = d2_date
# head(df2)
# print(xtable(df2[1:5,c(1,2,3)], type = "latex"), file = "filename2.tex")

ts1<-ts(data=df1$PJME_MW,
        start=c(lubridate::year(min(df1$Datetime)), lubridate::yday(min(df1$Datetime))),
        frequency=365)

ts_info(ts1)


### Expolring Variables  -------------------------------------------
# Saving widget to an HTML file
#references: 
#https://www.kaggle.com/product-feedback/25953   
#https://www.kaggle.com/farazrahman/talking-data-interactive-eda-basic-randomforest

ts_plot(ts1, title="Hourly Energy Consumption - PJME")
summary(df1)
ts_decompose(ts1)
ts_quantile(df1, period="monthly")
acf(ts1, lag.max=365.25*3)
### Preprocessing ---------------------------------------------------

ts_train=window(ts1, start=c(2002,1), end=c(2014,365))
ts_test=window(ts1,start=c(2015,1))
str(ts_test)



### Feature engineering ------------------------------------
df2<-df1%>%mutate(weekday=factor(lubridate::wday(Datetime, label=TRUE),ordered = FALSE),
                  month=factor(lubridate::month(Datetime, label=TRUE),ordered = FALSE),
                  lag365=dplyr::lag(PJME_MW, 365))%>%
  filter(!is.na(lag365))%>%
  arrange(Datetime)
head(df2, n=2)
## Spliting into train and test set 
h=1314
train<-df2[1:(nrow(df2)-h),]
test<-df2[(nrow(df2)-h+1):nrow(df2),]
x<-c("weekday","month","lag365")
y<-"PJME_MW"
### Linear regression -------------------------------------
m_lm=lm(PJME_MW ~ weekday+month+lag365, data=train)
summary(m_lm)
par(mfrow=c(2,2))
plot(m_lm)
test$pred_lm=predict(m_lm, newdata=test)
mape_lm=mean(abs(test$PJME_MW-test$pred_lm)/test$PJME_MW)
mape_lm

### XGBoost ---------------------------
#transform release_date in numeric (to use gbm)
h=1314
train<-df2[1:(nrow(df2)-h),]
test<-df2[(nrow(df2)-h+1):nrow(df2),]
x<-c("weekday","month","lag365")
y<-"PJME_MW"

train$Datetime<-as.numeric(train$Datetime)
test$Datetime<-as.numeric(test$Datetime)
summary(train$Datetime)
summary(test$Datetime)
boost.he=gbm(PJME_MW~. -PJME_MW, data=train, 
                 distribution="gaussian", n.trees=5000, interaction.depth=1)
#
#best.iter <- gbm.perf(boost.he, method="cv", plot=FALSE)
#for plot (se non l'avevo fatto prima)
par(mfrow=c(1,1))
#
#plot of training error
plot(boost.he$train.error^(1/2), type="l", ylab="training error")
#always decreasing with increasing number of trees
#
#
#relative influence plot
summary(boost.he) 
summary(boost.he) 
#let us modify the graphical paramters to obtain a better plot
#
#more space on the left
#
# default vector of paramters
mai.old<-par()$mai
mai.old
#new vector
mai.new<-mai.old
#new space on the left
mai.new[2] <- 2.1 
mai.new
#modify graphical parameters
par(mai=mai.new)
summary(boost.he, las=1) 
#las=1 horizontal names on y
summary(boost.he, las=1, cBar=10) 
#cBar defines how many variables
#back to orginal window
par(mai=mai.old)

# test set prediction for every tree (1:5000)
test

yhat.boost=predict(boost.he, newdata=test, n.trees=1:5000)

# calculate the error for each iteration
#use 'apply' to perform a 'cycle for' 
# the first element is the matrix we want to use, 2 means 'by column', 
#and the third element indicates the function we want to calculate

err = apply(yhat.boost,2,function(pred) mean((test$PJME_MW-pred)^2))
#
plot(err, type="l")

# error comparison (train e test)
plot((boost.he$train.error)^(1/2), type="l")
lines(err^(1/2), type="l", col=2)
#minimum error in test set
best=which.min(err)
abline(v=best, lty=2, col=4)
#
min(err) #minimum error




# 2 Boosting - deeper trees
boost.he=gbm(PJME_MW~ ., data=train, 
             distribution="gaussian", n.trees=5000, interaction.depth=4)

plot(boost.he$train.error, type="l")

par(mai=mai.new)

summary(boost.he, las=1, cBar=10)  #disegnare
par(mai=mai.old)

yhat.boost=predict(boost.he ,newdata=test,n.trees=1:5000)
err = apply(yhat.boost,2,function(pred) mean((test$PJME_MW-pred)^2))
plot(err, type="l")


plot(boost.he$train.error, type="l")
lines(err, type="l", col=2)
best=which.min(err)
abline(v=best, lty=2, col=4)
min(err)


# 3 Boosting - smaller learning rate 

boost.he=gbm(PJME_MW~ ., data=train, 
             distribution="gaussian", n.trees=5000, interaction.depth=1, shrinkage=0.01)
plot(boost.he$train.error, type="l")

par(mai=mai.new)

summary(boost.he, las=1, cBar=10) 
par(mai=mai.old)

yhat.boost=predict(boost.he ,newdata=test,n.trees=1:5000)
err = apply(yhat.boost,2,function(pred) mean((test$PJME_MW-pred)^2))
plot(err, type="l")


plot(boost.he$train.error, type="l")
lines(err, type="l", col=2)
best=which.min(err)
abline(v=best, lty=2, col=4)
min(err)


# 4 Boosting - combination of previous models
boost.he=gbm(PJME_MW~ ., data=train, 
             distribution="gaussian", n.trees=5000, interaction.depth=4, shrinkage=0.01)

plot(boost.he$train.error, type="l")
#

par(mai=mai.new)

summary(boost.he, las=1, cBar=10) 

par(mai=mai.old)

yhat.boost=predict(boost.he ,newdata=test,n.trees=1:5000)
err = apply(yhat.boost, 2, function(pred) mean((test$PJME_MW-pred)^2))
plot(err, type="l")


plot(boost.he$train.error, type="l")
lines(err, type="l", col=2)
best=which.min(err)
abline(v=best, lty=2, col=4)
err.boost= min(err)



boost.he
# partial dependence plots
#plot(boost.he, i.var=1, n.trees = best)
plot(boost.he, i.var=2, n.trees = best)
#plot(boost.he, i.var=3, n.trees = best)
#plot(boost.he, i.var=c(1,2), n.trees = best) #bivariate
#
#plot(boost.he, i.var=4, n.trees = best)
#plot(boost.he, i.var=c(1,3), n.trees = best) #bivariate

#plot(boost.he, i.var=c(2,3), n.trees = best) #bivariate
#plot(boost.he, i.var=c(2,4), n.trees = best) #bivariate

