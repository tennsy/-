library(caret)
library(klaR)
library(kknn)
library(e1071)
library(adabag)
library(randomForest)
library(rpart)
library(rpart.plot)
library(xgboost)
library(caretEnsemble)
library(ROCR)
library(pROC)

data <- read.csv("data.csv")
data<-as.data.frame(data[,-c(11,12)])
names(data)[13] <- "y"
levels(data$y)[1] <- "0"
levels(data$y)[2] <- "1"

n <- round(2/3*nrow(data))
set.seed(1)
sub_train=sample(1:nrow(data),n)
datatrain=data[sub_train,]  #训练集
datatest=data[-sub_train,]  #测试集
#naive bayes---------------------------------------------------------------------------
NB.model <- NaiveBayes(y~age+fnlwgt+education.num+hours.per.week+.,datatrain,usekernel=T,fL=1)
pred.NB <- predict(NB.model,datatrain[,-13])#预测
perf.values.NB=confusionMatrix(pred.NB$class,datatrain[,13],positive="1")
perf.values.NB
perf.values.NB$byClass
measure.NB <- postResample(pred.NB$class,datatrain[,13]) #预测效果
measure.NB

#KNN-----------------------------------------------------------------------------------
pre=preProcess(datatrain,method = "range")
newdatatrain=predict(pre,datatrain)
pre=preProcess(datatest,method = "range")
newdatatest=predict(pre,datatest)

model.tkknn <- train.kknn(y~.,kmax=100,newdatatrain,kernel = c("rectangular", "triangular", "epanechnikov", "optimal", "cos", "inv", "gaussian","triweight", "biweight"),distance=2,scale=T)
plot(model.tkknn)
model.tkknn #输出最优参数情况
table(newdatatrain[,13],model.tkknn$fitted.values[[66]]) #混淆矩阵
perf.values.KNN=confusionMatrix(model.tkknn$fitted.values[[66]],newdatatrain[,13],positive="1")
perf.values.KNN
perf.values.KNN$byClass
measure.KNN <- postResample(model.tkknn$fitted.values[[66]],newdatatrain[,13]) #预测效果
measure.KNN

model.kknn <- kknn(y~.,newdatatrain,newdatatest[,-13],k=66,scale=T,distance=2,kernel= "inv")
measure.kknn <- postResample(model.kknn$fitted.values,newdatatest[,13]) 
measure.kknn
perf.values.KNN=confusionMatrix(model.kknn$fitted.values,newdatatest[,13],positive="1")
perf.values.KNN
perf.values.KNN$byClass

pred.knn <- prediction(model.kknn$prob[,2] , newdatatest[,13])
perf.auc.knn <- performance(pred.knn,measure="auc")#所得的是一个list对象
AUC.KNN=unlist(perf.auc.knn@y.values)#AUC值
AUC.KNN

#作图
#敏感度-特异度曲线
perf1.knn <- performance(pred.knn,"sens","spec")
plot(perf1.knn,main="knn敏感度-特异度曲线")
#ROC
perf2.knn <- performance(pred.knn,"tpr","fpr")
plot(perf2.knn,main="knn-ROC")
abline(a=0,b=1,lwd=2,lty=2)
#召回率-精确率曲线
perf3.knn <- performance(pred.knn,"prec","rec")
plot(perf3.knn,main="knn召回率-精确率曲线")
#提升图
perf4.knn <- performance(pred.knn,"lift","rpp")
plot(perf4.knn,main="knn提升图")
#决策树--------------------------------------------------------------------------------
rp_rpart <- rpart(y~.,datatrain,minsplit=1,maxdepth=5,cp=0.001, parms =list(split="gini"))
rpart.plot(rp_rpart,type=4,fallen.leaves=TRUE) #画出树状图

cpmatrix <- printcp(rp_rpart)
plotcp(rp_rpart)
mincpindex <- which.min(cpmatrix[,4])
cponeSE <- cpmatrix[mincpindex,4]+cpmatrix[mincpindex,5]
cpindex <- min(which(cpmatrix[,3]<=cponeSE))
cpmatrix[cpindex,1]#所确定的cp值
#剪枝
rp_rpart2 <- prune(rp_rpart,cp=cpmatrix[cpindex,1])
rpart.plot(rp_rpart2,type=4) #每类判定条件更明确

pre <- data.frame(predict(rp_rpart2,datatrain[,-13]))
pre$class <- factor(rep("0",nrow(datatrain)),levels=c("0","1"))
for (i in 1:nrow(datatrain)){
  if (pre[i,1]<pre[i,2])   
    pre$class[i] <- factor("1",levels=c("0","1"))
}
perf.values.rp <- confusionMatrix(pre$class,datatrain[,13],positive="1")
perf.values.rp
perf.values.rp$byClass
measure.rp <- postResample(pre$class,datatrain[,13]) #预测效果
measure.rp

#bagging--------------------------------------------------------------------------------
m=10
errorvec <- rep(0,m)
for (i in 1:m){
  set.seed(1) 
  bagcv <- bagging.cv(y~.,data=datatrain,v=10,mfinal=i,control=rpart.control(minsplit=1,maxdepth=10))
  errorvec[i] <- bagcv$error
}
errorvec
set.seed(1) 
model.bag <- bagging(y~.,data=datatrain,mfinal=4,control=rpart.control(minsplit=1,maxdepth=10))

perf.values <- confusionMatrix(factor(model.bag$class),datatrain$y,positive="1")
perf.values
perf.values$byClass

measure.bag <- postResample(model.bag$class,datatrain$y) 
measure.bag

pred.bag <- predict(model.bag,datatest[,-13],type="class")
perf.values <- confusionMatrix(factor(pred.bag$class),datatest[,13],positive="1")
perf.values
perf.values$byClass

measure.bag <- postResample(pred.bag$class,datatest$y) 
measure.bag

pred.bag <- prediction( pred.bag$prob[,2], datatest[,13])
perf.auc.bag <- performance(pred.bag,measure="auc")#所得的是一个list对象
AUC.bag=unlist(perf.auc.bag@y.values)#AUC值
AUC.bag

model.bag$importance #变量相对重要性
importanceplot(model.bag) #变量相对重要性柱状图

#作图
#敏感度-特异度曲线
perf1.bag <- performance(pred.bag,"sens","spec")
plot(perf1.bag,main="bag敏感度-特异度曲线")
#ROC
perf2.bag <- performance(pred.bag,"tpr","fpr")
plot(perf2.bag,main="bag-ROC")
abline(a=0,b=1,lwd=2,lty=2)
#召回率-精确率曲线
perf3.bag <- performance(pred.bag,"prec","rec")
plot(perf3.bag,main="bag召回率-精确率曲线")
#提升图
perf4.bag <- performance(pred.bag,"lift","rpp")
plot(perf4.bag,main="bag提升图")
#RF(最优)-----------------------------------------------------------------------------------
datatrain <- datatrain[complete.cases(datatrain),]
datatest <- datatest[complete.cases(datatest),]

p <- ncol(datatrain)-1
err <- rep(0,p)
B <- rep(0,p)
for (i in 1:p){
  set.seed(1)
  rfmodel <- randomForest(y~.,data=datatrain,ntree=500,mtry=i,importance=T,proximity=T,control=rpart.control(minsplit=1,maxdepth=10))
  err[i] <- min(rfmodel$err.rate[,1])
  B[i] <- which.min(rfmodel$err.rate[,1])
}
err
B
mtry.optimal <- which.min(err)
ntree.optimal <- B[which.min(err)]
c(mtry.optimal,ntree.optimal)

set.seed(1)
model.rf <- randomForest(y~.,data=datatrain,ntree=182,mtry=2,importance=T,proximity=T)
plot(model.rf)#看树的个数与OOB mse的关系

perf.values <- confusionMatrix(model.rf$predicted,datatrain[,13],positive="1")
perf.values
perf.values$byClass

measure.rf <- postResample(model.rf$predicted,datatrain$y) 
measure.rf

pred.rf <- predict(model.rf,datatest[,-13],type="class")
perf.values <- confusionMatrix(pred.rf,datatest[,13],positive="1")
perf.values
perf.values$byClass

measure.rf <- postResample(pred.rf,datatest[,13]) 
measure.rf

roc.rf <- roc(newdatatest[,13], as.numeric(pred.rf))
plot(roc.rf, print.auc = TRUE, print.thres = TRUE,main="rf-ROC")

model.rf$importance#各变量的重要程度
varImpPlot(model.rf, main="Variable Importance Random Forest computer")#作图

#adaboost--------------------------------------------------------------------------
listcoeflearn <- c("Breiman","Freund","Zhu")
m <- 10
err <- matrix(0,nrow=length(listcoeflearn),ncol=m)
for (i in 1:length(listcoeflearn)){
  for (j in 1:m) {
    set.seed(1)
    cv.model <- boosting.cv(y~.,data=datatrain,v=2,boos=T,coeflearn=listcoeflearn[i],mfinal=j,control=rpart.control(minsplit=1,maxdepth=10))
    err[i,j] <- cv.model$error
  }
}
err
x <- 1:m
plot(err[1,],type="b",col="red",ylim=c(0,0.5))
points(err[2,],type="b",col="blue")
points(err[3,],type="b",col="green")
legend('topright',listcoeflearn,col=c("red","blue","green"),pch=rep(1,3))

model.ada  <- boosting(y~.,data=datatrain,boos=F,coeflearn="Breiman",mfinal=9,control=rpart.control(minsplit=1,maxdepth=10))

perf.values <- confusionMatrix(factor(model.ada$class),datatrain[,13],positive="1")
perf.values
perf.values$byClass

measure.ada <- postResample(model.ada$class,datatrain$y) 
measure.ada

pred.ada <- predict(model.ada,datatest[,-13],type="class")
perf.values <- confusionMatrix(factor(pred.ada$class),datatest[,13],positive="1")
perf.values
perf.values$byClass

measure.ada <- postResample(pred.ada$class,datatest[,13]) 
measure.ada

pred.ada <- prediction(pred.ada$prob[,2], datatest[,13])
perf.auc.ada <- performance(pred.ada,measure="auc")#所得的是一个list对象
AUC.ada=unlist(perf.auc.ada@y.values)#AUC值
AUC.ada

model.ada$importance
importanceplot(model.ada)

#作图
#敏感度-特异度曲线
perf1.ada <- performance(pred.ada,"sens","spec")
plot(perf1.ada,main="ada敏感度-特异度曲线")
#ROC
perf2.ada <- performance(pred.ada,"tpr","fpr")
plot(perf2.ada,main="ada-ROC")
abline(a=0,b=1,lwd=2,lty=2)
#召回率-精确率曲线
perf3.ada <- performance(pred.ada,"prec","rec")
plot(perf3.ada,main="ada召回率-精确率曲线")
#提升图
perf4.ada <- performance(pred.ada,"lift","rpp")
plot(perf4.ada,main="ada提升图")
#SVM（跑太慢）-----------------------------------------------------------------------------------
costset <- seq(-5,15,2)
gammaset <- seq(-15,3,2)
costexp <- 2**costset
gammaexp <- 2**gammaset
accuracy <- matrix(0,length(costset),length(gammaset))
for (i in 1:length(costset)) {
  for (j in 1:length(gammaset)){
    set.seed(1)
    cvmodel.svm <- svm(y~., newdatatrain,cross= 2,cost=costexp[i],gamma=gammaexp[j])
    accuracy[i,j] <- cvmodel.svm$tot.accuracy
  }
}
dimnames(accuracy)=list(costset,gammaset) #模型精度变量的列名和行名
accuracy
max(accuracy)#正确率最大值
which.max(accuracy)
set.seed(1)
model.svm <- svm(y~., newdatatrain,cost=2**1,gamma=2**(-3), probability = TRUE)
summary(model.svm)

pred.svm <- predict(model.svm,newdatatest[,-13])

perf.values <- confusionMatrix(pred.svm,newdatatest[,13],positive="1")
perf.values
perf.values$byClass

measure.svm <- postResample(pred.svm,newdatatest[,13]) 
measure.svm

roc.svm <- roc(newdatatest[,13], as.numeric(pred.svm))
plot(roc.svm, print.auc = TRUE, print.thres = TRUE,main="svm-ROC")

onehottrain <- model.matrix(~.-1,data=newdatatrain[,-13])#one-hot编码后的自变量
onehottrain
plot(cmdscale(dist(onehottrain[,-1])), col = as.integer(newdatatrain[,13]), pch = c("o","+")[1:5020 %in% model.svm$index +1])
legend("bottomleft",col=c(1,2,2),(byt = n),legend=c("y=no & a supporter","y=yes & a supporter","y=yes & not a supporter"),pch = c("+","+","o"))

#stacking-----------------------------------------------------------------------------
levels(datatrain$y) <- list(no="0", yes="1")

set.seed(1)
list.stacking <- caretList(x=datatrain[,-13],y=datatrain[,13], trControl=trainControl(method="cv",savePredictions='final',classProbs = TRUE),  methodList=c("rpart","rf"))
list.stacking#不能加kknn
model.stacking <- caretStack(list.stacking, method="glm")
model.stacking

#xgboost-----------------------------------------------------------------------------
newdata <- read.csv("data无-.csv")
newdata<-as.data.frame(newdata[,-c(11,12)])
names(newdata)[13] <- "y"
levels(newdata$y)[1] <- "0"
levels(newdata$y)[2] <- "1"

n <- round(2/3*nrow(newdata))
set.seed(1)
sub_train=sample(1:nrow(newdata),n)
train=newdata[sub_train,]  #训练集
test=newdata[-sub_train,]  #测试集

trainy <- as.numeric(as.factor(train[,13]))-1 #将因变量转化为numeric,且取值为0和1
trainx <- data.frame(train[,-13]) #训练集的x
colnames(trainx) <- names(train[,-13])
colxattr <- c(2,4,6,7,8,9,10,12)
trainxattr <- model.matrix(~.,data=trainx[,colxattr])[,-1] 
newtrainx <- cbind(trainx[,-colxattr],trainxattr)
newtrainx <- as.matrix(newtrainx)

param <- list(seed=1,objective="binary:logistic",max_depth=10,min_child_weight=0.1)
modelcv.xgb <- xgb.cv(data=newtrainx,label=trainy,params=param,nrounds=10,nfold=2)

#NN-----------------------------------------------------------------------------------
trainy <- model.matrix(~y-1,newdatatrain) #测试集的y
trainy <-data.frame(newdatatrain[,13])
trainx <- data.frame(newdatatrain[,-13]) #测试集的x
colnames(trainx) <- names(newdatatrain[,-13])

colxattr <- c(2,4,6,7,8,9,10,12)
trainxattr <- model.matrix(~.,data=trainx[,colxattr])[,-1] 
newtrainx <- cbind(trainx[,-colxattr],trainxattr)
newtrain <- as.data.frame(newtrain)


testy <- model.matrix(~y-1,newdatatest) #测试集的y
testy <-data.frame(newdatatest[,13])
testx <- data.frame(newdatatest[,-13]) #测试集的x
colnames(testx) <- names(newdatatest[,-13])

colxattr <- c(2,4,6,7,8,9,10,12)
testxattr <- model.matrix(~.,data=testx[,colxattr])[,-1] 
newtestx <- cbind(testx[,-colxattr],testxattr)

formulatrain <- as.formula(paste(paste(dimnames(newtrain)[[2]][c(1,2)],collapse = " + "),"~", paste(dimnames(newtrainx)[[2]], collapse = " + ")))
set.seed(1)
model.nn <- neuralnet(formulatrain,data=newtrain,hidden=c(9,6),linear.output=F)
predcv <- compute(model.nn,newtrainx) 
print(model.nn)

predcv$net.result 
predcvclass <- c("0","1")[apply(predcv$net.result, 1, which.max)]
predcvclass
table(predcvclass,train$y)

predcvclass<-as.factor(predcvclass)
per.values <- confusionMatrix(predcvclass,train$y,positive="1")
per.values
per.values$byClass
