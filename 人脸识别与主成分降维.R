library(caret)

faces<-read.table("faces.txt")
x<-as.matrix(faces)
library(RColorBrewer)
showMatrix <- function(x, ...) image(t(x[nrow(x):1,]), xaxt = 'none', yaxt = 'none', col = rev(colorRampPalette(brewer.pal(9, 'Greys'))(100)), ...)
showMatrix(matrix(x[1,],64,64))
par(mfrow=c(20,20),mar=rep(0,4))
for(i in 1:400){
  showMatrix(matrix(x[i,],64,64))
  rownames(x)<paste("P",rep(1:40,each=10)," 􀀀",rep(1:10,40),sep ="" )
}
library(MASS)
faces.class<-data.frame(cl=rep(1:40,each=10),faces)


#分类---------------------------------------------------------------------------------
library(class)
library(mclust)
library(caret)
library(klaR)
library(e1071)
library(psych)
library(rpart.plot)
library(rattle)
set.seed(1)
sub_train=sample(1:10,8)
idx<-c()
for (i in 0:39) {
  idx<-cbind(idx,c(10*i+1+sub_train))
}
train<-faces.class[idx,]
test<-faces.class[-idx,]
#lda
ptm<-proc.time();z<-lda(cl~.,data=train);proc.time()-ptm
plda<-predict(object=z,newdata=test[,-1])
perf.values.lda <- confusionMatrix(plda$class,factor(test$cl))
perf.values.lda$overall

#决策树
ptm<-proc.time();rp = rpart(cl~., data=train, method="class");proc.time()-ptm
pre <- predict(rp,test[,-1],type = "class")
perf.values.rp <- confusionMatrix(pre,factor(test[,1]))
perf.values.rp$overall

#knn
cl<-train[,1]
newtrain<-train[,-1]
tcl<-test[,1]
newtest<-test[,-1]
ptm<-proc.time();cl.knn<-knn(newtrain,newtest,cl,k=1,prob=T,use.all=T);proc.time()-ptm
pre.knn=as.factor(cl.knn)
perf.values.knn <- confusionMatrix(pre.knn,factor(tcl))
perf.values.knn$overall

#nb
ptm<-proc.time();nb<-naiveBayes(train, factor(cl));proc.time()-ptm
pred<-predict(nb,test)
pref.values.nb <- confusionMatrix(pred,factor(tcl))
pref.values.nb$overall

#pca降维-------------------------------------------------------------------
faces<-read.table("faces.txt")
faces<-as.matrix(faces)
cov<-cov(faces)
pca<-eigen(cov)#y由协方差阵计算主成分
sum(pca$value[1:70])/sum(pca$value)
faces<-data.frame(faces%*%pca$vectors[,c(1:70)])
faces.class<-data.frame(cl=rep(1:40,each=10),faces)

idx<-c()
for (i in 0:39) {
  idx<-cbind(idx,c(10*i+1+sub_train))
}
train<-faces.class[idx,]
test<-faces.class[-idx,]

#lda
ptm<-proc.time();z<-lda(cl~.,data=train);proc.time()-ptm
plda<-predict(object=z,newdata=test[,-1])
perf.values.lda <- confusionMatrix(plda$class,factor(test$cl))
perf.values.lda$overall

#决策树
ptm<-proc.time();rp = rpart(cl~., data=train, method="class");proc.time()-ptm
pre <- predict(rp,test[,-1],type = "class")
perf.values.rp <- confusionMatrix(pre,factor(test[,1]))
perf.values.rp$overall

#knn
cl<-train[,1]
newtrain<-train[,-1]
tcl<-test[,1]
newtest<-test[,-1]

cverr<-rep(0,10)
for(i in 1:10){
  set.seed(i)  # set the seed 
  kcl<-knn(newtrain,newtest,cl,k=i,use.all=T)
  cverr[i]<-classError(kcl,tcl)$errorRate
}
plot(1:10,cverr,type="l",main = "图2 KNN最优参数k")
kk<-which.min(cverr)
kk

ptm<-proc.time();cl.knn<-knn(newtrain,newtest,cl,k=1,prob=T,use.all=T);proc.time()-ptm
pre.knn=as.factor(cl.knn)
perf.values.knn <- confusionMatrix(pre.knn,factor(tcl))
perf.values.knn$overall

#nb
ptm<-proc.time();nb<-naiveBayes(train, factor(cl));proc.time()-ptm
pred<-predict(nb,test)
pref.values.nb <- confusionMatrix(pred,factor(tcl))
pref.values.nb$overall



#聚类---------------------------------------------------------------------------------
set.seed(1)
faces.k <- kmeans(faces, centers=40)
perf.values.1 <- confusionMatrix(factor(faces.k$cluster),factor(faces.class[,1]))
perf.values.1$overall



