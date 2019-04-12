library(Rwordseg)
library(tmcn)
library(tm)
library(jiebaR)
library(wordcloud2)
mydata <- read.csv('C:/Users/tenns/Documents/文本挖掘/comments_trans.csv',header=TRUE,stringsAsFactors =F,sep=",")
comments_segged <- segmentCN(strwords = mydata$comment)
#加载停用词
stop_words=readLines('C:/Users/tenns/Documents/文本挖掘/stopwords.txt',encoding="UTF-8")
#加入用户词典
installDict('C:/Users/tenns/Documents/文本挖掘/userwords.txt',dictname = 'mobile comments',dicttype = 'text')
#去除停用词和长度小于1 的词
removewords <- function(target_words, stop_words) {
  target_words <- target_words[target_words %in% stop_words==F]
  target_words <- target_words[nchar(target_words)>1]
  return(target_words)
}
comments_segged <- sapply(comments_segged, removewords, stop_words)
d1 <- createDTM(comments_segged)

#词云
f1 <- createWordFreq(d1)
wordcloud2(f1,fontWeight='bold',fontFamily = "微软雅黑")

#特征选择
selectFeaturesCHI <- function(dtm, classvec, n = 50) {
  OUT <- list()
  uniclass <- unique(classvec)
  for(i in 1:length(uniclass)) {
    tmp.chi <- data.frame(word = colnames(dtm), N = nrow(dtm), stringsAsFactors = FALSE)
    tmp.chi$A <- apply(dtm[classvec == uniclass[i],], 2, FUN = function(X) sum(X > 0))
    tmp.chi$C <- apply(dtm[classvec == uniclass[i],], 2, FUN = function(X) sum(X == 0))
    tmp.chi$B <- apply(dtm[classvec != uniclass[i],], 2, FUN = function(X) sum(X > 0))
    tmp.chi$D <- apply(dtm[classvec != uniclass[i],], 2, FUN = function(X) sum(X == 0))
    tmp.chi$LOGCHI <- log(tmp.chi$N) + 2*log(abs(tmp.chi$A * tmp.chi$D - tmp.chi$C * tmp.chi$B)) - 
      log(tmp.chi$A + tmp.chi$C) - log(tmp.chi$B + tmp.chi$D) - log(tmp.chi$A + tmp.chi$B) - log(tmp.chi$C + tmp.chi$D)
    tmp.chi$chi <- exp(tmp.chi$LOGCHI)
    OUT[[i]] <- tmp.chi[order(tmp.chi$chi, decreasing = TRUE), c("word", "chi")]
    rownames(OUT[[i]]) <- NULL
    OUT[[i]] <- OUT[[i]][1:n, ]
  }
  names(OUT) <- uniclass
  return(OUT)
}

varlist1 <- selectFeaturesCHI(d1, mydata$score_trans, 50)
var1 <- unique(unlist(lapply(varlist1, "[[", "word")))


d2 <- weightTfIdf(d1)

tfdata <- as.data.frame(as.matrix(d1[, var1]))
tfidfdata <- as.data.frame(as.matrix(d2[, var1]))
tfdata$class <- factor(mydata$score_trans)
tfidfdata$class <- factor(mydata$score_trans)

install.packages("Biobase")
library(pROC)
library(caret)
library(e1071)
# 朴素贝叶斯
model1 <- naiveBayes(class~., data = tfdata)
pred1_class <- predict(model1, tfdata, type = "class")
pred1_prob <- predict(model1, tfdata, type = "raw")[,1]
confusionMatrix(tfdata$class, pred1_class, positive = '1')
roc1 <- roc(tfdata$class, pred1_prob)
plot(roc1, print.auc = TRUE, print.thres = TRUE)

model2 <- naiveBayes(class~., data = tfidfdata)
pred2_class <- predict(model2, tfidfdata, type = "class")
pred2_prob <- predict(model2, tfidfdata, type = "raw")[, "1"]
confusionMatrix(tfidfdata$class, pred2_class, positive = "1")
roc2 <- roc(tfidfdata$class, pred2_prob)
plot(roc2, print.auc = TRUE, print.thres = TRUE)

# 10折交叉验证（TF）
num <- sample(1:10, nrow(tfdata), replace = TRUE)
res <- list()
n <- ncol(tfdata)
for ( i in 1:10) {
  train <- tfdata[num!=i, ]
  test <- tfdata[num==i, ]
  m0 = naiveBayes(class~., data = train)
  pred0 <- predict(m0, test, type = "class")
  res[[i]] <- confusionMatrix(pred0, test$class)
}
sapply(res, "[[", "overall")
mean(sapply(res, "[[", "overall")[1, ])

# 10折交叉验证（TF-IDF）
num <- sample(1:10, nrow(tfidfdata), replace = TRUE)
res <- list()
n <- ncol(tfidfdata)
for ( i in 1:10) {
  train <- tfidfdata[num!=i, ]
  test <- tfidfdata[num==i, ]
  m0 = naiveBayes(class~., data = train)
  pred0 <- predict(m0, test, type = "class")
  res[[i]] <- confusionMatrix(pred0, test$class)
}
sapply(res, "[[", "overall")
mean(sapply(res, "[[", "overall")[1, ])

# SVM
svmModel <- svm(class ~ ., data = tfdata, probability = TRUE)
pred3 <- predict(svmModel, tfdata, probability = TRUE)
pred3_class <- factor(as.vector(pred3))
pred3_prob <- attributes(pred3)$probabilities[, "1"]
confusionMatrix(tfdata$class, pred3_class, positive = "1")

roc3 <- roc(tfdata$class, pred3_prob)
plot(roc3, print.auc = TRUE, print.thres = TRUE)

#LDA模型
library(lda)
library(LDAvis)
w1 = comments_segged
comments = as.list(comments_segged)
doc.list <- strsplit(as.character(comments),split=" ")
f1 <- createWordFreq(d1)
f1 <- f1[f1$freq >=5, ]
get.terms <- function(x) {
  index <- match(x, f1$word) # 获取词的ID
  index <- index[!is.na(index)] #去掉没有查到的，也就是去掉了的词
  rbind(as.integer(index - 1), as.integer(rep(1, length(index)))) 
}#生成上图结构
documents <- lapply(doc.list, get.terms)
K <- 5 #主题数
G <- 5000 #迭代次数
alpha <- 0.10 
eta <- 0.02
set.seed(666) 
fit <- lda.collapsed.gibbs.sampler(documents = documents, K = K, vocab =  f1$word, num.iterations = G, alpha = alpha, eta = eta, initial = NULL, burnin = 0, compute.log.likelihood = TRUE)
theta <- t(apply(fit$document_sums + alpha, 2, function(x) x/sum(x))) #文档—主题分布矩阵
phi <- t(apply(t(fit$topics) + eta, 2, function(x) x/sum(x))) #主题-词语分布矩阵
doc.length <- sapply(documents, function(x) sum(x[2, ])) #每篇文章的长度，即有多少个词

json <- createJSON(phi = phi, theta = theta, 
                   doc.length = doc.length, vocab = f1$word,
                   term.frequency = f1$freq)#json为作图需要数据，下面用servis生产html文件，通过out.dir设置保存位置
serVis(json, out.dir = 'C:/Users/tenns/Documents/文本挖掘/vis', open.browser = FALSE)
writeLines(iconv(readLines("C:/Users/tenns/Documents/文本挖掘/vis/lda.json"), from = "GBK", to = "UTF8"), 
           file("C:/Users/tenns/Documents/文本挖掘/vis/lda.json", encoding="UTF-8"))
serVis(json)
#特征绘图
library('ggplot2')
indexes = c()
feature_list = list('物流'=c('物流','快递'), '摄像头'=c('摄像头','拍照'), '外观'=c('外观','刘海','边框'), 'face id'=c('识别','面部','faceid','face','解锁'),
  '价格'=c('价格'), '速度'=c('速度'), '屏幕'=c('屏幕'), '手感'=c('手感'), '发热'=c('发热'), '系统'=c('系统','死机'),
  '售后'=c('售后'), '电池'=c('电池'), '信号'=c('信号'), '质量'=c('黑屏','划痕','黑屏','灰尘'))
for (i in (1:length(feature_list))){
  scores = c()
  for (j in (1:length(comments_segged))){
    if (length(intersect(unlist(feature_list[i]),unlist(comments_segged[j])))>0){
      scores = append(scores , mydata$score[j])
    }
  }
  index = mean(scores)
  indexes = append(indexes,index)
}
factor_data = data.frame(index = indexes,factor = names(feature_list))
factor_data2 = factor_data[order(-factor_data[index]),]
ggplot(factor_data2,aes(x=reorder(factor,index), y=index,fill=factor))+geom_col( position="dodge")+coord_flip()

#情感分析   worked
library(qdap)
data(NTUSD)

dict1 <- sentiment_frame(positives = NTUSD$positive_chs, negatives = NTUSD$negative_chs, pos.weights = 1, neg.weights = -1)

p1 <- polarity(comments_segged, polarity.frame = dict1)
p1$all
p1$group#情感得分平均值


p2<-data.frame(p1$all$polarity,mydata$score)

p2=p2[which(rowSums(p2==0)==0),]#去除0值

sum(p2$p1.all.polarity)/length(p2$p1.all.polarity)#去除0指后情感得分平均值

list1=c()
for(i in 1:length(p2$p1.all.polarity)){
  if((p2$p1.all.polarity[i]>=0&p2$mydata.score[i]>3)|(p2$p1.all.polarity[i]<=0&p2$mydata.score[i]<=3)){
    list1=c(list1,1)
  }
  else{
    list1=c(list1,0)
  }
}#计算情感的分正确率
sum(list1)#情感的分正确个数
sum(list1)/length(p2$p1.all.polarity)#情感得分正确率





# 关联规则
library(arules)
library(arulesViz)

para1 <- unlist(strsplit(mydata$comment, split = "\r\n"))
para2 <- para1[nchar(para1) > 10]
para3 <- segmentCN(para2, returnType = "tm")
zz <- file("para3.txt", "w", encoding = "UTF-8")
writeLines(para3, con = zz)
close(zz)

tr1 <- read.transactions("para3.txt", format = "basket", sep = " ", encoding = "UTF-8")
arules::inspect(tr1[1:5])

rules2 <- apriori(tr1, parameter = list(support=0.001, confidence=0.5))
arules::inspect(rules2)

plot(rules2)


