install.packages("ggplot2")
#install.packages("readtext")
install.packages("psych")
install.packages("reshape2")

#library(readtext)
library(ggplot2)
library(psych)
library(reshape2)

# Set working directory
setwd("~/Documents/BookClub/BC2EmoCL")


# Load Expression Recognition Heatmap, removing headers
#grep "\"" result_in_6bfemsr6.txt > result_in_6bfemsr6_nh.txt

in_name <- "predict_in_6bmsrTr7_8192_01_200_ol.txt"
in_test <- read.delim(in_name, sep = "", header = T, na.strings = " ", fill = T)
in_name2 <- "predict_in_6bmsrTr7_8192_01_200_wol.txt"
in_test2 <- read.delim(in_name2, sep = "", header = T, na.strings = " ", fill = T)
in_name3 <- "predict_in_6bmsrTr7_8192_01_200_nol.txt"
in_test3 <- read.delim(in_name3, sep = "", header = T, na.strings = " ", fill = T)
mr <- nrow(in_test)

#Process Max Ensemble vote
t2 <- in_test[,1:10]
t2[,11] <- FALSE


for (i in 1:mr){
  if( t2$CorrectClass[i] == t2$PredictClassMax[i] ){
    t2[i,11] <- TRUE
  }
}

t2r <- t2[t2$V11 == TRUE,]
t2w <- t2[t2$V11 != TRUE,]


#Process Ensemble vote equal to Trustowrthy vote
t3 <- in_test[,1:10]
t3[,11] <- FALSE


for (i in 1:mr){
  if( t3$CorrectClass[i] == t3$PredictClassTr[i] ){
    t3[i,11] <- TRUE
  }
}

t3r <- t3[t3$V11 == TRUE,]
t3w <- t3[t3$V11 != TRUE,]



#Untrusted Accuracy
length(t2r$V11)/(length(t2r$V11) + length(t2w$V11))
length(t3r$V11)/(length(t3r$V11) + length(t3w$V11))

#Trusted Accuracy Meta
(sum(t2r$TrustFlag05 == 1)+sum(t2w$TrustFlag05 == 0))/(length(t2r$V11) + length(t2w$V11))
(sum(t2r$TrustFlagTr == 1)+sum(t2w$TrustFlagTr == 0))/(length(t2r$V11) + length(t2w$V11))
(sum(t3r$TrustFlagTr == 1)+sum(t3w$TrustFlagTr == 0))/(length(t3r$V11) + length(t3w$V11))
(sum(t3r$TrustFlagCurr == 1)+sum(t3w$TrustFlagCurr == 0))/(length(t3r$V11) + length(t3w$V11))
(sum((t3r$TrustFlagCurr == 1) & (t3r$TrustThresholdCurr >= 0.5))+sum((t3w$TrustFlagCurr == 0) | (t3w$TrustThresholdCurr < 0.5)))/(length(t3r$V11) + length(t3w$V11))

#Precision Meta
p1 = sum(t2r$TrustFlag05 == 1)/(sum(t2r$TrustFlag05 == 1) + sum(t2w$TrustFlag05 == 1))
p2 = sum(t2r$TrustFlagTr == 1)/(sum(t2r$TrustFlagTr == 1) + sum(t2w$TrustFlagTr == 1))
p3 = sum(t3r$TrustFlagTr == 1)/(sum(t3r$TrustFlagTr == 1) + sum(t3w$TrustFlagTr == 1))
p4 = sum(t3r$TrustFlagCurr == 1)/(sum(t3r$TrustFlagCurr == 1) + sum(t3w$TrustFlagCurr == 1))
p5 = sum((t3r$TrustFlagCurr == 1) & (t3r$TrustThresholdCurr >= 0.5))/(sum((t3r$TrustFlagCurr == 1) & (t3r$TrustThresholdCurr >= 0.5)) + sum((t3w$TrustFlagCurr == 1) & (t3w$TrustThresholdCurr >= 0.5)))
p1
p2
p3
p4
p5

#Recall Meta
r1 = sum(t2r$TrustFlag05 == 1)/(sum(t2r$TrustFlag05 == 1) + sum(t2r$TrustFlag05 == 0)) #length(t2r$Score)
r2 = sum(t2r$TrustFlagTr == 1)/(sum(t2r$TrustFlagTr == 1) + sum(t2r$TrustFlagTr == 0))
r3 = sum(t3r$TrustFlagTr == 1)/(sum(t3r$TrustFlagTr == 1) + sum(t3r$TrustFlagTr == 0))
r4 = sum(t3r$TrustFlagCurr == 1)/(sum(t3r$TrustFlagCurr == 1) + sum(t3r$TrustFlagCurr == 0))
r5 = sum((t3r$TrustFlagCurr == 1) & (t3r$TrustThresholdCurr >= 0.5))/(sum((t3r$TrustFlagCurr == 1) & (t3r$TrustThresholdCurr >= 0.5)) + sum((t3r$TrustFlagCurr == 0) | (t3r$TrustThresholdCurr < 0.5)))
r1
r2
r3
r4
r5

#F1
2. * p1 * r1 / (p1 + r1)
2. * p2 * r2 / (p2 + r2)
2. * p3 * r3 / (p3 + r3)
2. * p4 * r4 / (p4 + r4)
2. * p5 * r5 / (p5 + r5)

#Specificity Meta
sum(t2w$TrustFlag05 == 0)/(sum(t2w$TrustFlag05 == 0) + sum(t2w$TrustFlag05 == 1)) #length(t2w$Score)
sum(t2w$TrustFlagTr == 0)/(sum(t2w$TrustFlagTr == 0) + sum(t2w$TrustFlagTr == 1))
sum(t3w$TrustFlagTr == 0)/(sum(t3w$TrustFlagTr == 0) + sum(t3w$TrustFlagTr == 1))
sum(t3w$TrustFlagCurr == 0)/(sum(t3w$TrustFlagCurr == 0) + sum(t3w$TrustFlagCurr == 1))
sum((t3w$TrustFlagCurr == 0) | (t3w$TrustThresholdCurr < 0.5))/(sum((t3w$TrustFlagCurr == 0) | (t3w$TrustThresholdCurr < 0.5)) + sum((t3w$TrustFlagCurr == 1) & (t3w$TrustThresholdCurr >= 0.5)))



# Change of trust threshold graph
t4 <- in_test[,1:10]
t4[,11] <- in_test2[,7]

ggplot(data=t4, aes(x = as.numeric(row.names(t4)))) +
  geom_line(linetype=1, color = 'darkgreen', aes(y=TrustThresholdCurr)) + 
  geom_line(linetype=2, color = 'red', aes(y=V11)) +
  geom_line(linetype=3, color = 'blue', aes(y=TrustThresholdTr)) +
  xlab("Images") + ylab("Trusted Threshold")





#Accuracy - trust threshold diagramm
t5 <- in_test[,1:11]
for (i in 1:mr){
  tmp <- unlist(strsplit(as.character(in_test[i,11]), "/"))
  tl <- length(tmp)
  t5[i,12] <- unlist(tmp)[tl-1]
}
t5 <- t5[order(t5$V12),]

df <- data.frame()

i_s <- 0
s_cur <- ""

tp_abs <- 0
tn_abs <- 0

tp_cur <- 0
tn_cur <- 0
fp_cur <- 0
fn_cur <- 0
tr_ac <- 0.
ac <- 0.
ac_t <- 0.
pr <- 0.
rc <- 0.
tr_mean <- 0.

for (i in 1:mr){
  
  # Change of session
  if(s_cur != t5[i,12]){
    
    if(i_s != 0){
      ac <- tp_abs / (tp_abs + tn_abs)
      ac_t <- (tp_cur + tn_cur) / (tp_cur + tn_cur + fp_cur + fn_cur)
      pr <- tp_cur / (tp_cur + fp_cur)
      rc <- tp_cur / (tp_cur + fn_cur)
      
      tr_mean <- tr_ac / (i - i_s)
      df[nrow(df) + 1,1:5] <- c(tr_mean, ac, ac_t, pr, rc)
    }
    
    s_cur <- t5[i,12]
    tp_abs <- 0
    tn_abs <- 0
    tp_cur <- 0
    tn_cur <- 0
    fp_cur <- 0
    fn_cur <- 0
    tr_ac <- 0.
    ac <- 0.
    ac_t <- 0.
    pr <- 0.
    rc <- 0.
    tr_mean <- 0.
    i_s <- i
  }
  
  #Step increments
  if( t5$CorrectClass[i] == t5$PredictClassTr[i] ){
    tp_abs <- tp_abs + 1
    if(t5$TrustFlagCurr[i] == 1){
      tp_cur <- tp_cur + 1
    }
    else{
      fn_cur <- fn_cur + 1
    }
  }
  else{
    tn_abs <- tn_abs + 1
    if(t5$TrustFlagCurr[i] == 1){
      fp_cur <- fp_cur + 1
    }
    else{
      tn_cur <- tn_cur + 1
    }
  }
  
  tr_ac <- tr_ac + t5$TrustThresholdCurr[i]
  
}
ac <- tp_abs / (tp_abs + tn_abs)
ac_t <- (tp_cur + tn_cur) / (tp_cur + tn_cur + fp_cur + fn_cur)
pr <- tp_cur / (tp_cur + fp_cur)
rc <- tp_cur / (tp_cur + fn_cur)

tr_mean <- tr_ac / (i - i_s)
df[nrow(df) + 1,1:5] <- c(tr_mean, ac, ac_t, pr, rc)
#names(df) <- c("Tr_mean", "Acc", "Acc_t", "Precission", "Recall")

ggplot(data=df, aes(x = V1)) +
  #geom_point(color = 'black', aes(y=V2)) + 
  geom_point(color = 'red', aes(y=V3)) + 
  #geom_point(color = 'darkgreen', aes(y=V4)) +
  #geom_point(color = 'blue', aes(y=V5)) + 
  xlab("Mean Threshold") + ylab("Accuracy")






# -> Previous leftovers
# Load Expression Recognition Heatmap, removing headers
#grep "\"" result_in_6bfmsr6.txt > result_in_6bfmsr6_nh.txt
hname <- "result_vgg_6bfmsr6_nh.txt"
h_test <- read.delim(hname, sep = "", header = F, na.strings = " ", fill = T)
#hr <- nrow(h_test)

th <- h_test[,1:2]
th[,2:9] <- h_test[,5:12]

names(th) <- c("Expr", "AN", "CE", "DS", "HP", "NE", "SA", "SC", "SR")
rownames(th) <- th[,1]
#th[,1] <- NULL

thm <- melt(th)
names(thm) <- c("Expression", "Guess", "Score")
               
ggplot(thm, aes(x = Expression, Guess)) +
  geom_tile(aes(fill = Score), alpha=0.9) +
  geom_text(aes(label = round(Score, 4)))