install.packages("ggplot2")
#install.packages("readtext")
install.packages("psych")
install.packages("reshape2")

#library(readtext)
library(ggplot2)
library(psych)
library(reshape2)

# Set working directory
fpath <- "~/Documents/BookClub/BC2EmoCL"
setwd(fpath)

in_nameT <- "predict_in_6bmsrTr7*.txt"
flist <- list.files(path = fpath, pattern = "predict_in_6bmsrTr7kf*")

df1 <- data.frame(matrix(ncol = 6, nrow = 0))
colnames(df1) <- c("a", "ta", "p", "r", "f", "s")
df3 <- data.frame(matrix(ncol = 6, nrow = 0))
colnames(df3) <- c("a", "ta", "p", "r", "f", "s")

for (in_name in flist){
print(in_name)

in_test <- read.delim(in_name, sep = "", header = T, na.strings = " ", fill = T)
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
a1 <- length(t2r$V11)/(length(t2r$V11) + length(t2w$V11))
a3 <- length(t3r$V11)/(length(t3r$V11) + length(t3w$V11))
a1
a3

#Trusted Accuracy Meta
ta1 <- (sum(t2r$TrustFlag05 == 1)+sum(t2w$TrustFlag05 == 0))/(length(t2r$V11) + length(t2w$V11))
ta3 <- (sum(t3r$TrustFlagTr == 1)+sum(t3w$TrustFlagTr == 0))/(length(t3r$V11) + length(t3w$V11))
ta1
ta3

#Precision Meta
p1 <- sum(t2r$TrustFlag05 == 1)/(sum(t2r$TrustFlag05 == 1) + sum(t2w$TrustFlag05 == 1))
p3 <- sum(t3r$TrustFlagTr == 1)/(sum(t3r$TrustFlagTr == 1) + sum(t3w$TrustFlagTr == 1))
p1
p3


#Recall Meta
r1 = sum(t2r$TrustFlag05 == 1)/(sum(t2r$TrustFlag05 == 1) + sum(t2r$TrustFlag05 == 0)) #length(t2r$Score)
r3 = sum(t3r$TrustFlagTr == 1)/(sum(t3r$TrustFlagTr == 1) + sum(t3r$TrustFlagTr == 0))
r1
r3


#F1
f1 <- 2. * p1 * r1 / (p1 + r1)
f3 <- 2. * p3 * r3 / (p3 + r3)
f1
f3

#Specificity Meta
s1 <- sum(t2w$TrustFlag05 == 0)/(sum(t2w$TrustFlag05 == 0) + sum(t2w$TrustFlag05 == 1)) #length(t2w$Score)
s3 <- sum(t3w$TrustFlagTr == 0)/(sum(t3w$TrustFlagTr == 0) + sum(t3w$TrustFlagTr == 1))
s1
s3

df1[nrow(df1) + 1,] <- c(a1, ta1, p1, r1, f1, s1) 
df3[nrow(df3) + 1,] <- c(a3, ta3, p3, r3, f3, s3) 

}

mean(df1$a)
mean(df1$ta)
mean(df1$p)
mean(df1$r)
mean(df1$f)
mean(df1$s)

mean(df3$a)
mean(df3$ta)
mean(df3$p)
mean(df3$r)
mean(df3$f)
mean(df3$s)

sd(df1$a)
sd(df1$ta)
sd(df1$p)
sd(df1$r)
sd(df1$f)
sd(df1$s)

sd(df3$a)
sd(df3$ta)
sd(df3$p)
sd(df3$r)
sd(df3$f)
sd(df3$s)

wilcox.test(df1$a, df3$a, paired = TRUE, alternative = "two.sided")
wilcox.test(df1$ta, df3$ta, paired = TRUE, alternative = "two.sided")
wilcox.test(df1$p, df3$p, paired = TRUE, alternative = "two.sided")
wilcox.test(df1$r, df3$r, paired = TRUE, alternative = "two.sided")
wilcox.test(df1$f, df3$f, paired = TRUE, alternative = "two.sided")
wilcox.test(df1$s, df3$s, paired = TRUE, alternative = "two.sided")
