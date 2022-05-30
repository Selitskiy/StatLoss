install.packages("ggplot2")
install.packages("readtext")
install.packages("psych")
install.packages("reshape2")

library(readtext)
library(ggplot2)
library(psych)
library(reshape2)

# Set working directory
setwd("~/BookClub/BC2EmoCL")


# Load Expression Recognition Heatmap, removing headers
#grep "\"" result_in_6bfemsr6.txt > result_in_6bfemsr6_nh.txt

in_name <- "predict_in_6bfmsrTr7.txt"
in_test <- read.delim(in_name, sep = "", header = T, na.strings = " ", fill = T)
mr <- nrow(in_test)

#Process
t2 <- in_test[,1:7]
t2[,8] <- FALSE


for (i in 1:mr){
  if( t2$CorrectClass[i] == t2$MaxScoreClass[i] ){
    t2[i,8] <- TRUE
  }
}

t2r <- t2[t2$V8 == TRUE,]
t2w <- t2[t2$V8 != TRUE,]



#Untrusted Accuracy
length(t2r$V8)/(length(t2r$V8) + length(t2w$V8))

#Trusted Accuracy Meta
(sum(t2r$TrustFlag05 == 1)+sum(t2w$TrustFlag05 == 0))/(length(t2r$V8) + length(t2w$V8))
(sum(t2r$TrustFlagTr == 1)+sum(t2w$TrustFlagTr == 0))/(length(t2r$V8) + length(t2w$V8))

#Precision Meta
sum(t2r$TrustFlag05 == 1)/(sum(t2r$TrustFlag05 == 1) + sum(t2w$TrustFlag05 == 1))
sum(t2r$TrustFlagTr == 1)/(sum(t2r$TrustFlagTr == 1) + sum(t2w$TrustFlagTr == 1))

#Recall Meta
sum(t2r$TrustFlag05 == 1)/(sum(t2r$TrustFlag05 == 1) + sum(t2r$TrustFlag05 == 0)) #length(t2r$Score)
sum(t2r$TrustFlagTr == 1)/(sum(t2r$TrustFlagTr == 1) + sum(t2r$TrustFlagTr == 0))

#Specificity Meta
sum(t2w$TrustFlag05 == 0)/(sum(t2w$TrustFlag05 == 0) + sum(t2w$TrustFlag05 == 1)) #length(t2w$Score)
sum(t2w$TrustFlagTr == 0)/(sum(t2w$TrustFlagTr == 0) + sum(t2w$TrustFlagTr == 1))




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