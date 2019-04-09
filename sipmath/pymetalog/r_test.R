# Title     : TODO
# Objective : TODO
# Created by: colinsmith
# Created on: 2/24/19


#library(rmetalog)
#data("fishSize")
#print(summary(fishSize))

#install.packages('devtools', repos='http://cran.us.r-project.org')
library(devtools)

load_all('/Users/colinsmith/Documents/Projects/sipmath/rmetalog-master/')

df <- read.csv('/Users/colinsmith/Documents/Projects/PyMetalog/ddms.csv', header = F, sep = ",", colClasses=c('numeric'))$V1
data("fishSize")
print(df)
#fishSize$FishSize
print(summary(fishSize))

my_metalog <- metalog(
  df,
  term_limit = 10,
  term_lower_bound = 9,
  bounds = c(0,60),
  boundedness = 'u',
  step_len = 0.01
  )
print(my_metalog$A)
print(my_metalog$Y)



print(summary(my_metalog))
#plot(my_metalog)
#load("/Users/colinsmith/Documents/Projects/sipmath/rmetalog-master/data/fishSize.RData")
#print(ls())

#write.csv(fishSize,
#  file="fishout.csv")