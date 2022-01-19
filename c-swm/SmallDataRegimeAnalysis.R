Data=read.csv("Eval_results/Different_targets/shapes_eval_eval_results.csv",header = F)

head(Data)
tail(Data)
dim(Data)
MRR_Baseline=Data$V6[grepl(Data$V1,pattern = "orignal")]
MRR_DVNC=Data$V6[grepl(Data$V1,pattern = "VQVAE")]

HITs_Baseline=Data$V5[grepl(Data$V1,pattern = "orignal")]
HITs_DVNC=Data$V5[grepl(Data$V1,pattern = "VQVAE")]


mean(MRR_DVNC)
sd(MRR_DVNC)
mean(MRR_Baseline)
sd(MRR_Baseline)
boxplot(list(MRR_Baseline,MRR_DVNC),names=c("GNN","GNN+DVNC"))

wilcox.test(MRR_Baseline,MRR_DVNC)



mean(HITs_DVNC)
sd(HITs_DVNC)
mean(HITs_Baseline)
sd(HITs_Baseline)
boxplot(list(HITs_Baseline,HITs_DVNC),names=c("GNN","GNN+DVNC"))

wilcox.test(HITs_Baseline,HITs_DVNC)
