#organize data and plot 
library(ggplot2)


AllFIles=list.files("Eval_results/AdaptiveQuantization/")

#file=AllFIles[1]

DataTable=NULL


for (file in AllFIles){

Data=read.csv(paste0("Eval_results/AdaptiveQuantization/",file),header = F)
tail(Data)


Task=paste0(strsplit(file,split ="_")[[1]][1:3],collapse = "_")

names(Data)=c("name","Quantization_target","average_test_loss",
              "k","hits_at_k","RR","Extraloss","All_CB_indexes")

Data$Method=""

for (i in 1:nrow(Data)){
  
  Vec=strsplit(Data$name[i],"_")[[1]]
  idx1=which(Vec=="true")
  idx2=which(Vec=="edge")
  if((idx2-idx1)==5){
    Data$Method[i]=paste0(Vec[idx1+2],"_",Vec[idx1+3])
  }

  if((idx2-idx1)==4){
    Data$Method[i]=paste0(Vec[idx1+2])
  }
  
}

head(Data)

Data$Method


VecPlot=Data

tail(VecPlot)


ggplot(data = VecPlot, aes(x=Method, y=RR,fill=Method)) +
  geom_boxplot()+ggtitle(Task)



ggsave(filename = paste0("Images/",Task,"_RR.png"),height = 7,width = 11,device = "png")


ggplot(data = VecPlot, aes(x=Method, y=hits_at_k,fill=Method)) +
  geom_boxplot()+ggtitle(Task)

ggsave(filename = paste0("Images/",Task,"_hits.png"),height = 7,width = 11)

ggplot(data = VecPlot, aes(x=Method, y=average_test_loss,fill=Method)) +
  geom_boxplot()+ggtitle(Task)

ggsave(filename = paste0("Images/",Task,"_testloss.png"),height = 7,width = 11)


###organize the data into a table
names(Data)
Vec=Data[,c("hits_at_k","RR","Extraloss","average_test_loss")]
unique(Vec$Method)
MeanValues=aggregate(Vec,by=list(Data$Method),mean)
names(MeanValues)[1]="Method"
SdValues=aggregate(Vec,by=list(Data$Method),sd)
names(SdValues)=c("Method","SD_hits_at_k","SD_RR","SD_Extraloss","SD_average_test_loss")
Values=cbind(MeanValues,SdValues[,c(2:5)])
Values$Task=Task

DataTable=rbind(DataTable,Values)
}


#####Organize the data table
library(openxlsx)


head(DataTable)

DataTable_iid=DataTable[!grepl(DataTable$Task,pattern = "OOD"),]

DataTable_ood=DataTable[grepl(DataTable$Task,pattern = "OOD"),]

write.xlsx(DataTable_iid, 'Images/cswm_iid.xlsx')
write.xlsx(DataTable_ood, 'Images/cswm_ood.xlsx')
