#organize data and plot 
library(ggplot2)



#####training part

AllFIles=list.files("storage/")
Data=NULL

for (file in AllFIles){

Vec=read.csv(paste0("storage/",file,"/log.csv"),header = T)
Vec=Vec[Vec[,1]!="update",]
Vec$Task=strsplit(file,"_")[[1]][1]

NameVec=strsplit(file,"_")[[1]]

if(length(NameVec)==4){
  Vec$Method=NameVec[2]
  
}

if(length(NameVec)==5){
  Vec$Method=paste0(NameVec[c(2,3)],collapse = "_")
  
}

Data=rbind(Data,Vec)

}
unique(Data$Method)


Data$frames=as.numeric(Data$frames)
Data$rreturn_mean=as.numeric(Data$rreturn_mean)
Data$return_max=as.numeric(Data$return_max)
Data$num_frames_mean=as.numeric(Data$num_frames_mean)


unique(Data$Task)
#Task="MiniGrid-DLEnv-random-v0" 



for (Task in unique(Data$Task)){
  VecPlot=Data[Data$Task==Task,]
  
  
  head(VecPlot)
  
  
  ggplot(data = VecPlot, aes(x=frames, y=rreturn_mean,color=Method)) +
    geom_smooth(size=1)+ggtitle(Task)
  
  
  
  ggsave(filename = paste0("Images/",Task,"_RR.png"),height = 7,width = 11,device = "png")
  
  
  ggplot(data = VecPlot, aes(x=frames, y=return_max ,color=Method)) +
    geom_smooth(size=1)+ggtitle(Task)
  
  ggsave(filename = paste0("Images/",Task,"_ReturnMax.png"),height = 7,width = 11)
  
  
  
  ggplot(data = VecPlot, aes(x=frames, y=num_frames_mean ,color=Method)) +
    geom_smooth(size=1)+ggtitle(Task)
  
  ggsave(filename = paste0("Images/",Task,"_num_frames_mean.png"),height = 7,width = 11)
  

}



######evaluation part 

Data=read.csv("ExperimentalResults.csv",header = F)

head(Data)

names(Data)=c("Task","num_frames","fps","duration","return_per_episode","num_frames_per_episode")
tail(Data$return_per_episode)


Data$Reward=as.numeric(unlist(lapply(strsplit(Data$return_per_episode,split = " "),FUN = function(x){x[1]})))


Data$Episode_length=as.numeric(unlist(lapply(strsplit(Data$num_frames_per_episode,split = " "),FUN = function(x){x[1]})))

head(Data$Task)
Data$Method=unlist(lapply(strsplit(Data$Task,split = "_"),FUN = function(x){if(length(x)==5){return(x[3])} else {return(paste0(x[3],x[4],collapse ="[_]"))}}))


Data$Env=unlist(lapply(strsplit(Data$Task,split = "_"),FUN = function(x){x[1]}))

Data$TrainEnv=unlist(lapply(strsplit(Data$Task,split = "_"),FUN = function(x){x[2]}))


unique(Data$Env)

unique(Data$TrainEnv)

#VecPlot=Data[(Data$TrainEnv=="MiniGrid-DLEnv-random-v0")&(grepl(Data$Env,pattern = "Room")),]

VecPlot=Data[(Data$TrainEnv=="MiniGrid-DLEnv-random-v0"),]


ggplot(data = VecPlot, aes(x=Method, y=Reward,fill=Method)) +
  geom_boxplot()+ggtitle("minigrid OOD")
ggsave(filename = paste0("Images/",Task,"_OODEvalReward.png"),height = 7,width = 11)


ggplot(data = VecPlot, aes(x=Method, y=Reward,fill=Method)) +
  geom_boxplot()+ggtitle("minigrid OOD")+geom_violin()


ggplot(data = VecPlot, aes(x=Method, y=Episode_length,fill=Method)) +
  geom_boxplot()+ggtitle("minigrid OOD")



ggplot(data = VecPlot, aes(x=Method, y=Reward,fill=Method)) +
  geom_bar(stat="identity", position=position_dodge())+ggtitle("minigrid OOD")


ggplot(data = Data, aes(x=Env, y=Reward,fill=Method)) +
  geom_bar(stat="identity", position=position_dodge())+ggtitle("minigrid OOD")

