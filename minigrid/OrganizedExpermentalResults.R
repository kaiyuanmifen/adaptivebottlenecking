#Load the files

TargetFile="ExperimentalResults.csv"
Data=read.csv(TargetFile,header = F)
head(Data)

Reward=Reduce(strsplit(Data$V5,split = " "),f = rbind)
Reward=data.frame(Reward)
for (i in 1:ncol(Reward)){
  Reward[,i]=as.numeric(Reward[,i])  
}


N_frames=Reduce(strsplit(Data$V6,split = " "),f = rbind)
N_frames=data.frame(N_frames)
for (i in 1:ncol(N_frames)){
  N_frames[,i]=as.numeric(N_frames[,i])  
}

Data=cbind(Data[,c(1:4)],Reward,N_frames)
Data=data.frame(Data)
names(Data)=c("TaskName","Num_frames","fps","duration",
              "mean_reward","sd_reward","min_reward","max_reward",
              "mean_frame_per_episode","sd_frame_per_episode",
              "min_frame_per_episode","max_frame_per_episode")

Data$EnvsName=gsub("_.*$", "",Data$TaskName)
Data$Model=unlist(lapply(Data$TaskName,FUN = function(x){paste(strsplit(x,"[_]")[[1]][c(2:length(strsplit(x,"[_]")[[1]]))],collapse = "_")}))
head(Data)
rownames(Data)=NULL
#organize into nice tables 
AllEnvs=unique(Data$EnvsName)
print(unique(Data$Model))
AllModels=c("DLEnvs_LSTM" ,"DLEnvs_RIM","DLEnvs_RIMSP","DLEnvs_RIMSP_SEPred")
            # "MiniGrid-Empty-5x5-v0_LSTM",
            # "MiniGrid-Empty-5x5-v0_RIM",
            # "MiniGrid-Empty-5x5-v0_RIMSP",
            # "MiniGrid-MultiRoom-N2-S4-V0_LSTM", 
            # "MiniGrid-MultiRoom-N2-S4-V0_RIM",
            # "MiniGrid-MultiRoom-N2-S4-V0_RIMSP",
            # "MiniGrid-DoorKey-5x5-v0_LSTM",
            # "MiniGrid-DoorKey-5x5-v0_RIM",
            # "MiniGrid-DoorKey-5x5-v0_RIMSP")
for (Colname in c("mean_reward","sd_reward","min_reward","max_reward",
                 "mean_frame_per_episode","sd_frame_per_episode",
                 "min_frame_per_episode","max_frame_per_episode")){

Vec=Data[,c("Model","EnvsName",Colname)]
names(Vec)=c("Model","Env","Value")
Vec2=NULL
for (Model in AllModels){
  Vec3=Vec[Vec$Model==Model,]
  Vec2=rbind(Vec2,Vec3$Value[match(AllEnvs,Vec3$Env)])
} 
Vec2=data.frame(Vec2)
rownames(Vec2)=AllModels
colnames(Vec2)=AllEnvs
write.csv(Vec2,file = paste0("../figures/",Colname,".csv"),row.names = T)}
