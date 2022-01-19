#organize data and plot 
library(ggplot2)

Data=read.csv("../../MARL_Results.csv",header = F)
tail(Data)
names(Data)=c("Data","N_agents","Method","Episode","trainReward","testReward","OODtestReward","ExtraLoss")

#names(Data)=c("Data","N_agents","Method","Episode",
 #             "trainReward","testReward","OODtestReward","CBLoss",
  #            "AllAttScores_train","AllAttScores_test","AllAttScores_OODtest")
tail(Data)

#Data=Data[Data$N_agents%in%c("1","5","20"),]

unique(Data$N_agents)

N_agent=5

names(Data)

unique(Data$N_agents)

data="Drone"

Data$testReward=as.numeric(Data$testReward)

Data$OODtestReward=as.numeric(Data$OODtestReward)


head(Data)
tail(Data)
table(Data$trainReward)
VecPlot=Data[(!is.na(Data$Episode))&(Data$Episode>=00)&(Data$N_agents==N_agent),]

tail(VecPlot)

table(VecPlot$OODtestReward)

unique(VecPlot$testReward)

unique(VecPlot$Episode)
VecPlot$Episode=as.numeric(VecPlot$Episode)

ggplot(data = VecPlot, aes(x=Episode, y=testReward)) +
  geom_line()+scale_color_brewer(palette="Paired")

ggplot(data = VecPlot, aes(x=Episode, y=trainReward)) +
  geom_line()+scale_color_brewer(palette="Paired")



ggplot(data = VecPlot, aes(x=Episode, y=OODtestReward)) +
  geom_line()+scale_color_brewer(palette="Paired")


ggplot(data = VecPlot, aes(x=Episode, y=testReward)) +
  geom_smooth(size=2)+scale_color_brewer(palette="Paired")

ggplot(data = VecPlot, aes(x=Episode, y=trainReward)) +
  geom_smooth(size=2)+scale_color_brewer(palette="Paired")


ggplot(data = VecPlot, aes(x=Episode, y=OODtestReward)) +
  geom_smooth(size=2)+scale_color_brewer(palette="Paired")



ggplot(data = VecPlot, aes(x=Episode, y=testReward,color=Method)) +
  geom_smooth(size=2)+ggtitle(paste0(data," N_agents:",N_agent))



ggplot(data = VecPlot, aes(x=Episode, y=trainReward,color=Method)) +
  geom_smooth(size=2)+ggtitle(paste0(data," N_agents:",N_agent))



ggplot(data = VecPlot, aes(x=Episode, y=OODtestReward,color=Method)) +
  geom_smooth(size=2)+ggtitle(paste0(data," N_agents:",N_agent))



ggplot(data = VecPlot, aes(x=Episode, y=Q1,color=Method)) +
  geom_smooth(size=2)+ggtitle(paste0(data," N_agents:",N_agent))+xlim(0, 50)

VecPlot=Data[Data$Method=="Adaptive_Quantization",]

p=ggplot(data = VecPlot, aes(x=Episode, y=Q1_train)) +
  geom_smooth(size=2,color="blue")+ggtitle(paste0(data," N_agents:",N_agent))+
  xlim(0, 50)+
  ylab("Count")

p+geom_smooth(size=2,color="red",aes(x=Episode, y=Q2_train))+
  geom_smooth(size=2,color="green",aes(x=Episode, y=Q3_train))




p=ggplot(data = VecPlot, aes(x=Episode, y=Q1_test)) +
  geom_smooth(size=2,color="blue")+ggtitle(paste0(data," N_agents:",N_agent))+
  xlim(0, 50)+
  ylab("Count")

p+geom_smooth(size=2,color="red",aes(x=Episode, y=Q2_test))+
  geom_smooth(size=2,color="green",aes(x=Episode, y=Q3_test))




p=ggplot(data = VecPlot, aes(x=Episode, y=Q1_OODtest)) +
  geom_smooth(size=2,color="blue")+ggtitle(paste0(data," N_agents:",N_agent))+
  xlim(0, 50)+
  ylab("Count")

p+geom_smooth(size=2,color="red",aes(x=Episode, y=Q2_OODtest))+
  geom_smooth(size=2,color="green",aes(x=Episode, y=Q3_OODtest))





plot(Data$Q1~Data$Episode,type='p')



ggplot(data = VecPlot, aes(x=Episode, y=OODtestReward,color=Method)) +
  geom_smooth(size=2)+ggtitle(data)+ggtitle(paste0(data," N_agents:",N_agent))+xlim(0, 200)



ggplot(data = VecPlot, aes(x=Episode, y=testReward,color=Method)) +
  geom_smooth(size=2)+xlim(0, 100)



ggplot(data = VecPlot, aes(x=Episode, y=trainReward,color=Method)) +
  geom_smooth(size=2)+xlim(0, 100)


ggplot(data = VecPlot, aes(x=Episode, y=OODtestReward,color=Method)) +
  geom_smooth(size=2)+xlim(0, 300)



ggplot(data = VecPlot, aes(x=Episode, y=OODtestReward)) +
  geom_smooth(size=2)+scale_color_brewer(palette="Paired")


ggplot(data = VecPlot, aes(x=Episode, y=trainReward)) +
  geom_smooth(size=2)+scale_color_brewer(palette="Paired")


ggplot(data = VecPlot, aes(x=Episode, y=trainReward,linetype=hypothesis,colour=hypothesis)) +
  geom_smooth(size=2)+scale_color_brewer(palette="Paired")



ggplot(data = VecPlot, aes(x=Episode, y=testReward,linetype=hypothesis,colour=hypothesis)) +
  geom_smooth(size=2)+scale_color_brewer(palette="Paired")


ggplot(data = VecPlot, aes(x=Episode, y=OODtestReward,linetype=hypothesis,colour=hypothesis)) +
  geom_smooth(size=2)+scale_color_brewer(palette="Paired")





VecPlot=Data[(!is.na(Data$Episode))&(Data$Episode>40)&(Data$N_agents==N_agent),]


ggplot(VecPlot, aes(x=Method, y=OODtestReward,color=Method)) + 
  geom_boxplot()+ coord_flip()+
    ggtitle(paste("Done env, N_agent=",N_agent))

ggsave(filename = paste0("../Images/Drone/OODTest_",N_agent,".png"),height = 7,width = 11)



ggplot(VecPlot, aes(x=Method, y=testReward,color=Method)) + 
  geom_boxplot()+ coord_flip()+
  ggtitle(paste("Done env, N_agent=",N_agent))

ggsave(filename = paste0("../Images/Drone/IndistributionTest_",N_agent,".png"),height = 7,width = 11)



