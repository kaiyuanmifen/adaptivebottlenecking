#organize data and plot 
library(ggplot2)

Data=read.csv("../../_DeepQResults.csv",header = F)

tail(Data)
#names(Data)=c("Episode","score","avg_score","epsilon")
names(Data)=c("data","Method","Episode","score","avg_score","epsilon","ExtraLoss","test_score","test_avg_score","test_ExtraLoss","testOOD_score","testOOD_avg_score","testOOD_ExtraLoss")

unique(Data$data)
data= "CartPole-v1"

VecPlot=Data[Data$data==data,]

ggplot(data = VecPlot, aes(x=Episode, y=score)) +
  geom_line()



ggplot(data = VecPlot, aes(x=Episode, y=score)) +
  geom_smooth(size=2)


ggplot(data = VecPlot, aes(x=Episode, y=avg_score)) +
  geom_line()


ggplot(data = VecPlot, aes(x=Episode, y=avg_score,color=Method)) +
  geom_smooth(size=2)+ggtitle(data)



ggplot(data = VecPlot, aes(x=Episode, y=test_avg_score,color=Method)) +
  geom_smooth(size=2)+ggtitle(data)



ggplot(data = VecPlot, aes(x=Episode, y=testOOD_avg_score,color=Method)) +
  geom_smooth(size=2)+ggtitle(data)


ggplot(data = VecPlot, aes(x=Episode, y=test_avg_score,color=Method)) +
  geom_smooth(size=2)



ggplot(data = VecPlot, aes(x=Episode, y=avg_score)) +
  geom_smooth(size=2)

ggplot(data = VecPlot, aes(x=Episode, y=testOOD_avg_score,color=Method)) +
  geom_smooth(size=2)




ggplot(data = VecPlot, aes(x=Episode, y=test_avg_score,color=Method)) +
  geom_smooth(size=2)


ggplot(data = VecPlot, aes(x=Episode, y=testOOD_avg_score,color=Method)) +
  geom_smooth(size=2)



ggplot(data = VecPlot, aes(x=Episode, y=trainReward)) +
  geom_smooth(size=2)+scale_color_brewer(palette="Paired")+xlim(0,200)+ggtitle(data)



VecPlot$Episode=as.factor(VecPlot$Episode)
ggplot(data = VecPlot, aes(x=Episode,y=trainReward)) +
  geom_boxplot()+ggtitle(data)





ggplot(data = VecPlot, aes(x=Episode, y=trainReward,color=hypothesis)) +
  geom_smooth(size=2,method="loess",span=0.3)+xlim(0,200)+ggtitle(data)



ggplot(data = VecPlot, aes(x=Episode, y=testReward,color=hypothesis)) +
  geom_point(size=2)+xlim(0,10)+ggtitle(data)


ggplot(data = VecPlot, aes(x=Episode, y=trainReward,color=hypothesis)) +
  geom_smooth(size=2)+xlim(0,100)+ggtitle(data)


ggplot(data = VecPlot, aes(x=Episode, y=OODtestReward,color=hypothesis)) +
  geom_smooth(size=2)+xlim(0,100)

ggplot(data = VecPlot[VecPlot$hypothesis==5,], aes(x=Episode, y=OODtestReward,color=hypothesis)) +
  geom_point(size=2)+xlim(0,100)


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




ggplot(data = VecPlot, aes(x=Episode, y=trainReward,linetype=hypothesis,colour=hypothesis)) +
  geom_smooth(size=2)+scale_color_brewer(palette="Paired")+ggtitle(data)


ggplot(data = VecPlot, aes(x=Episode, y=testReward,linetype=hypothesis,colour=hypothesis)) +
  geom_smooth(size=0.5,span = 0.1,method = 'loess')+scale_color_brewer(palette="Paired")+ggtitle(data)


ggplot(data = VecPlot[VecPlot$Episode<=100,], aes(x=Episode, y=trainReward,fill=hypothesis)) +
  geom_boxplot()+ggtitle(data)




VecPlot=Data[(!is.na(Data$Episode))&(Data$Episode>40)&(Data$N_agents==N_agent),]


ggplot(VecPlot, aes(x=Method, y=OODtestReward,color=Method)) + 
  geom_boxplot()+ coord_flip()+
    ggtitle(paste("Done env, N_agent=",N_agent))

ggsave(filename = paste0("../Images/Drone/OODTest_",N_agent,".png"),height = 7,width = 11)



ggplot(VecPlot, aes(x=Method, y=testReward,color=Method)) + 
  geom_boxplot()+ coord_flip()+
  ggtitle(paste("Done env, N_agent=",N_agent))

ggsave(filename = paste0("../Images/Drone/IndistributionTest_",N_agent,".png"),height = 7,width = 11)



