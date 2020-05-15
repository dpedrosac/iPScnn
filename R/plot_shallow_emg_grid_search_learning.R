#fname = "emg_shallow_learning_all.csv"
#fname = "emg_shallow_learning_pigd.csv"
fname = "emg_shallow_learning_td.csv"
# fname = "rob_emg_shallow_learning_all.csv"

require(ggplot2)
stat = read.csv(fname)

# define plotfunction
plotfun <- function(varname)
{
  p = ggplot(data=stat, 
       aes(x=seconds, y=dv, 
           group=datalength,
           shape=datalength,
           color=datalength)) + 
               geom_line() + 
               geom_point() +
               scale_x_discrete("seconds") +
               scale_y_continuous(varname) + 
             facet_grid(.~method*featureset )     
return(p)
}

stat = data.frame(stat)
stat$dv = stat$cor
p1 = plotfun("correlation")
p1 + theme(panel.spacing.x=unit(c(1,1,1,5), "lines"))    
stat$dv = stat$rmse
p2 = plotfun("rmse")

jpeg(file=paste(fname,"a.jpeg",sep=""),1600,900)
print(p1)
dev.off()

