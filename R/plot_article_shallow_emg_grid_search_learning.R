fname = "emg_shallow_learning_all.csv"

require(ggplot2)
stat = read.csv(fname)

# define plotfunction
plotfun <- function(varname)
{
  p = ggplot( data=stat
            , aes(x=seconds, y=dv))    + 
      geom_line(color = "blue")                      + 
      geom_point(color = "blue", size = 3)                     +
      geom_errorbar(aes(ymin = corll
                       ,ymax = corul
		       ,width = 0)) +
      scale_x_continuous("seconds", breaks = c(1:7))      +
      scale_y_continuous(varname)      + 
      facet_grid(.~method*featureset )     
return(p)
}

stat = data.frame(stat)
stat = subset(stat, datalength == "same")
stat$dv = stat$cor
p1 = plotfun("correlation")
p1 + theme(panel.spacing.x=unit(c(1,1,1,5), "lines"))    

#jpeg(file=paste(fname,"_paper.jpeg",sep=""),1600,900)
print(p1)
#dev.off()

