fname = "emg_shallow_learning_all.csv"

require(ggplot2)
stat = read.csv(fname)
# reorder factor levels for plotting
stat = subset(stat, method == "ranger" & datalength == "same")

theme_set(
    theme_classic(base_size = 40)
)

# define plotfunction
plotfun <- function(varname)
{
  p = ggplot(data=stat,
       aes(x=seconds, y=dv, 
           group=featureset,
#           shape=featureset,
           color=featureset)) +
      geom_errorbar(aes(ymin = corll
                       ,ymax = corul
		       ,width = 0)
                                           ,position = position_dodge(.5))+
               geom_point(position = position_dodge(.5),size  = 6) +
               scale_x_continuous("sampling interval (s)", breaks = c(1:7)) +
               scale_y_continuous(varname, breaks = scales::pretty_breaks()) 
return(p)
}


stat$dv = stat$cor
p1 = plotfun("correlation")

jpeg(file=paste("cor_by_seconds_grouping_featureset.jpeg",sep=""),1600,1200)
print(p1)
dev.off()

