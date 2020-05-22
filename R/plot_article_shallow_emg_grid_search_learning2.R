fname = "emg_shallow_learning_all.csv"

require(ggplot2)
stat = read.csv(fname)
# reorder factor levels for plotting
stat$method = factor(stat$method, levels = c("lm", "svmPoly", "knn","ranger"))

theme_set(
    theme_classic(base_size = 40)
)

# define plotfunction
plotfun <- function(varname)
{
  p = ggplot(data=stat,
       aes(x=method, y=dv, 
           group=featureset,
#           shape=featureset,
           color=featureset)) +
      geom_errorbar(aes(ymin = corll
                       ,ymax = corul
		       ,width = 0)
                       ,position = position_dodge(.5)) +
               geom_point(position = position_dodge(.5), size  = 6) +
               scale_x_discrete("learning algorithm") +
               scale_y_continuous(varname, breaks = scales::pretty_breaks()) 
return(p)
}

stat = data.frame(stat)
stat = subset(stat, seconds == 7 & datalength == "same")
stat$dv = stat$cor
p1 = plotfun("correlation")

jpeg(file=paste("cor_by_method_grouping_featureset.jpeg",sep=""),1600,1600)
print(p1)
dev.off()

