load("./results/shallow_emg_grid_search_learning.RData")
require(ggplot2)

# prepare plotting structure from output
for( i in c(1:length(out)))
{
    stat$method    [i] = out[[i]]$method    [[1]]
    stat$datalength[i] = out[[i]]$datalength[[1]] 
    stat$featureset[i] = out[[i]]$featureset[[1]]
    stat$seconds   [i] = out[[i]]$seconds   [[1]]
    stat$cor       [i] = out[[i]]$cor       [[1]]
    stat$rmse      [i] = out[[i]]$rmse      [[1]]    
}
  

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
p1 + example + theme(panel.spacing.x=unit(c(1,1,1,5), "lines"))    
stat$dv = stat$rmse
p2 = plotfun("rmse")
