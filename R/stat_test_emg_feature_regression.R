require(ggplot2)
source("local-functions.R")

doanova = FALSE

# define error functions
mse <- function(t,p){ mean((t-p)^2) }
rms <- function(t,p){ sqrt(mse(t, p)) }
se <-  function(x){sd(x)/sqrt(length(x))}

# define return function for apply
applyfun <- function(x){
    list(classifier = x$clf
        ,featureset = x$feature_set
        ,mse        = mse(x$y_true, x$y_pred)
        ,rms        = rms(x$y_true, x$y_pred)
        ,cor        = cor(x$y_true, x$y_pred)
        )
}

# read data
d = read.pickle("iPScnn/data/EMG/results/results_shallow_regression.bin")

# make dataframe from data  
df = data.frame(t(sapply(d$results, applyfun)))
df = data.frame(sapply(df, unlist))
df = df[!df$featureset == "all",]

# convert to numeric
df$mse = as.numeric(as.character(df$mse))
df$rms = as.numeric(as.character(df$rms))
df$cor = as.numeric(as.character(df$cor))

# define interaction factor (a unique string for each subgroup)
df$cgrp = interaction(df$classifier, df$features)

# calc anova if requested
if (doanova){
  df.aov = aov(rms ~ classifier * featureset, data = df)
  ph = TukeyHSD(df.aov)
}

# calc rms and correlation means and se for plotting
rms.mu = aggregate(df$rms, by = list(df$featureset, df$classifier), mean)
rms.se = aggregate(df$rms, by = list(df$featureset, df$classifier), se  )
cor.mu = aggregate(df$cor, by = list(df$featureset, df$classifier), mean)
cor.se = aggregate(df$cor, by = list(df$featureset, df$classifier), se  )

# bind together data for plotting and give sensible colnames
plotdata = cbind(rms.mu, rms.se$x, cor.mu$x, cor.se$x)
colnames(plotdata) <- c("features", "classifier", "RMS", "rmsse", "cor", "corse")



# define plots

pd <- position_dodge(0.1) 

p1 = ggplot(plotdata, aes(x=classifier, y=RMS, colour=features, group = features)) + 
     geom_line(position = pd) +  
     geom_errorbar(aes(ymin=RMS-rmsse, ymax=RMS+rmsse), width=.1, position = pd, color = "black") +
     geom_point( position = pd, cex = 4) 

p2 = ggplot(plotdata, aes(x=classifier, y=cor, colour=features, group = features)) + 
     geom_line(position = pd) +  
     geom_errorbar(aes(ymin=cor-corse, ymax=cor+corse), width=.1, position = pd, color = "black") +
     geom_point( position = pd, cex = 4) 

p3 = ggplot(df, aes(sample = rms)) + 
     stat_qq() +
     stat_qq_line() +
     facet_grid(.~cgrp)

