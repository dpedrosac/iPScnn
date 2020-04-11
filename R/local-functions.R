slf <- function(){source("local-functions.R")}

local.parameters <- function()
{
lp = list()

# parameters for shallow learning algorithms
lp$shallow = list()
lp$shallow$tvsplit              = .1 #training/validation split
lp$shallow$detailsfile          = "../data/EMG/detailsEMG.csv"
lp$shallow$emg.maxwindowlength  = 8
lp$shallow$emg.frequency        = 200
lp$shallow$emg.channels         = c(4, 5, 8)
lp$shallow$emg.task             = "tap"
lp$shallow$emg.nfolds           = 10
lp$shallow$emg.datadir          = "../data/EMG"
lp$shallow$emg.storagefile      = "../data/EMG/emgshallowfeaturedata.csv" 
lp$shallow$emg.featurenames     = c("all", "rms", "hudgins", "du")
lp$shallow$emg.features.all     = c( "IAV" , "MAV", "RMS"
                                   , "WAMP", "WL" , "ZC"
                                   , "SSC" , "VAR")
lp$shallow$emg.features.rms     = c("RMS")
lp$shallow$emg.features.hudgins = c("MAV", "WL", "ZC", "SSC")
lp$shallow$emg.features.du      = c("IAV", "VAR", "WL"
                                       , "ZC", "SSC", "WAMP")
lp$shallow$emg.keptfeatures = list(
  all = 
  as.vector(outer( lp$shallow$emg.features.all
                 , lp$shallow$emg.channels
                 , paste, sep="_"))
 ,rms = 
  as.vector(outer( lp$shallow$emg.features.rms
                 , lp$shallow$emg.channels
                 , paste, sep="_"))
 ,hudgins = 
  as.vector(outer( lp$shallow$emg.features.hudgins
                 , lp$shallow$emg.channels
                 , paste, sep="_"))
 ,du = 
  as.vector(outer( lp$shallow$emg.features.du
                 , lp$shallow$emg.channels
                 , paste, sep="_"))
 )

return(lp)    
}

# define error functions
rmse <- function(true, predicted)
{
  sqrt(mean((true-predicted)^2))
}

# read pickle file
read.pickle <- function(fname){
    require(reticulate)
    use_python("/usr/bin/python3")
    pd <- import("pandas")
    pickle_data <- pd$read_pickle(fname)
return(pickle_data)
}


shallow.emg.get.pickle <- function(fname, lpar, details)
# read in emg feature data from pickle file, reduce to required
# features, calculate mean feature values, append subject, trial
# and updrs information
{
    dfm  = read.pickle(fname)        
    #dfm = data.frame(t(base::colMeans(dfm)))
    dfm = dfm[,lpar$shallow$emg.keptfeatures$all]
    pathstrings  = strsplit(fname,"/")
    corefilename = pathstrings[[1]][length(pathstrings[[1]])] 
    filestrings  = strsplit(corefilename,"_")
    dfm$time     = as.numeric(rownames(dfm))
    dfm$subid    = substr(corefilename, 1, 11)
    dfm$cond     = filestrings[[1]][5]
    dfm$trial    = as.numeric(gsub("trial", "", filestrings[[1]][7]))
    if( all(dfm$cond == "OFF") ){
      dfm$updrs = details$updrsOFF[ details$Name == dfm$subid[1]]
    }else{
      dfm$updrs = details$updrsON[  details$Name == dfm$subid[1]]
    }
return(dfm)
}


read.data <- function()
{
    lpar = local.parameters()
    ddir = paste( lpar$shallow$emg.datadir
                , "/features_split"
                , lpar$shallow$emg.maxwindowlength
                , sep = "")
    fnames = dir(ddir, paste("*", lpar$shallow$emg.task, "*", sep="")
               , full.names = TRUE)

    details = read.csv(lpar$shallow$detailsfile, header = TRUE)
    dfout = do.call(rbind, lapply(fnames, shallow.emg.get.pickle, lpar, details))
    write.csv(dfout, lpar$shallow$emg.storagefile)
return(dfout)
}
