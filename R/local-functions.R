slf <- function(){source("local-functions.R")}

local.parameters <- function()
{
lp = list()

lp$metadatafile  = "../../../data/patientenliste_onoff.xlsx"
lp$updrsdatafile = "updrsdata.csv"
#lp$updrsdatafile = "../../../data/updrsdata.csv"

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
#lp$shallow$emg.storagefile      = "../data/EMG/emgshallowfeaturedata.csv"
lp$shallow$emg.storagefile      = "emgshallowfeaturedata.csv"
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

# find workers (cores) on all nodes of a computing cluster
# i.e. detect number of cores on all nodes
find_workers <- function(nodes) {
  nodes <- unique(nodes)
  cl <- makeCluster(nodes)
  on.exit(stopCluster(cl))

  ns <- clusterCall(cl, fun = detectCores)
  rep(nodes, times = ns)
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
      dfm$updrs = details$Gesamt.OFF[  details$Pseudonym == dfm$subid[1] 
                                     & details$Bedingung == "OFF"]
    }else{
      dfm$updrs = details$Gesamt.ON [  details$Pseudonym == dfm$subid[1]
                                     & details$Bedingung == "ON"]
    }
return(dfm)
}


read.data <- function(verbose = FALSE)
{
    if(verbose) start_time <- Sys.time()
    require(doParallel)
    registerDoParallel()
    lpar = local.parameters()
    ddir = paste( lpar$shallow$emg.datadir
                , "/features_split"
                , lpar$shallow$emg.maxwindowlength
                , sep = "")
    fnames = dir(ddir, paste("*", lpar$shallow$emg.task, "*", sep="")
               , full.names = TRUE)

   # read in details file with updrs data, subids, inclusion info
    #old: details = read.csv(lpar$shallow$detailsfile, header = TRUE)
    details = read.csv(lpar$updrsdatafile)

   # remove not included subjects
    details = details[details$Einschluss == 1,]

  # filter files, remove those for which no detail info exists
    fnames = fnames[do.call(c, sapply(unique(details$Pseudonym), grep, fnames))]
    
  # read in pickle data, process and concatenate   
    dfout = do.call(rbind, mclapply(fnames, shallow.emg.get.pickle, lpar, details))
    write.csv(dfout, lpar$shallow$emg.storagefile)

  # output running time if requested  
    if(verbose) print( Sys.time() - start_time)
    
return(dfout)
}

calc.td_pigd <- function(d)
{
# calculate tremor dominance and postural instability gait disorder
# types according to Stebbins et al. 2013 (DOI: 10.1002/mds.25383)

tremor = d$MDS.UPDRS.2.10
       + d$Haltetremor.Rechts
       + d$Haltetremor.Links
       + d$Bewegungstremor.Rechts
       + d$Bewegungstremor.Links
       + d$Amplitude.ROE
       + d$Amplitude.LOE
       + d$Amplitude.RUE
       + d$Amplitude.LUE
       + d$Amplitude.Kiefer
       + d$Konstanz
    

pigd   = d$MDS.UPDRS.2.12
       + d$MDS.UPDRS.2.13
       + d$Gehen
       + d$Blockaden
       + d$Posturale.Stabilit
    

ratio = (tremor / 11) / (pigd / 5)

group = rep(NA, nrow(d))
group[ratio > .9 & ratio < 1.15          ] = "indeterminate"
group[ratio >= 1.15                      ] = "td"
group[ratio <= .9                        ] = "pigd"    
group[sign(tremor == 1) & pigd       == 0] = "td" 
group[     tremor == 0  & sign(pigd) == 1] = "pigd" 
group[     tremor == 0  & pigd       == 0] = "indeterminate"

out = list(tremor = tremor
          , pigd  = pigd
          , ratio = ratio
          , group = group
          , subid = d$Pseudonym)
return(out)    
}
