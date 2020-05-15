# localized version of shallow emg learning which is intended to run
# remotely. Therefore all necessary files need to be in the current
# directory

# prerequisites:
# 1. download patientenlist_onoff.xlsx, save UPDRS sheet as updrsdata.csv
#    to the working directory. Beforehand:
#    - Name ON/OFF column "Bedingung"
#    - complete Pseudonym column with subids (which are missing in every 2nd)
#    - manually delete double rows (1 data of subject is doubled)
#    - replace "x" with NA
# 2. preprocess data using the read.data() function from local-functions.R
#    which reads in pickle data files containing the EMG features and writes
#    them to a .csv file 

# get starttime for performance check
  start_time = Sys.time()
  
# local constants
  dryrun    = FALSE # calculate no model, just output sample statistics (debug)
  debug     = FALSE  # stop after model generation for inspection
  hostnames = c("horst","udo") # hosts for cluster computing 
                               # all hosts need "urs" as a user

# load R libraries
  require(caret)
  require(doParallel)

# load custom libraries and local parameters/constants
  source("local-functions.R")
  lpar = local.parameters()

# init parallel processing
  workers <- find_workers(hostnames)
  cl <- makeCluster(workers)
  registerDoParallel(cl) # old method for one machine

# define subgroups (tremor dominant / postural instability/gait diffculty
# updrsdata.csv is a sheet saved manually from patientenliste_onoff.xlsx
# redefine new group pigd+intermediate
  updrsdata = read.csv(lpar$updrsdatafile)
  subgroups = calc.td_pigd(updrsdata[updrsdata$Bedingung=="OFF",])
  subgroups$cgroup = subgroups$group
  subgroups$cgroup[ subgroups$cgroup == "pigd"
                  | subgroups$cgroup == "indeterminate"] = "pigd"

# load data
  sampledata       = read.csv(lpar$shallow$emg.storagefile, header=TRUE)
  sampledata$updrs = as.numeric(sampledata$updrs)

for(subgroup in c("td", "pigd"))
#for(subgroup in "all")
{
  outfileroot = "emg_shallow_learning"
  if(subgroup == "all")
  {
    alldata = sampledata
    outfile = paste(outfileroot,"all",sep="_")
  } else {    
    alldata = sampledata[sampledata$subid %in% subgroups$subid[subgroups$cgroup == subgroup],]
    outfile = paste(outfileroot,subgroup,sep="_")
  }

cat(paste("Starting analysis for subgroup",subgroup))
cat(". Output is diverted to file\n\n")  
  
  
if(!debug) sink(file = paste(outfile,".txt",sep=""), split = TRUE)

cat(paste("Processing ", length(unique(alldata$subid)), "subjects"))
cat("\n\n")
     
# generate random vectors for splitting in training and test set
# for all the possible time window values.
# The first part is a elaborate way for getting some constants to make this
# generalizable to other time windows than 8s (which is probably unnecessary)
 # first we calculate the according constants
   ntimepoints         = length(unique(alldata$time))
   nsamplesperinterval = diff(unique(alldata$time))[1]
   ntimepointspersec   = floor(lpar$shallow$emg.frequency / nsamplesperinterval)
   nusabletimepoints   = floor(ntimepoints / ntimepointspersec)
   alldata$timeranking = as.integer(as.factor(alldata$time))
 # then we remove the data not evenly divisible to a second
   alldata = alldata[alldata$timeranking <= (ntimepointspersec * nusabletimepoints),]
 # and aggregate for every second
   alldata$secs = ceiling(alldata$timeranking / ntimepointspersec)
   maxsecs = max(alldata$secs)
   alldata = aggregate(.~ secs + subid + cond + trial, data = alldata, mean)

 # generate data sets aggregated over 1,2 ... all seconds for all subgroups
   aggfun <- function(sec)
   {
     nuseabletimepoints = floor(max(alldata$secs)/sec)*sec
     datatmp = alldata[alldata$secs <= nuseabletimepoints,]
     datatmp$agginterval = ceiling(datatmp$secs / sec)
     outdata = aggregate(.~ agginterval + subid + cond + trial, data = datatmp, mean)
     return(outdata)
   }
   aggdata = lapply(unique(alldata$secs), aggfun)


 # generate the random sequences corresponding to all data sets and
 # to the "same" and "max" methods
   max.rand.fun   <- function(x)
                     {
                       v = sample(nrow(x), round(nrow(x)*.1))
                       out = list(test  = v
                                 ,train = c(1:nrow(x))[-v])
                       return(out)
                     }

   same.rand.fun  <- function(x)
                     {
                       ntrials = sum(x$secs == unique(x$secs)[1])
                       nint    = length(unique(x$agginterval))
                       m = matrix(0, nrow = ntrials, ncol = nint)
                       ix = sample(c(1:nint), ntrials, replace = TRUE)
                       m[cbind(c(1:ntrials), ix)] = 1
                       selectint = which(t(m) == 1)
                       v = sample.int(ntrials, round(ntrials*.1))
                       out = list(test  = selectint[ v]
                                 ,train = selectint[-v])
                       return(out)
                     }
                                 
  ix.list = list(max  = lapply(aggdata,  max.rand.fun)
                ,same = lapply(aggdata, same.rand.fun))

# (8) generate the control structure for grid search learning
  ctrl = trainControl( method  = "repeatedcv"
                     , number  = lpar$shallow$emg.nfolds
                     , repeats = 10)

# define shallow learning methods used and the parameters for
# grid search
# see https://topepo.github.io/caret/available-models.html
# for a list of models
grid = list(
   lm      = NULL
   #  ,rpart   = data.frame(.cp = seq(0, 1, .1))
   #  ,treebag = NULL
   #  ,bag     = NULL # produces error
   #  ,brnn    = NULL # package not available
   ## ,bridge  = NULL
   ## ,blackboost = NULL
   #  ,knn  = expand.grid(.k = seq(1,20,1))
  ,knn = NULL
  ,svmPoly = NULL
   ## ,svmRadial = NULL
   ## ,M5 = NULL # requires JAVA
   #  ,mlp = NULL #works
   #  ,mlpML = NULL
   #  ,neuralnet = NULL
   #,rf = NULL ## rf stopped working magically, use ranger instead
   #,rf = expand.grid(.mtry = seq(1,50,5))
  ,ranger = NULL
)

# define shallow learning driver function
sh.driver <- function(factors)
{
 # retrieve factors from input vector
  datalength = factors[1]
  featureset = factors[2]
  method     = factors[3]
  sec        = as.numeric(factors[4])
  cnt        = factors[5]

 # get data, remove non-predictor and non-response variables as well as
 # unused features from dataset
  coredata = aggdata[[sec]]
  coredata = subset(coredata
                   ,select = c("updrs"
                              ,unlist(lpar$shallow$emg.keptfeatures[featureset])))

  cat("\n\n")  
  cat(paste("Now processing model",cnt,"of",nmodels))
  cat(paste(" using", datalength, featureset, method, sec,"\n"))
    
 # retrieve training and test data from all data (filter by datalength (same/max)
 # and by sec in this step as well)
  traindata = coredata[ix.list[[datalength]][[sec]]$train,]
  testdata  = coredata[ix.list[[datalength]][[sec]]$test ,]

 # prepare output list
  out = list( method     = method
            , datalength = datalength
            , featureset = featureset
            , seconds    = sec
            , ntraindata = nrow(traindata)
            , ntestdata  = nrow(testdata)
            )
    
 if(!dryrun) # if in dryrun we don't train the model but only output sample 
 {           # statistics for debugging
   # train model
    model = train( updrs ~ .
                 , traindata
                 , method    = method
                 , tuneGrid  = grid[[method]]
                 , trControl = ctrl)
    print(model)

   # make predictions, calculate and organise output in list
    pred = predict(model, testdata)
    correl  =  cor.test(testdata$updrs, pred)

   # append model output and predictions to output list
    #out$model = model  # saving this makes the output massive (>1GB)
    out$cor   = correl$estimate
    out$corll = correl$conf.int[1]
    out$corul = correl$conf.int[2]
    out$rmse  = rmse(testdata$updrs, pred)
  
    if(debug) browser()
  } # end of dryrun conditional

  return(out)
}

# create matrix of all combinations of length, featureset and methods
  datalength  = c("max", "same")
  featuresets = c("all", "du", "rms", "hudgins")
  methods     = names(grid)
  # secs        = as.character(c(1:maxsecs))
  secs = as.character(c(1:7))
  allfactors  = as.matrix(expand.grid(datalength, featuresets, methods, secs))
  allfactors  = cbind(allfactors, as.character(c(1:nrow(allfactors))))
  nmodels     = nrow(allfactors)

# apply shallow learning driver function across combination matrix rows  
  out = apply(allfactors, 1, sh.driver)

# do a loop instead if apply hangs up...
# out = list()
# for( i in c(1:nrow(allfactors)))
# {
#     out[[i]] = sh.driver(allfactors[i,])
# }

# stop sinking model summaries to txt file, save results
  if(!debug) sink(file = NULL)

# convert list output to data frame, and write data to file
  if(!debug) stat = do.call(rbind.data.frame, out) 
  if(!debug) write.csv(stat, paste(outfile,".csv",sep=""))

} # end of subgroup loop

# get end time, print message
  end_time = Sys.time()
  print(end_time - start_time)
