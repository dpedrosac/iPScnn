# (1) load R libraries
  require(caret)

# (2) load custom libraries
  source("local-functions.R")
  source("./emg-shallow/util_shallow_emg_regression_driver.R")

# (3) define/subset database 
  data.level     = "trial" # level of data abstraction (subject/trial)
  use.difference = FALSE   # use raw updrs or difference between of ON/OFF

# (4) load data, load and define local parameters/constants
  lpar = local.parameters()
  alldata       = read.csv(lpar$shallow$emg.storagefile, header=TRUE)
  alldata$updrs = as.numeric(alldata$updrs)
  shallowmethods = c( "svm-linear"
                    , "svm-poly"
                    , "svm-radial"
                    , "knn-regression"
                    , "regression-tree"
                    , "random-forest")

# (5) apply data subsetting as defined above (3)
  if(data.level == "subject"){alldata = aggregate(.~subid+cond, alldata, mean)}
  if(data.level == "subject" & use.difference == TRUE)
  {
      ONdata   = subset(alldata[order(alldata$subid),], cond == "ON" )    
      OFFdata  = subset(alldata[order(alldata$subid),], cond == "OFF")
      coredata = OFFdata - ONdata
  }

# (6) remove non-predictor and non-response variables from dataset
  coredata = subset(alldata,select = -c(X, trial, cond, subid))

# (7) create folds for cross validation
  folds = createFolds(coredata$updrs, k = lpar$shallow$emg.nfolds)

# (8) fun shallow learning function which is defined in 
#     emg-shallow/util_shallow_emg_regression_driver.R
  emg.shl.res = emg.shallow.regression(coredata, lpar, folds, shallowmethods)

# (9) print results to console
  print(aggregate(.~method+featureset, emg.shl.res, mean))




