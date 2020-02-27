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

# (7) split in training and validation set
  val.ix = sample(nrow(coredata), round(nrow(coredata)*.1))
  traindata = coredata[-val.ix,]
  testdata  = coredata[ val.ix,]

  ctrl = trainControl(method = "repeatedcv", number = 10, repeats = 10)
                                        #
# see https://topepo.github.io/caret/available-models.html
# for a list of models
grid = list(
   lm      = NULL
#   ,rpart   = data.frame(.cp = seq(0, 1, .1))
#   ,treebag = NULL
#   ,bag     = NULL # produces error
#   ,brnn    = NULL # package not available
   ## ,bridge  = NULL
   ## ,blackboost = NULL
# ,kknn = NULL
   ## ,svmPoly = NULL
   ## ,svmRadial = NULL
   ##,M5 = NULL # requires JAVA
#    ,mlp = NULL #works
#    ,mlpML = NULL
#   ,neuralnet = NULL
   ,rf = NULL
)

sh.driver <- function(method)
{
  model = train(updrs~.,traindata, method = method, tuneGrid = grid[[method]], trControl=ctrl)
  pred = predict(model, testdata)
  out = list( method = method
                  , cor  =  cor(testdata$updrs, pred)
                  , rmse = rmse(testdata$updrs, pred)
                  , model = model)
   
  return(out)
}

out = lapply(names(grid), sh.driver)

## # (9) print results to console





