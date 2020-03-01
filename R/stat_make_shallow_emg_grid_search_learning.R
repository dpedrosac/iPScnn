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

    traindata.du      = traindata[, c("updrs", lpar$shallow$emg.keptfeatures$du)]
    testdata.du       = testdata [, c("updrs", lpar$shallow$emg.keptfeatures$du)]
    traindata.rms     = traindata[, c("updrs", lpar$shallow$emg.keptfeatures$rms)]
    testdata.rms      = testdata [, c("updrs", lpar$shallow$emg.keptfeatures$rms)]
    traindata.hudgins = traindata[, c("updrs", lpar$shallow$emg.keptfeatures$hudgins)]
    testdata.hudgins  = testdata [, c("updrs", lpar$shallow$emg.keptfeatures$hudgins)]

  ctrl = trainControl(method = "repeatedcv", number = 10, repeats = 10)

     ## treebag <- bag(bbbDescr, logBBB, B = 10,
     ##                bagControl = bagControl(fit = ctreeBag$fit,
     ##                                        predict = ctreeBag$pred,
     ##                                        aggregate = ctreeBag$aggregate))

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
 ,knn  = expand.grid(.k = seq(1,20,1))
   ## ,svmPoly = NULL
   ## ,svmRadial = NULL
   ##,M5 = NULL # requires JAVA
#    ,mlp = NULL #works
#    ,mlpML = NULL
#   ,neuralnet = NULL
#   ,rf = expand.grid(.mtry = seq(1,50,5))
)

sh.driver <- function(method, traindata, testdata)
{
  model = train(updrs~.,traindata, method = method, tuneGrid = grid[[method]], trControl=ctrl)
  pred = predict(model, testdata)
  out = list( method = method
                  , cor  =  cor(testdata$updrs, pred)
                  , rmse = rmse(testdata$updrs, pred)
                  , model = model)
   
  return(out)
}

out.all      = lapply(names(grid), sh.driver, traindata        , testdata   )
out.du       = lapply(names(grid), sh.driver, traindata.du     , testdata.du)
out.rms      = lapply(names(grid), sh.driver, traindata.rms    , testdata.rms)
out.hudgins  = lapply(names(grid), sh.driver, traindata.hudgins, testdata.hudgins)

## # (9) print results to console





