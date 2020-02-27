emg.shallow.regression <- function(data, lpar, folds = NULL, shallowmethods)
{
require(e1071)
require(rpart)
require(randomForest)

# define shallow learning function
  do.shl <- function(foldname, featurename, method) 
  {
    # reduce data to current featureset (all, rms, du ...) only
      fdat  = data[, c("updrs", lpar$shallow$emg.keptfeatures[[featurename]])]

    # reduce data to current fold subsets (train/test) only
      train = fdat[-folds[[foldname]],]
      test  = fdat[ folds[[foldname]],]

    # train model(s)
      if        (method == "svm-linear"){
        model = svm(updrs ~ ., data = train, kernel = "linear")
      } else if (method == "svm-radial"  ){
        model = svm(updrs ~ ., data = train, kernel = "radial")
      } else if (method == "svm-poly"  ){
        model = svm(updrs ~ ., data = train, kernel = "polynomial")
      } else if (method == "knn-regression"  ){
        model = knnreg(updrs ~ ., data = train, k = 5)
      } else if (method == "regression-tree"  ){
        model = rpart(updrs ~ ., data = train, method = "anova")
      } else if (method == "random-forest"  ){
        model = randomForest(updrs ~ ., data = train, ntree = 1000)
      }

    # make prediction
      pred  = predict(model, test)

    # calculate error and correlation between true values and prediction
    # and organise output
      out =    c( method     = as.character(method)
                , featureset = as.character(featurename)
                , rmse       = rmse(test$updrs, pred)
                , cor        = cor( test$updrs, pred))

  return(out)
  }

# create combinations of folds, features (all, hudgings, du) and
# method vectors such that each unique combination will be run
  ff = expand.grid( names(folds)
                  , lpar$shallow$emg.featurenames
                  , shallowmethods)

# apply learning to each combination of fold, featuresset and method
  out = mapply(do.shl, ff[,1], ff[,2], ff[,3])

# format output to data frame, convert numbers to numeric
  out = as.data.frame(t(out))
  out$rmse = as.numeric(as.character(out$rmse))
  out$cor  = as.numeric(as.character(out$cor ))
    
return(out)
}
