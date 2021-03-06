# import libraries
library(tidyverse)
library(caret)
library(visdat)

# parameters
file.path <- "./data/"

# data import
training <- read_csv(paste0(file.path, "pml-training.csv"))
testing <- read_csv(paste0(file.path, "pml-testing.csv"))

# first exploratory analyses
table(training$classe)
training[sample(1:nrow(training), 100),] %>% vis_dat(sort_type = T) # we see that a lot of the feature data is missing
## calculate fill rates
na_perc <- function(col){sum(is.na(col))/nrow(col)}
fill.col <- 1 - sapply(1:ncol(training), function(col_idx){
  training[, col_idx] %>% na_perc
  })

# data preprocessing
training <- training %>% 
  mutate(classe = factor(classe, 
                         levels = c("A", "B", "C", "D", "E"))) # need factors for classification problem

# exclude cols w. < 50% fill rate
exclude.cols <- which(fill.col < 1)
na.cols <- which(fill.col >= .5 & fill.col < 1)
training_preproc <- training %>% select(-exclude.cols)

# impute missing values and create missing value indicators (mvi)
training_imputed <- training %>% select(c(1, na.cols)) %>% 
  mutate(magnet_dumbbell_z_imp = ifelse(is.na(magnet_dumbbell_z), 
                                        mean(magnet_dumbbell_z, na.rm = T), 
                                        magnet_dumbbell_z),
         magnet_forearm_y_imp = ifelse(is.na(magnet_forearm_y), 
                                       mean(magnet_forearm_y, na.rm = T), 
                                       magnet_forearm_y),
         magnet_forearm_z_imp = ifelse(is.na(magnet_forearm_z), 
                                       mean(magnet_forearm_z, na.rm = T), 
                                       magnet_forearm_z),
         magnet_dumbbell_z_mvi = ifelse(is.na(magnet_dumbbell_z), 1, 0),
         magnet_forearm_y_mvi = ifelse(is.na(magnet_forearm_y),  1, 0),
         magnet_forearm_z_mvi = ifelse(is.na(magnet_forearm_z),  1, 0)) %>%
  select(-c(magnet_dumbbell_z, magnet_forearm_y, magnet_forearm_z))

# create training data set
training_preproc <- merge(training_preproc, training_imputed, by = "X1") %>% select(-"X1")
training_preproc[sample(1:nrow(training_preproc), 100),] %>% vis_dat(sort_type = T)



# train first model: xgb
# create the caret experiment using the trainControl() function
ctrl <- trainControl(
  method = "cv", number = 5, # 5-fold CV
  selectionFunction = "best", # select the best performer
  savePredictions = TRUE
)

# xgb
if(file.exists("model_xgb.rds")){
  model.xgb <- readRDS("model_xgb.rds")
} else {
  xgb.grid <- expand.grid(nrounds = c(50, 100, 150), 
                          max_depth = c(6, 7, 8), eta = 0.3, 
                          subsample = 1, colsample_bytree = 1, 
                          gamma = 0, min_child_weight = 1)
  
  model.xgb <- train(classe ~ ., 
                     method = "xgbTree", 
                     data = training_preproc,
                     trControl = ctrl,
                     tuneGrid = xgb.grid)
  model.xgb
  saveRDS(model.xgb, file = "model_xgb.rds")
}

# random forest - ranger implementation for speed
if(file.exists("model_ranger.rds")){
  model.ranger <- readRDS("model_ranger.rds")
} else {
  ranger.grid <- expand.grid(mtry = 5:20, splitrule = "gini", 
                             min.node.size = c(1, 3, 5))
  model.ranger <- train(classe ~ .,
                        method = "ranger",
                        data = training_preproc,
                        trControl = ctrl, 
                        tuneGrid = ranger.grid)
  model.ranger
  saveRDS(model.ranger, file = "model_ranger.rds")
}

# make predictions with ranger
testing_preproc <- testing %>% select(-exclude.cols)
# impute missing values and create missing value indicators (mvi)
testing_imputed <- testing %>% select(c(1, na.cols)) %>% 
  mutate(magnet_dumbbell_z_imp = ifelse(is.na(magnet_dumbbell_z), 
                                        mean(training$magnet_dumbbell_z, na.rm = T), 
                                        magnet_dumbbell_z),
         magnet_forearm_y_imp = ifelse(is.na(magnet_forearm_y), 
                                       mean(training$magnet_forearm_y, na.rm = T), 
                                       magnet_forearm_y),
         magnet_forearm_z_imp = ifelse(is.na(magnet_forearm_z), 
                                       mean(training$magnet_forearm_z, na.rm = T), 
                                       magnet_forearm_z),
         magnet_dumbbell_z_mvi = ifelse(is.na(magnet_dumbbell_z), 1, 0),
         magnet_forearm_y_mvi = ifelse(is.na(magnet_forearm_y),  1, 0),
         magnet_forearm_z_mvi = ifelse(is.na(magnet_forearm_z),  1, 0)) %>%
  select(-c(magnet_dumbbell_z, magnet_forearm_y, magnet_forearm_z))

testing_preproc <- merge(testing_preproc, testing_imputed, by = "X1") %>% select(-"X1")
predictions <- predict(model.ranger, newdata = testing_preproc)
