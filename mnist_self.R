library(lfda)
library(caret)

### make sure to run setup_mnist_data.py

setwd("~/Documents/NYU/FirstYear/MathematicalTools/hw/ds-ga-1013-project/")
train_data <- read.csv("data/mnist/mnist_train_data.csv", header = F)
test_data <- read.csv("data/mnist/mnist_test_data.csv", header = F)
train_labels <- read.csv("data/mnist/mnist_train_labels.csv", header = F)[[1]] + 1
test_labels <- read.csv("data/mnist/mnist_test_labels.csv", header = F)[[1]] + 1

columns_to_use <- !apply(train_data,2, function(x) all(x == 0))
train_data <- scale(train_data[, columns_to_use], scale = F)
test_data <- scale(test_data[, columns_to_use], scale = F)


PCA_RUN <- function(tr_data, te_data, DEBUG = FALSE) {
  sv1 <- svd(tr_data)
  index_hg <- sum(abs(sv1$d) >= 0.1)
  reconstruction_train <-sv1$u[, 1:index_hg] %*% diag(sv1$d[1:index_hg])
  
  sv2 <- svd(te_data)
  reconstruction_test <-sv2$u[, 1:index_hg] %*% diag(sv2$d[1:index_hg]) %*% t(sv2$v[, 1:index_hg]) %*% sv1$v[, 1:index_hg]
  
  return(list(train = reconstruction_train, test = reconstruction_test))
}

mv <- PCA_RUN(train_data, test_data)

run_self <- function(X, y, n_components, beta = 0.6) {
  start.time <- Sys.time()
  
  model <- self(X, y, r = n_components, beta = beta, metric = 'plain')
  
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  return(model)
}


## 75% of the sample size
smp_size <- floor(3200)

## set the seed to make your partition reproducible
set.seed(123)
self.indices <- sample(seq_len(nrow(train_data)), size = smp_size)



self.model <- run_self(mv$train[self.indices, ], train_labels[self.indices], n_components = 10, beta = 0.9)
reduced_train <- as.data.frame(mv$train %*% self.model$T)
reduced_test <- as.data.frame(mv$test %*% self.model$T)

trctrl <- trainControl(method = "repeatedcv", number = 5)

knn_fit <- train(x = reduced_train, y = as.factor(train_labels), method = "knn",
                 trControl=trctrl,
                 tuneLength = 5)


# model <- knn3(x = reduced_train, y = as.factor(train_labels), k = 7)


predictions <- predict(knn_fit, reduced_test)