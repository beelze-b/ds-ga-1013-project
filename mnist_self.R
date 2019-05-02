library(lfda)
library(caret)

### make sure to run setup_mnist_data.py


train_data <- read.csv("data/mnist/mnist_train_data.csv", header = F)
test_data <- read.csv("data/mnist/mnist_test_data.csv", header = F)
train_labels <- read.csv("data/mnist/mnist_train_labels.csv", header = F)[[1]]
test_labels <- read.csv("data/mnist/mnist_test_labels.csv", header = F)[[1]]

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

run_self <- function(X, y, n_components, beta = 0.5) {
  model <- self(X, y, r = n_components, beta = beta, metric = 'orthonormalized')
  return(model)
}
self.model <- run_self(train_data, train_labels)
reduced_train <- mv$train %*% self.model$T
reduced_test <- mv$test %*% self.model$T