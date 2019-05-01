library(lfda)

### make sure to run setup_mnist_data.py


train_data <- read.csv("data/mnist/mnist_train_data.csv", header = F)
test_data <- read.csv("data/mnist/mnist_test_data.csv", header = F)
train_labels <- read.csv("data/mnist/mnist_train_labels.csv", header = F)[[1]]
test_labels <- read.csv("data/mnist/mnist_test_labels.csv", header = F)[[1]]



run_self <- function(X, y, n_components, beta = 0.5) {
  model <- self(X, y, r = n_components, beta = beta, metric = 'orthonormalized')
  
}