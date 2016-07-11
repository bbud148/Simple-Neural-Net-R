library(nnet)

# read files in, no special parameters needed
train <- read.csv("train.csv")
test <- read.csv("test.csv")
train$label <- as.factor(train$label)

# split up training set for rapid testing
train1 <- train[1:30000,]
train2 <- train[30001:42000,]

# Neural Net Multinomial Prediction
multinom.model <- multinom(label ~. , train1, MaxNWts = 8000)
multinom.prediction <- predict(multinom.model, train2)  # 88% accuracy!

# Find a few numbers to see where the prediction went wrong
length(which(multinom.prediction == train2$label))  # choose a FALSE one
image <- matrix(as.numeric(train2[9766,-1]), nrow = 28, ncol =28, byrow =TRUE)  # convert to matrix
image(1:28, 1:28, image, col = gray(0:255/255))

# Now attempt on entire set
model <- multinom(label ~. , train, MaxNWts = 8000)
prediction <- predict(model, test)
submission <- data.frame(ImageId = 1:length(prediction), Label = prediction)
write.csv(submission, "nnetmulti.csv", row.names = FALSE)  # 0.87671, not bad for first try

# Taking a look at some of the odd ones
which(multinom.prediction == train$label)
image <- matrix(as.numeric(train[41936,-1]), nrow = 28, ncol =28, byrow = TRUE)  # convert to matrix
image(1:28, 1:28, image, col = gray(0:255/255))

# overall, the incorrect predictions from the model stems from very poor handwriting.
# This can be improved by training more images. 
