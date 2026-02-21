library(SLIDE)
library("reticulate")
library("readxl")
library(dplyr)
library(tidyr)
library("randtoolbox")

start_time <- Sys.time()
np = import("numpy")
ori_data = np$load("MTR.npy")
keys = c('data_CEN', 'data_ADM', 'data_SHW', 'data_WAC',
         'data_CAB', 'data_TIH', 'data_FOH', 'data_NOP',
         'data_QUB', 'data_TAK', 'data_SWH', 'data_SKW',
         'data_HFC', 'data_CHW', 'data_SYP', 'data_HKU',
         'data_KET')
df = read_excel("station_indices.xlsx", col_names = FALSE)
datas = list()
indices = 1:17
for (index in indices) {
  sliced_data = ori_data[, index, ]
  data = sliced_data
  data_name = paste("data", index, sep = "_")
  datas[[data_name]] = data
}

names(datas) = keys

days = 1:194
train_days = days[1:116]
test_days = days[156:194]
train_datas = lapply(datas, function(matrix) matrix[train_days, ])
test_datas = lapply(datas, function(matrix) matrix[test_days, ])
X = do.call(cbind, train_datas)
Y = do.call(cbind, test_datas)
pvec = rep(70, 17)

standardizeX_centered <- function(X, pvec, center = TRUE) {
  d <- length(pvec)
  norms <- rep(0, d)
  svec <- rep(0, d)
  pcum <- c(0, cumsum(pvec))
  
  if (center) {
    Xmean <- colMeans(X)
    X <- X - matrix(Xmean, nrow(X), ncol(X), byrow = TRUE)
  } else {
    Xmean <- rep(0, ncol(X))
  }
  
  return(list(X = X, Xmean = Xmean))
}

# standarlize
train = standardizeX(X, pvec=pvec)
train_X = train$X
train$svec
test = standardizeX_centered(Y, pvec=pvec)
test_X = test$X

max(train$svec)
min(train$svec)

# SLIDE
slist = create_structure_list(train_X, pvec, n_lambda = 50, 
                              lambda_min = 0.01, 
                              lambda_max = max(train$svec), 
                              standardized = TRUE, eps = 1e-6, k_max = 1000)
out_bcv = slide_BCV(train_X, pvec = pvec, structure_list = slist$Slist, 
                    n_fold = 3, p_fold = 3, k_max = 1000, eps = 1e-6)
out_bcv$structure_min
out_slide = slide_givenS(train_X, pvec = pvec, S = out_bcv$structure_min, 
                         k_max = 10000, eps = 1e-6, standardized = T)
end_time <- Sys.time()
end_time - start_time

save_path_V = "slide_V.csv"
write.csv(out_slide$V, 
          save_path_V, 
          row.names = FALSE)
save_path_U = "slide_U.csv"
write.csv(out_slide$U, 
          save_path_U, 
          row.names = FALSE)
save_path_S = "slide_S.csv"
write.csv(out_bcv$structure_min, 
          save_path_S, 
          row.names = FALSE)


qr_result <- qr(out_slide$V)
V <- qr.Q(qr_result)
t(V) %*% V
V

train_X %*% V

error_matrix = test_X - test_X %*% V %*% t(V)

# MSE
total_mse = mean(error_matrix^2)
mse_values = numeric(17)
for (i in 1:17) {
  start_col = (i - 1) * 70 + 1
  end_col = i * 70
  mse_values[i] = mean(error_matrix[, start_col:end_col]^2)
}

for (i in 1:17) {
  print(mse_values[i])
}

# MAE
total_mae = mean(abs(error_matrix))
mae_values = numeric(17)
for (i in 1:17) {
  start_col = (i - 1) * 70 + 1
  end_col = i * 70
  mae_values[i] = mean(abs(error_matrix[, start_col:end_col]))
}
for (i in 1:17) {
  print(mae_values[i])
}
