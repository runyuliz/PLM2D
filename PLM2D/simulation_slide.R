library(SLIDE)
library("reticulate")
library("readxl")
library(dplyr)
library(tidyr)
library(parallel)

start_time <- Sys.time()
keys = c('data_WAC', 'data_TIH', 'data_TAK', 'data_SKW', 'data_CHW')
seeds_list <- c(1501, 2586, 2653, 1055, 705, 106, 589, 2468, 2413, 1600,
                2464, 228, 915, 794, 3021, 3543, 1073, 3351, 1744, 1084,
                926, 3049, 1117, 642, 4767, 501, 4066, 333, 4684, 486,
                1962, 393, 4842, 4866, 1755, 2515, 3585, 4315, 4966, 2099,
                3599, 4121, 29, 65, 838, 3906, 3773, 4635, 3161, 2659,
                4615, 4628, 2451, 2846, 1144, 3078, 1103, 168, 1670, 2570,
                2377, 4395, 4257, 3862, 23, 2633, 3340, 2215, 3682, 4724,
                1907, 84, 227, 296, 1001, 2138, 711, 2801, 2527, 3752,
                3321, 3181, 4183, 1615, 3024, 1413, 763, 655, 4797, 3849,
                4153, 468, 157, 1295, 497, 4740, 2940, 3456, 373, 79)

# one seed
n = 50
seed = 1501
pvec = rep(70, 5)
datas = list()
for (key in keys){
  file_path <- sprintf("generated_data/n=%d/seed_%d_%s.csv", n, seed, key)
  datas[[key]] <- read.csv(file_path, stringsAsFactors = FALSE)
}
X = do.call(cbind, datas)
X_matrix = as.matrix(X)
train = standardizeX(X_matrix, pvec=pvec)

slist = create_structure_list(train$X, pvec, n_lambda = 50,
                              lambda_min = 0.01,
                              lambda_max = max(train$svec),
                              standardized = TRUE, eps = 1e-6, k_max = 1000)
out_bcv = slide_BCV(train$X, pvec = pvec, structure_list = slist$Slist,
                    n_fold = 3, p_fold = 3, k_max = 1000, eps = 1e-6)
out_slide = slide_givenS(train$X, pvec = pvec, S = out_bcv$structure_min,
                         k_max = 1000,
                         eps = 1e-6, standardized = T)

end_time <- Sys.time()
end_time - start_time

save_path_S = sprintf("slide/n=%d, seed=%d, S.csv", n, seed)
write.csv(out_bcv$structure_min,
           save_path_S,
           row.names = FALSE)
save_path_U = sprintf("slide/n=%d, seed=%d, U.csv", n, seed)
write.csv(out_slide$U, 
          save_path_U,
          row.names = FALSE)
save_path_V = sprintf("slide/n=%d, seed=%d, V.csv", n, seed)
write.csv(out_slide$V,
          save_path_V,
          row.names = FALSE)


# num_cores <- max(1, parallel::detectCores() - 4)
# cl <- makeCluster(num_cores)
# clusterExport(cl, varlist = c("seeds_list", "keys", "n", "pvec", 
#                               "standardizeX", "create_structure_list", 
#                               "slide_BCV", "slide_givenS"))
#
# parLapply(cl, seeds_list, function(seed) {
# datas = list()
# for (key in keys){
#   file_path <- sprintf("generated_data/n=%d/seed_%d_%s.csv", n, seed, key)
#   datas[[key]] <- read.csv(file_path, stringsAsFactors = FALSE)
# }
# X = do.call(cbind, datas)
# X_matrix = as.matrix(X)
# train = standardizeX(X_matrix, pvec=pvec)
# 
# slist = create_structure_list(train$X, pvec, n_lambda = 50, 
#                               lambda_min = 0.01, 
#                               lambda_max = max(train$svec), 
#                               standardized = TRUE, eps = 1e-6, k_max = 1000)
# out_bcv = slide_BCV(train$X, pvec = pvec, structure_list = slist$Slist, 
#                     n_fold = 3, p_fold = 3, k_max = 1000, eps = 1e-6)
# out_slide = slide_givenS(train$X, pvec = pvec, S = out_bcv$structure_min, 
#                          k_max = 10000,
#                          eps = 1e-6, standardized = T)
# save_path_S = sprintf("slide/n=%d, seed=%d, S.csv", n, seed)
# write.csv(out_bcv$structure_min, 
#           save_path_S, 
#           row.names = FALSE)
# # save_path_U = sprintf("slide/n=%d, seed=%d, U.csv", n, seed)
# # write.csv(out_slide$U, 
# #           save_path_U, 
# #           row.names = FALSE)
# save_path_V = sprintf("slide/n=%d, seed=%d, V.csv", n, seed)
# write.csv(out_slide$V, 
#           save_path_V, 
#           row.names = FALSE)
# })
# stopCluster(cl)

