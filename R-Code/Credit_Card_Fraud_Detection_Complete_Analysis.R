install.packages(c("readr", "dplyr", "lubridate", "ggplot2",
               "caret", "xgboost", "pROC", "PRROC", "tidyr"))
install.packages("Ckmeans.1d.dp", repos = "https://cloud.r-project.org")
library(Ckmeans.1d.dp)
library(readr)
library(dplyr)
library(lubridate)
library(ggplot2)
library(caret)
library(xgboost)
library(pROC)
library(PRROC)
library(tidyr)

set.seed(42)

# 1. Create output folders
output_dir <- "fraud_detection_outputs"
visual_dir <- file.path(output_dir, "visualizations")
dir.create(visual_dir, recursive = TRUE, showWarnings = FALSE)

# 2. Load data
if (file.exists("fraudTest.csv")) {
  data <- read_csv("fraudTest.csv", show_col_types = FALSE)
} else {
  message("Select your fraudTest.csv file")
  data <- read_csv(file.choose(), show_col_types = FALSE)
}

required_cols <- c("trans_date_trans_time", "dob", "amt", "city_pop",
                   "lat", "long", "merch_lat", "merch_long",
                   "category", "is_fraud")
missing_cols <- setdiff(required_cols, names(data))
if (length(missing_cols) > 0) {
  stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
}

data <- data %>%
  mutate(
    trans_date_trans_time = ymd_hms(trans_date_trans_time),
    dob = ymd(dob),
    is_fraud = as.integer(is_fraud),
    hour = hour(trans_date_trans_time),
    dayofweek_num = wday(trans_date_trans_time, week_start = 1) - 1,
    month = month(trans_date_trans_time),
    age = as.numeric(difftime(trans_date_trans_time, dob, units = "days")) / 365.25
  )

# Haversine distance (km) between customer and merchant coordinates
rad <- pi / 180
lat1 <- data$lat * rad
lon1 <- data$long * rad
lat2 <- data$merch_lat * rad
lon2 <- data$merch_long * rad
a <- sin((lat2 - lat1) / 2)^2 +
  cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2)^2
data$distance_km <- 6371 * 2 * asin(sqrt(pmax(0, pmin(1, a))))
data$age <- pmax(18, pmin(100, data$age))

cat("Dataset dimensions:", nrow(data), "rows x", ncol(data), "columns\n")
cat("\nTarget distribution:\n")
print(table(data$is_fraud, useNA = "ifany"))
print(round(prop.table(table(data$is_fraud)) * 100, 3))

# 3. Statistical summaries
amount_summary <- data %>%
  group_by(is_fraud) %>%
  summarise(
    transactions = n(),
    mean_amount = mean(amt, na.rm = TRUE),
    median_amount = median(amt, na.rm = TRUE),
    sd_amount = sd(amt, na.rm = TRUE),
    .groups = "drop"
  )
write_csv(amount_summary, file.path(output_dir, "amount_statistical_summary.csv"))

category_summary <- data %>%
  group_by(category) %>%
  summarise(
    transactions = n(),
    fraud_count = sum(is_fraud == 1, na.rm = TRUE),
    fraud_rate = mean(is_fraud == 1, na.rm = TRUE) * 100,
    .groups = "drop"
  ) %>%
  arrange(desc(fraud_rate))
write_csv(category_summary, file.path(output_dir, "category_fraud_summary.csv"))

# Optional non-parametric comparison of transaction amounts
wilcox_result <- wilcox.test(amt ~ is_fraud, data = data)
writeLines(capture.output(wilcox_result),
           file.path(output_dir, "wilcoxon_test.txt"))

# Helper to save high-resolution plots
save_plot <- function(plot_object, filename, width = 8, height = 5) {
  ggsave(file.path(visual_dir, filename), plot_object,
         width = width, height = height, dpi = 300, bg = "white")
}

# 4. Visualization 1: Fraud vs legitimate class distribution
p1 <- ggplot(data, aes(x = factor(is_fraud,
                                  levels = c(0, 1),
                                  labels = c("Legitimate", "Fraud")))) +
  geom_bar(fill = "#78A6A3") +
  labs(title = "Fraud vs Legitimate Transactions",
       x = "Transaction class", y = "Number of transactions") +
  theme_minimal(base_size = 12)
save_plot(p1, "01_fraud_distribution.png")

# 5. Visualization 2: Transaction amount histogram
p2 <- ggplot(data, aes(x = amt, fill = factor(is_fraud))) +
  geom_histogram(bins = 60, position = "identity", alpha = 0.55) +
  coord_cartesian(xlim = c(0, 1500)) +
  scale_fill_manual(values = c("0" = "#83B7B2", "1" = "#D98C8C"),
                    labels = c("Legitimate", "Fraud")) +
  labs(title = "Transaction Amount Distribution",
       x = "Transaction amount (USD)", y = "Frequency", fill = "Class") +
  theme_minimal(base_size = 12)
save_plot(p2, "02_transaction_amount_distribution.png")

# 6. Visualization 3: Fraud rate by transaction category
p3 <- ggplot(category_summary,
             aes(x = reorder(category, fraud_rate), y = fraud_rate)) +
  geom_col(fill = "#8FAFC2") +
  coord_flip() +
  labs(title = "Fraud Rate by Transaction Category",
       x = "Transaction category", y = "Fraud rate (%)") +
  theme_minimal(base_size = 12)
save_plot(p3, "03_fraud_by_category.png")

# 7. Visualization 4: Fraud rate by hour
hour_summary <- data %>%
  group_by(hour) %>%
  summarise(fraud_rate = mean(is_fraud == 1, na.rm = TRUE) * 100,
            transactions = n(), .groups = "drop")
p4 <- ggplot(hour_summary, aes(x = hour, y = fraud_rate)) +
  geom_line(color = "#4C8582", linewidth = 1) +
  geom_point(color = "#4C8582", size = 2) +
  scale_x_continuous(breaks = 0:23) +
  labs(title = "Fraud Rate by Hour of Day",
       x = "Hour (0-23)", y = "Fraud rate (%)") +
  theme_minimal(base_size = 12)
save_plot(p4, "04_fraud_by_hour.png")

# 8. Visualization 5: Correlation heatmap
numeric_data <- data %>%
  select(amt, city_pop, hour, dayofweek_num, month,
         age, distance_km, is_fraud) %>%
  mutate(across(everything(), as.numeric))
cor_matrix <- cor(numeric_data, use = "pairwise.complete.obs")
cor_long <- as.data.frame(as.table(cor_matrix))
names(cor_long) <- c("Variable1", "Variable2", "Correlation")
p5 <- ggplot(cor_long, aes(x = Variable1, y = Variable2, fill = Correlation)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.2f", Correlation)), size = 3) +
  scale_fill_gradient2(low = "#C98282", mid = "white", high = "#78A6A3",
                       midpoint = 0, limits = c(-1, 1)) +
  labs(title = "Correlation Heatmap", x = NULL, y = NULL,
       fill = "Correlation") +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 35, hjust = 1))
save_plot(p5, "05_correlation_heatmap.png", 8, 6)

# 9. Visualization 6: Amount boxplot by class
box_data <- data %>%
  mutate(transaction_type = factor(is_fraud, levels = c(0, 1),
                                   labels = c("Legitimate", "Fraud")))
p6 <- ggplot(box_data, aes(x = transaction_type, y = amt,
                           fill = transaction_type)) +
  geom_boxplot(outlier.alpha = 0.15) +
  coord_cartesian(ylim = c(0, 1500)) +
  scale_fill_manual(values = c("Legitimate" = "#83B7B2",
                               "Fraud" = "#D98C8C")) +
  labs(title = "Transaction Amount by Class",
       x = "Transaction class", y = "Transaction amount (USD)") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")
save_plot(p6, "06_amount_boxplot.png")

# 10. Prepare model data
model_data <- data %>%
  select(is_fraud, amt, city_pop, hour, dayofweek_num,
         month, age, distance_km, category) %>%
  filter(complete.cases(.))

model_data$is_fraud <- as.integer(model_data$is_fraud)
model_data$category <- droplevels(as.factor(model_data$category))

# Stratified 70% train / 15% validation / 15% test split
train_id <- createDataPartition(model_data$is_fraud, p = 0.70,
                                list = FALSE)
train_df <- model_data[train_id, ]
remaining <- model_data[-train_id, ]
val_id <- createDataPartition(remaining$is_fraud, p = 0.50,
                              list = FALSE)
val_df <- remaining[val_id, ]
test_df <- remaining[-val_id, ]

# One-hot encode category. Align columns across all splits.
x_formula <- ~ amt + city_pop + hour + dayofweek_num +
  month + age + distance_km + category - 1
x_train <- model.matrix(x_formula, data = train_df)
x_val <- model.matrix(x_formula, data = val_df)
x_test <- model.matrix(x_formula, data = test_df)

align_columns <- function(mat, reference_names) {
  absent <- setdiff(reference_names, colnames(mat))
  if (length(absent) > 0) {
    zeros <- matrix(0, nrow = nrow(mat), ncol = length(absent),
                    dimnames = list(NULL, absent))
    mat <- cbind(mat, zeros)
  }
  mat[, reference_names, drop = FALSE]
}
x_val <- align_columns(x_val, colnames(x_train))
x_test <- align_columns(x_test, colnames(x_train))

y_train <- train_df$is_fraud
y_val <- val_df$is_fraud
y_test <- test_df$is_fraud

# 11. Train class-weighted XGBoost
positive_weight <- sum(y_train == 0) / sum(y_train == 1)
dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dval <- xgb.DMatrix(data = x_val, label = y_val)
dtest <- xgb.DMatrix(data = x_test, label = y_test)

params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  max_depth = 5,
  eta = 0.08,
  subsample = 0.85,
  colsample_bytree = 0.85,
  scale_pos_weight = positive_weight
)

model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 220,
  watchlist = list(train = dtrain, validation = dval),
  verbose = 0
)

# Select threshold on validation set by maximising F1
val_prob <- predict(model, dval)
thresholds <- seq(0.05, 0.99, by = 0.01)
f1_values <- sapply(thresholds, function(th) {
  pred <- ifelse(val_prob >= th, 1, 0)
  tp <- sum(pred == 1 & y_val == 1)
  fp <- sum(pred == 1 & y_val == 0)
  fn <- sum(pred == 0 & y_val == 1)
  precision <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
  recall <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
  ifelse(precision + recall == 0, 0,
         2 * precision * recall / (precision + recall))
})
best_threshold <- thresholds[which.max(f1_values)]
cat("Validation-selected threshold:", best_threshold, "\n")

# 12. Evaluate once on held-out test data
test_prob <- predict(model, dtest)
test_pred <- ifelse(test_prob >= best_threshold, 1, 0)

# caret confusionMatrix: rows are predictions, columns are reference labels
cm <- confusionMatrix(
  factor(test_pred, levels = c(0, 1)),
  factor(y_test, levels = c(0, 1)),
  positive = "1"
)
print(cm)

roc_obj <- roc(response = y_test, predictor = test_prob, quiet = TRUE)
roc_auc <- as.numeric(auc(roc_obj))

pr_obj <- pr.curve(
  scores.class0 = test_prob[y_test == 1],
  scores.class1 = test_prob[y_test == 0],
  curve = TRUE
)
pr_auc <- pr_obj$auc.integral

metrics <- data.frame(
  threshold = best_threshold,
  accuracy = unname(cm$overall["Accuracy"]),
  precision = unname(cm$byClass["Pos Pred Value"]),
  recall = unname(cm$byClass["Sensitivity"]),
  specificity = unname(cm$byClass["Specificity"]),
  F1 = unname(cm$byClass["F1"]),
  ROC_AUC = roc_auc,
  PR_AUC = pr_auc
)
print(metrics)
write_csv(metrics, file.path(output_dir, "model_metrics.csv"))
write_csv(as.data.frame(cm$table),
          file.path(output_dir, "confusion_matrix_values.csv"))

# 13. Visualization 7: Confusion matrix
cm_plot_data <- as.data.frame(cm$table)
names(cm_plot_data) <- c("Predicted", "Actual", "Count")
p7 <- ggplot(cm_plot_data,
             aes(x = Actual, y = Predicted, fill = Count)) +
  geom_tile(color = "white", linewidth = 1) +
  geom_text(aes(label = Count), size = 5) +
  scale_fill_gradient(low = "#E8F0EF", high = "#4C8582") +
  labs(title = "Confusion Matrix", x = "Actual class",
       y = "Predicted class") +
  theme_minimal(base_size = 12)
save_plot(p7, "07_confusion_matrix.png", 7, 5)

# 14. Visualization 8: ROC curve
roc_plot_data <- data.frame(
  false_positive_rate = 1 - roc_obj$specificities,
  true_positive_rate = roc_obj$sensitivities
)
p8 <- ggplot(roc_plot_data,
             aes(x = false_positive_rate, y = true_positive_rate)) +
  geom_line(color = "#4C8582", linewidth = 1.1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed",
              color = "grey50") +
  coord_equal() +
  labs(title = paste0("ROC Curve (AUC = ", round(roc_auc, 4), ")"),
       x = "False Positive Rate", y = "True Positive Rate") +
  theme_minimal(base_size = 12)
save_plot(p8, "08_roc_curve.png", 7, 5)

# 15. Visualization 9: Precision-Recall curve
pr_plot_data <- data.frame(
  recall = pr_obj$curve[, 1],
  precision = pr_obj$curve[, 2]
)
p9 <- ggplot(pr_plot_data, aes(x = recall, y = precision)) +
  geom_line(color = "#B87979", linewidth = 1.1) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(title = paste0("Precision-Recall Curve (AUC = ",
                      round(pr_auc, 4), ")"),
       x = "Recall", y = "Precision") +
  theme_minimal(base_size = 12)
save_plot(p9, "09_precision_recall_curve.png", 7, 5)

# 16. Feature importance (additional graph)
importance <- xgb.importance(feature_names = colnames(x_train),
                             model = model)
print(importance)
write_csv(as.data.frame(importance),
          file.path(output_dir, "feature_importance.csv"))

if (nrow(importance) > 0) {
  importance_plot <- xgb.ggplot.importance(
    importance[1:min(10, nrow(importance)), ]
  )
  save_plot(importance_plot, "10_feature_importance.png", 8, 5)
}

# 17. Save model, predictions and statistical outputs
xgb.save(model, file.path(output_dir, "credit_card_fraud_xgboost.model"))

predictions <- data.frame(
  actual = y_test,
  fraud_probability = test_prob,
  predicted_class = test_pred
)
write_csv(predictions, file.path(output_dir, "fraud_test_predictions.csv"))

write_csv(amount_summary,
          file.path(output_dir, "amount_statistical_summary.csv"))
write_csv(category_summary,
          file.path(output_dir, "category_fraud_summary.csv"))
writeLines(capture.output(wilcox_result),
           file.path(output_dir, "wilcoxon_test.txt"))

cat("\nAnalysis completed successfully.\n")
cat("All visualizations saved in: ",
    normalizePath(visual_dir), "\n", sep = "")
cat("Metrics and output files saved in: ",
    normalizePath(output_dir), "\n", sep = "")
