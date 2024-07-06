# load/install libraries
library(ggplot2)
library(reshape2)
install.packages("caret")
library(caret)

# load data set
df_original <- read.table("kc_house_data.csv",sep = ",",header = TRUE)

# change the year_built column into the age of the house
names(df_original)[names(df_original) == "yr_built"] <- "Age"
df_original$Age <- 2024 - df_original$Age

# select the first 2000 rows to explore

df_subset <- df_original[1:2000, ]


# calculate the correlation matrix

cor_matrix <- cor(df_subset[,sapply(df_subset, is.numeric)])

# visualization of the matrix

house_cor_matrix <- melt(cor_matrix)


# heat map for correlation between features
ggplot(house_cor_matrix, aes(Var1, Var2, fill=value)) +
  geom_tile() +
  scale_fill_gradient2(low="blue", high="red", mid="white", 
                       midpoint=0, limit=c(-1,1), space="Lab", 
                       name="Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1),
        axis.title.x = element_blank(),
        axis.title.y = element_blank()) +
  coord_fixed()
# This heatmap visualizes the correlation between different features of houses within the King County dataset. Warmer colors (red) indicate a positive correlation, while cooler colors (blue) show a negative correlation. Features that are highly correlated with the price, such as sqft_living and grade, are clearly of interest. Notably, sqft_living15 and sqft_above also show strong positive correlations with several features. On the other hand, id shows little to no correlation with other features, suggesting it might not be useful for predictive modeling. The presence of multicollinearity, evident between features like sqft_living and sqft_above, could impact the performance of some regression models

#histogram for the response variable price
#
ggplot(df_subset, aes(x = price)) + 
  geom_histogram(binwidth = 5000, fill = "blue", color = "green") + 
  theme_minimal() + 
  ggtitle("House Prices Distribution") + 
  xlab("Price") + 
  ylab("Frequency")
#The histogram shows a right-skewed distribution of house prices, with a majority of homes in the lower price range and few high-priced outliers. The skewness indicates more affordable homes and some luxury properties.

#boxplots for relation between the response variable price and categorical variables

# Number of bedrooms and Price

ggplot(df_subset, aes(x = factor(bedrooms), y = price)) + 
  geom_boxplot() + 
  theme_minimal() + 
  ggtitle("House Prices by Number of Bedrooms") + 
  xlab("Number of Bedrooms") + 
  ylab("Price")

# Grade and Price 

ggplot(df_subset, aes(x = factor(grade), y = price)) + 
  geom_boxplot() + 
  theme_minimal() + 
  ggtitle("House Prices by Level of Grade") + 
  xlab("Grade") + 
  ylab("Price")

# Floors and Price

ggplot(df_subset, aes(x = factor(floors), y = price)) + 
  geom_boxplot() + 
  theme_minimal() + 
  ggtitle("House Prices by Number of Floors") + 
  xlab("Number of Floors") + 
  ylab("Price")

# View and Price

ggplot(df_subset, aes(x = factor(view), y = price)) + 
  geom_boxplot() + 
  theme_minimal() + 
  ggtitle("House Prices by Quality of view") + 
  xlab("View Quality") + 
  ylab("Price")

# Condition and Price

ggplot(df_subset, aes(x = factor(floors), y = price)) + 
  geom_boxplot() + 
  theme_minimal() + 
  ggtitle("House Prices by House Condition") + 
  xlab("Level of House Condition") + 
  ylab("Price")

#The series of boxplots analyze the impact of various features on house prices in King County. The number of bedrooms shows a positive trend with price, with larger homes generally commanding higher prices. Grade, indicative of construction quality and design, reveals a strong positive relationship with price. Houses with more floors tend to have higher prices, although this is less pronounced. View quality also affects prices, with homes offering better views fetching higher prices. However, house condition seems to have a less clear impact on price. Together, these visualizations emphasize the significance of size, quality, and aesthetics on property values.

# Scatter Plots for Numerical variables and Price

# Square ft of living area and Price

ggplot(df_subset, aes(x = sqft_living, y = price)) + 
  geom_point(alpha = 0.4) + 
  geom_smooth(method = "lm", color = "red") +
  theme_minimal() + 
  ggtitle("Relationship between Sf of Living Area and Price") + 
  xlab("Square Footage of the Home") + 
  ylab("Price")

# House Age and Price

ggplot(df_subset, aes(x = Age, y = price)) + 
  geom_point(alpha = 0.4) + 
  geom_smooth(method = "lm", color = "red") +
  theme_minimal() + 
  ggtitle("Relationship between Age of the House and Price") + 
  xlab("Age") + 
  ylab("Price")

# Sf of basement and Price

ggplot(df_subset, aes(x = sqft_basement, y = price)) + 
  geom_point(alpha = 0.4) + 
  geom_smooth(method = "lm", color = "red") +
  theme_minimal() + 
  ggtitle("Relationship between Sf of Basement and Price") + 
  xlab("Square Footage of the Basement") + 
  ylab("Price")

#Base on the above graphs, the explanatory variables that shows correlation with the 
#response variables are Number fo bedrooms, Grade, View, Square Footage of Living Area and Square Footage of Basement

#These scatter plots reveal the relationships between house prices and various factors in King County. The living area square footage shows a strong positive correlation with price, as illustrated by the upward trend line. Conversely, the age of the house appears to have a negligible effect on price, with the trend line being relatively flat. Square footage of the basement also displays a positive correlation with price, although not as pronounced as living area square footage. These visuals highlight that larger living areas and basements tend to increase house prices, while the age of the house is not a decisive factor.


# model selection

# Define a function to apply power transformations to a numeric column
powr_transformations = function(col_name, powr = 2) {
  return (sprintf("I((%s - mean(%s))^%f)", col_name, col_name, powr))
}

# Define a function to create interaction terms between two columns
interaction = function(col_A, col_B) {
  return (sprintf("%s * %s", col_A, col_B))
}

# Define a function to create a formula for linear regression
create_formula = function(target, features) {
  return (sprintf("%s ~ %s", target, paste(features, collapse = " + ")))
}

# Define a function to get all possible features including original columns, power transformations, and interactions
get_all_possible_features = function(numeric_cols, factor_cols) {
  features_raw = c(numeric_cols, factor_cols)
  
  features_powr_transformations = list()
  for (i in 1:length(numeric_cols)) {
    features_powr_transformations = c(features_powr_transformations, powr_transformations(numeric_cols[i]))
  }
  
  features_interaction = list()
  for (i in 1:length(numeric_cols)) {
    for (j in (i + 1):length(features_raw)) {
      features_interaction = c(features_interaction, interaction(features_raw[i], features_raw[j]))
    }
  }
  return (c(features_raw, features_powr_transformations, features_interaction))
}

# Define a function to calculate Root Mean Squared Error (RMSE) using k-fold cross-validation
get_rmse = function(target, features, df, k_folds) {
  if (length(features) == 0) {
    features = c(1)
  }
  formula = create_formula(target, features)
  total_rmse = 0
  folds = createFolds(1:length(df[, 1]), k = k_folds)
  for (fold in 1:k_folds) {
    train_data = df[-folds[[fold]], ]
    test_data = df[folds[[fold]], ]
    
    model = lm(formula, data = df)
    
    predictions = predict(model, newdata = test_data)
    total_rmse = total_rmse + sqrt(mean((test_data[[target]] - predictions)^2))
  }
  return (total_rmse / k_folds)
}

create_df = function(all_possible_features, selected, selected_rmse, change) {
  len = length(all_possible_features) + 3
  result = data.frame(matrix(ncol = len, nrow = 0))
  result = rbind(result, unlist(list(sum(unlist(selected) == TRUE), selected_rmse, change, selected)))
  result[, 4:len] <- lapply(result[, 4:len], as.logical)
  result[, 1:2] <- lapply(result[, 1:2], as.numeric)
  colnames(result) <- c("index", "rmse", "change", all_possible_features)
  return (result)
}

# Define a helper function for features selection using forward or backward selection method
features_selection_helper = function(df, target, all_possible_features, selected, k_folds, method) {
  best_so_far = selected
  best_so_far_rmse = Inf
  change = ""
  cond = method == "backward"
  for (i in 1:length(all_possible_features)) {
    if (selected[i] == cond) {
      next_to_try = selected
      next_to_try[i] = !next_to_try[i]
      rmse = get_rmse(target, all_possible_features[next_to_try], df, k_folds)
      if (rmse < best_so_far_rmse) {
        best_so_far = next_to_try
        best_so_far_rmse = rmse
        change = all_possible_features[i]
      }
    }
  }
  return (list(create_df(all_possible_features, best_so_far, best_so_far_rmse, change), best_so_far))
}

# Define a function for features selection using forward or backward selection method
features_selection = function(df, target, all_possible_features, k_folds = 5, method = "forward") {
  if (!(method %in% list("forward", "backward"))) {
    return (-1)
  }
  result <- data.frame(matrix(ncol = length(all_possible_features) + 3, nrow = 0))
  init_value = method == "backward"
  current_selected = rep(init_value, length(all_possible_features))
  base = create_df(all_possible_features, current_selected, get_rmse(target, all_possible_features[current_selected], df, k_folds), "NA")
  result = rbind(result, base)
  for (i in 1:length(all_possible_features)) {
    res = features_selection_helper(df, target, all_possible_features, current_selected, k_folds, method)
    result = rbind(result, res[[1]])
    current_selected = res[[2]]
  }
  return (result)
}

# Storing column names, target variable, and column types for preprocessing
target = "price"
numeric_cols = c("bedrooms", "grade", "sqft_above", "sqft_basement")
factor_cols = c("view")
# Converting the 'view' column to a factor variable
df_original$view = as.factor(df_original$view)

# Get all possible features based on numeric and factor columns
all_possible_features = get_all_possible_features(numeric_cols, factor_cols)

model_full = lm(create_formula(target, all_possible_features), data=df_original[1:5000, ])
plot(model_full$fitted.values, model_full$residuals, main = "Residual plot",
     xlab = "Fitted values",
     ylab = "Residuals")

qqnorm(model_full$residuals)
qqline(model_full$residuals)

# Transforming the 'price' column by taking the natural logarithm
df_original$price = log(df_original$price)

# Splitting the dataset into training and testing sets
df_train = df_original[1:5000, ]
df_test = df_original[5000:10000, ]

# Selecting the training dataset for further processing
df = df_train

suppressWarnings({
  selections_forward = features_selection(df, target, all_possible_features, method="forward")
  selections_backward = features_selection(df, target, all_possible_features, method="backward")
})


selections_forward

selections_backward

plot(selections_forward$index, selections_forward$rmse, col = "blue", xlab = "index", ylab = "RMSE", main = "RMSE Plot")
points(selections_backward$index, selections_backward$rmse, col = "red")
legend("topright", legend = c("forward selection", "backward selection"), col = c("blue", "red"), pch = 1)

plot(selections_forward$index, selections_forward$rmse, col = "blue", xlab = "index", ylab = "RMSE", main = "RMSE Plot zoom in", ylim = c(0.344, 0.35))
points(selections_backward$index, selections_backward$rmse, col = "red")
legend("topright", legend = c("forward selection", "backward selection"), col = c("blue", "red"), pch = 1)

selection_table = selections_backward
features = all_possible_features[unlist(selection_table[selection_table$index==8, ][4:length(names(selection_table))])]
model = lm(create_formula(target, features), data=df)
summary(model)

plot(model$fitted.values, model$residuals, main = "Residual plot",
     xlab = "Fitted values",
     ylab = "Residuals")

qqnorm(model$residuals)
qqline(model$residuals)

predictions <- predict(model, newdata = df_test)
sqrt(mean((predictions - df_test$price)^2)) # RMSE on testing data

new_df = df[1,]
pred = predict(model, new_df, se.fit=TRUE, interval="prediction", level=0.95)
pred
print(sprintf("PI: [%f, %f]", exp(pred$fit[2]), exp(pred$fit[3])))

