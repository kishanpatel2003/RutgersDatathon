### Kishan Patel
### Fall '23 Rutgers Datathon

packages_to_install <- c("forecast", "ggplot2", "gridExtra", "readr","kableExtra")
install.packages(packages_to_install)
library(forecast)
library(ggplot2)
library(gridExtra)
library(readr)
library(kableExtra)
#Import the data set
data_raw = read.csv('/Users/kishanpatel/Desktop/Datathon/Datathon_Fall2023_Dataset.csv', stringsAsFactors = F, header = T)
str(data_raw)
#Convert the data into a time series
data = ts(data_raw$Anomaly, start = 1850, frequency = 12)
head(data)
#Plot the data to visualize it. This is not necessary for the actual forecast but it's nice to see what you are working with
autoplot(data, col = 'indianred4') +
  theme_bw(base_size = 14) +
  ggtitle('Global Land and Ocean Temperature anomalies (1901-2000 mean as 0)') +
  xlab('') + ylab('Temperature (Celsius)')
#Create "train" and "test" data sets. Then create a forecast model on the train data
h = 36
data_train = window(data,end = c(2019, 08))
data_test = window(data, start = c(2019, 09))
data_stl12 = stlf(data_train, method = 'naive', h = h)
#Obtain the actual forecasted values
forecast_final = (data_stl12$mean)
forecast_final
#cross-validation
#define the desired STL model as a function (including all parameters), then send full data (NOT just training data), h, and window to it using tsCV()
#If you end up not using stl you will still have to do a version of this to get the CV error.
h = 36
window = 36
fstl = function(x, h){forecast(stlf(x, h = h))}
error = tsCV(data, fstl, h = h, window = window)
#FUNCTION cv_mape
#Returns cross-validation mean absolute percent error
#We are providing this function to you as it might be to difficult to code on your own but in return we require a
#1 slide explanation on your presentation of how it works so we can see that you actually understand it.
cv_mape = function(error, actual) {
  actual_table = data.frame(matrix(NA, nrow = length(actual), ncol = h))
  for(i in 1:(length(actual)-window)) {
    if ((i+window+h-1) <= length(actual)) {actual_table[i+window-1, ] = actual[(i+window):(i+window+h-1)]}
    else {actual_table[i+window-1, 1:(length(actual)-(i+window-1))] = actual[(i+window):(length(actual))]}
  }
  return(100*mean(abs(as.matrix(error) / as.matrix(actual_table)), na.rm = T))
}
#Performance table data frame
#'rmse'= Root Mean Square Error
#'mae'= Mean Absolute Error
#'mape'= Mean Absolute Percent Error
# We are providing you the code for the full error table too, but just like with cv-mape we require at least a 1 slide
# explanation of what this code does and how it works.
perf_stl = data.frame(rbind(
  cbind('rmse',
        formatC(round(accuracy(data_stl12)[ , 'RMSE'], 5), format = 'f', digits = 5),
        formatC(round(sqrt(mean((data_test - data_stl12$mean)^2)), 5), format = 'f', digits = 5),
        formatC(round(sqrt(mean(error^2, na.rm = T)), 5), format = 'f', digits = 5)),
  cbind('mae',
        formatC(round(accuracy(data_stl12)[ , 'MAE'], 5), format = 'f', digits = 5),
        formatC(round(mean(abs(data_test - data_stl12$mean)), 5), format = 'f', digits = 5),
        formatC(round(mean(abs(error), na.rm = T), 5), format = 'f', digits = 5)),
  cbind('mape',
        formatC(round(accuracy(data_stl12)[ , 'MAPE'], 5), format = 'f', digits = 5),
        formatC(round(mean(100*(abs(data_test - data_stl12$mean)) / data_test), 5), format = 'f', digits = 5),
        formatC(round(cv_mape(error, data), 5), format = 'f', digits = 5))),
  stringsAsFactors = F)

kable(perf_stl, caption = 'Performance - Temperature Anomalies horizon = 12, window = 36', align = 'r', col.names = c('', 'train', 'test', 'cv')) %>%
  kable_styling(full_width = F, position = 'l') %>%
  column_spec(2, width = '7em') %>%
  column_spec(3, width = '4.5em') %>%
  column_spec(4, width = '4.5em')

### END OF STARTER CODE

### Let's start with an ARIMA model.

best_arima <- auto.arima(data_train)
summary(best_arima)
arima_forecast <- forecast(best_arima, h=h)
plot(arima_forecast)
farima <- function(x, h) { forecast(auto.arima(x), h = h) }
# Auto.arima uses a stepwise search
error_arima = tsCV(data, farima, h=h, window=window)
}
# ^ this was too computationally intensive to run so I created a subset to workaround
subset_of_data <- data[seq(1, length(data), by = 5)]
best_model <- auto.arima(subset_of_data)
p <- best_model$p
d <- best_model$d
q <- best_model$q

farima_fixed <- function(x, h) {
  forecast(Arima(x, order=c(p,d,q)), h=h)
}

error_arima <- tsCV(data, farima_fixed, h=h, window=window)

# Let's look at our ARIMA model error

perf_arima = data.frame(rbind(
  cbind('rmse',
        formatC(round(accuracy(arima_forecast)[ , 'RMSE'], 5), format = 'f', digits = 5),
        formatC(round(sqrt(mean((data_test - arima_forecast$mean)^2)), 5), format = 'f', digits = 5),
        formatC(round(sqrt(mean(error_arima^2, na.rm = T)), 5), format = 'f', digits = 5)),
  cbind('mae',
        formatC(round(accuracy(arima_forecast)[ , 'MAE'], 5), format = 'f', digits = 5),
        formatC(round(mean(abs(data_test - arima_forecast$mean)), 5), format = 'f', digits = 5),
        formatC(round(mean(abs(error_arima), na.rm = T), 5), format = 'f', digits = 5)),
  cbind('mape',
        formatC(round(accuracy(arima_forecast)[ , 'MAPE'], 5), format = 'f', digits = 5),
        formatC(round(mean(100*(abs(data_test - arima_forecast$mean)) / data_test), 5), format = 'f', digits = 5),
        formatC(round(cv_mape(error_arima, data), 5), format = 'f', digits = 5))),
  stringsAsFactors = F)

kable(perf_arima, caption = 'Performance - Temperature Anomalies horizon = 12, window = 36', align = 'r', col.names = c('', 'train', 'test', 'cv')) %>%
  kable_styling(full_width = F, position = 'l') %>%
  column_spec(2, width = '7em') %>%
  column_spec(3, width = '4.5em') %>%
  column_spec(4, width = '4.5em')

### Now that we have completed ARIMA, let's try Facebook's Prophet model.
install.packages('prophet')
library(prophet)

df_prophet <- data.frame(ds = time(data), y = as.numeric(data))
m <- prophet(df_prophet, yearly.seasonality=FALSE, daily.seasonality=FALSE, weekly.seasonality=FALSE)
future <- make_future_dataframe(m, periods = 36, freq = "month")
forecast <- predict(m, future)

plot(m, forecast)
prophet_plot_components(m, forecast)
forecast_test <- dplyr::filter(forecast, ds %in% as.Date(time(data_test)))
predictions_prophet <- forecast_test$yhat
error_prophet <- as.numeric(data_test) - predictions_prophet

# Let's look at our Prophet model error.

perf_prof = data.frame(rbind(
  cbind('rmse',
        formatC(round(accuracy(fbp_forecast)[ , 'RMSE'], 5), format = 'f', digits = 5),
        formatC(round(sqrt(mean((data_test - fbp_forecast$mean)^2)), 5), format = 'f', digits = 5),
        formatC(round(sqrt(mean(error_fbp^2, na.rm = T)), 5), format = 'f', digits = 5)),
  cbind('mae',
        formatC(round(accuracy(fbp_forecast)[ , 'MAE'], 5), format = 'f', digits = 5),
        formatC(round(mean(abs(data_test - fbp_forecast$mean)), 5), format = 'f', digits = 5),
        formatC(round(mean(abs(error_fbp), na.rm = T), 5), format = 'f', digits = 5)),
  cbind('mape',
        formatC(round(accuracy(fbp_forecast)[ , 'MAPE'], 5), format = 'f', digits = 5),
        formatC(round(mean(100*(abs(data_test - fbp_forecast$mean)) / data_test), 5), format = 'f', digits = 5),
        formatC(round(cv_mape(error_fbp, data), 5), format = 'f', digits = 5))),
  stringsAsFactors = F)
kable(perf_prof, caption = 'Performance - Temperature Anomalies horizon = 12, window = 36', align = 'r', col.names = c('', 'train', 'test', 'cv')) %>%
  kable_styling(full_width = F, position = 'l') %>%
  column_spec(2, width = '7em') %>%
  column_spec(3, width = '4.5em') %>%
  column_spec(4, width = '4.5em')


### Now let's try a Long Short-Term Memory model (LSTM).
install.packages('keras')
library(keras)
install_keras()

max_val <- max(data)
min_val <- min(data)
scaled_data <- (data - min_val) / (max_val - min_val)

x <- as.matrix(scaled_data, ncol=1)
y <- as.matrix(scaled_data[-1], ncol=1)

x <- array_reshape(x, c(nrow(x), 1, ncol(x)))

model <- keras_model_sequential() %>%
  layer_lstm(units=50, input_shape=c(1,1)) %>%
  layer_dense(units=1)
model %>% compile(optimizer='adam', loss='mean_squared_error')

model %>% fit(x, y, epochs=50, batch_size=1, verbose=1)
scaled_data_test <- (data_test - min_val) / (max_val - min_val)
x_test <- as.matrix(scaled_data_test, ncol=1)
x_test <- array_reshape(x_test, c(nrow(x_test), 1, ncol(x_test)))
lstm_predictions <- model %>% predict(x_test) # Assuming x_test is the test dataset
lstm_predictions <- lstm_predictions * (max_val - min_val) + min_val
error_lstm <- data_test - lstm_predictions
rmse_lstm <- sqrt(mean(error_lstm^2))
mae_lstm <- mean(abs(error_lstm))
mape_lstm <- mean(abs((data_test - lstm_predictions) / data_test)) * 100

# Let's check out our results.

perf_lstm = data.frame(rbind(
  cbind('rmse',
        formatC(rmse_lstm, format = 'f', digits = 5),
        NA,  # This is assuming you don't have train error, similar to the ARIMA table
        formatC(rmse_lstm, format = 'f', digits = 5)),
  cbind('mae',
        formatC(mae_lstm, format = 'f', digits = 5),
        NA,
        formatC(mae_lstm, format = 'f', digits = 5)),
  cbind('mape',
        formatC(mape_lstm, format = 'f', digits = 5),
        NA,
        formatC(mape_lstm, format = 'f', digits = 5))),
  stringsAsFactors = F)

kable(perf_lstm, caption = 'Performance - LSTM Model', align = 'r', col.names = c('', 'train', 'test', 'cv')) %>%
  kable_styling(full_width = F, position = 'l') %>%
  column_spec(2, width = '7em') %>%
  column_spec(3, width = '4.5em') %>%
  column_spec(4, width = '4.5em')

### This concludes our analysis. The timeseries forecast that yielded the least error was the ARIMA Model.
### Now let's actually use our model to make predictions about the future
final_arima_model <- Arima(data, order=c(4,1,1))
forecast_results <- forecast(final_arima_model, h=96)
forecasted_anomalies <- forecast_results$mean
forecast_matrix <- matrix(forecasted_anomalies, nrow=12)
yearly_averages <- colMeans(forecast_matrix)
print(yearly_averages)
df_yearly <- data.frame(Year = 2023:2030, Average_Anomaly = yearly_averages)

# Plotting
library(ggplot2)
ggplot(df_yearly, aes(x = Year, y = Average_Anomaly)) +
  geom_line(color = "blue", group = 1) +
  geom_point(color = "red") +
  labs(title = "Yearly Average Anomalies: 2023-2030",
       y = "Average Anomaly",
       x = "Year") +
  theme_minimal()
