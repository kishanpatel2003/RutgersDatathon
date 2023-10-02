# Rutgers Datathon Fall 2023 – Global Temperature Anomalies Forecast
Time-series forecasting of climate variability for Rutgers Fall 2023 Datathon

[![R](https://img.shields.io/badge/R-Statistical%20Computing-blue)](https://www.r-project.org/)
[![ggplot2](https://img.shields.io/badge/ggplot2-Data%20Visualization-red)](https://ggplot2.tidyverse.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-purple)](https://keras.io/)

## Overview
In the Rutgers Fall 2023 Datathon, I undertook a project aimed at forecasting global temperature anomalies. By leveraging R's capabilities, I engaged in a comprehensive analysis and predictive modeling to understand and forecast temperature fluctuations.

## Technologies & Packages
- **Languages:** R
- **Packages:** `forecast`, `prophet`, `keras`, `ggplot2`, `gridExtra`

## Project Details:

### 1. Data Collection and Preprocessing
Acquired a dataset depicting global temperature anomalies starting from the year 1850. The data underwent various preprocessing tasks ensuring its readiness for modeling.

### 2. Exploratory Data Analysis (EDA)
Conducted EDA to understand the dataset's underlying patterns, seasonality, and trends. Visualizations aided in grasping the dataset's historical temperature shifts.

![Image Alt Text](/ArimaForecast.jpg)

### 3. Model Building & Selection:
Several predictive models were attempted:
- **ARIMA**: MAE ~ 0.069%
- **Meta's Prophet Model**: A contemporary tool developed by Meta (Facebook) adept for datasets having strong seasonal patterns.
- **LSTM Neural Network**: Leveraging the power of deep learning for sequence prediction, the data was normalized to enable efficient training.

### 4. Model Evaluation:
The ARIMA model excelled in predictive accuracy when juxtaposed with the other models. It achieved an MAE of approximately 0.069%.

### 5. Real-world Implications:

![Image Alt Text](/AnamolyForecast.jpg)

The forecasting results underscore a continued trajectory in temperature anomalies, signaling consistent climatic shifts. These findings not only reinforce current discussions surrounding global climate patterns but also punctuate the urgency required in addressing them.
