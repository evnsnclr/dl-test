# Flood Prediction Up to 24-Hour Horizon Using Deep Time Series Models

## Project Summary

Recurrent coastal flooding has become an increasingly critical issue due to sea level rise and intensified weather patterns. When water levels exceed road embankment heights, the affected infrastructure becomes unsafe for both vehicles and pedestrians. In response, transportation agencies like the Delaware Department of Transportation (DelDOT) have installed water level monitoring gauges at vulnerable roadway segments to support real-time closure decisions. 

Currently, road closures are reactive—triggered after the water level surpasses a predefined threshold. To improve proactivity, this project aims to develop predictive models capable of forecasting water levels up to 24 hours in advance. Such forecasts can empower agencies with additional lead time for early warnings, resource allocation, and proactive traffic management.

## Approach

This project explores the use of deep learning-based time series models for water level forecasting in the Delaware Bay. In particular, we will train and benchmark deep learning forecasting methods with varying complexity and architectural design:

### Models

- **Vanilla Chronos-Bolt Base**: A 205M-parameter, general-purpose time series foundation model developed by Amazon.
- **Fine-tuned Chronos-Bolt Base**: Adapted using historical Delaware Bay water level data for domain-specific performance.
- **LSTM (Long Short-Term Memory)**: A widely-used recurrent neural network (RNN) architecture for sequence modeling, known for capturing short- and medium-term temporal dependencies.
- **Transformer Encoder-Decoder**: An attention-based model architecture suitable for long-range sequence forecasting.

### Implementation

We will implement fine-tuning pipelines for Chronos-Bolt and custom training scripts for LSTM and Transformer models using PyTorch. Model performance will be evaluated based on metrics such as Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE). Additionally, we will compare zero-shot and fine-tuned performance of Chronos-Bolt to quantify the benefit of task-specific adaptation.

## Related Works

- **Delaware Bay Operational Forecast System (DBOFS)**: The current operational standard for water level forecasting in Delaware, developed by NOAA. DBOFS utilizes the Regional Ocean Modeling System (ROMS), a three-dimensional, physics-based hydrodynamic solver. While reliable, DBOFS is computationally intensive and limited by coarse update cycles and model complexity.

- **LSTM Networks**: A classical deep learning approach for time series forecasting. Recently, LSTM models have been applied to predict water levels in Florida (Forde et al., 2024), Europe (Vizi et al., 2023), and Korea (Cho et al., 2022).

- **Informer**: A Transformer-based model designed specifically for time series forecasting (Zhou et al., 2021). It has been shown to outperform traditional models such as LSTM and ARIMA, as well as other Transformer variants like Reformer and LogTrans, on benchmark datasets including ETT, ECL, and Weather.

- **Chronos-Bolt Foundation Model**: Recent advances in deep learning have produced time series foundation models like Chronos-Bolt. Based on the T5 encoder-decoder architecture and trained on over 100 billion time series observations (Ansari et al., 2024), Chronos-Bolt is capable of zero-shot forecasting—making predictions without additional fine-tuning. However, forecast accuracy can potentially be improved through fine-tuning and the incorporation of relevant covariates.

## Datasets

We will use historical water level data from DelDOT Hydro Monitoring Stations, accessible via their official website: [DelDOT Water Monitoring Data](https://deldot.gov/). The dataset includes:

- Water level measurements at five-minute intervals
- Data from over 50 locations across the Delaware Bay
- Records spanning more than five years
- High temporal resolution and geographic coverage suitable for training and evaluating deep learning models

## Team Members

- **Ziyi Ma** - zma75@gatech.edu
- **Xinghao Qi** - xqi48@gatech.edu  
- **Evan Smith** - esmith446@gatech.edu

## References

- Ansari, A. F., et al. (2024). Chronos: Learning the Language of Time Series.
- Cho, K., et al. (2022). Water level prediction using LSTM in Korea.
- Forde, S., et al. (2024). LSTM models for water level prediction in Florida.
- Vizi, Z., et al. (2023). Water level forecasting in Europe using neural networks.
- Zhou, H., et al. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting.