
  <H1>
    Long-Term Hail Risk Assessment with Deep Neural Networks
  </H1>

## Abstract

Hail risk assessment is necessary to estimate and reduce damage to crops, orchards, and infrastructure. Also, it helps to estimate and reduce consequent losses for businesses and, particularly, insurance companies. But hail forecasting is challenging. Data used for designing models for this purpose are tree-dimensional geospatial time series. Hail is a very local event with respect to the resolution of available datasets. Also, hail events are rare - only 1% of targets in observations are marked as "hail".

Models for nowcasting and short-term hail forecasts are improving. Introducing machine learning models to the meteorology field is not new. There are also various climate models reflecting possible scenarios of climate change in the future. But there are no machine learning models for data-driven forecasting of changes in hail frequency for a given area.

The first possible approach for the latter task is to ignore spatial and temporal structure and develop a model capable of classifying a given vertical profile of meteorological variables as favorable to hail formation or not. Although such an approach certainly neglects important information, it is very light weighted and easily scalable because it treats observations as independent from each other. The more advanced approach is to design a neural network capable to process geospatial data. Our idea here is to combine convolutional layers responsible for the processing of spatial data with recurrent neural network blocks capable to work with temporal structure.

This study compares two approaches and introduces a model suitable for the task of forecasting changes in hail frequency for ongoing decades.
