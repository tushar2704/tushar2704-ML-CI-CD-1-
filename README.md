# New York Taxi Spending Analysis

## Problem definition
Predict the average money spent on taxi rides for each region of New York per given day and hour.

This problem is a supervised regression problem. Supervised because we have the actual value of the value we’re trying to predict and regression because what we’re trying to predict is a continuous variable (as opposed to categorical).

## Data problems I fixed

![Negative and zero values graph](/images/Negative_and_zero_values_graph.png)

*Negative total_amount values:* removed them since they are likely faulty data points and there weren’t many of them
*Zero values for total_amount:* Same with negative values. Zero values are removed.


![Graph showing the too high values](/images/too_high_values_graph.png)


*Too-high values for total_amount:* some values for total_amount were too high, going as high as 600000. As these are unlikely values for a taxi fare, I decided to come up with an upper limit. The average taxi_fare was ~$16 and there were only 1166 data points higher than $200. Compared to the 7667792 data points, this is not a great loss of information. Thus, I decided to remove data points with a total_amount value higher than 200.

## Original features of the model
Here is the list of features that can be used for model development that came with the original data: [‘PULocationID’, ‘transaction_date’,’ transaction_month’,’ transaction_day’, ‘transaction_hour’, ‘trip_distance’,’ total_amount’, ‘count_of_transactions’]

You can refer to [this document](https://www1.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf) for the meaning of each feature.

## Feature engineering
I’ve added 3 sets of new features to the model. First set is time-based feature. These include, weekend and holiday boolean.

Second set is location-based information. We have Location IDs per region but there is a higher level abstraction for regions called Boroughs. This information came from the source of the main data.

The last set is weather related data. I’ve downloaded this data from this website. It’s free to use for development.

Here is a list of all features used in the final model:
['PULocationID', 'transaction_month', 'transaction_day', 'transaction_hour', 'transaction_week_day', 'weekend', 'is_holiday', 'Borough’, 'temperature', 'humidity', 'wind speed', 'cloud cover', 'amount of precipitation’, ‘total_amount’]

## The algorithms I tried and the results
I tried Decision Trees, Random Forest and Gradient Boosting. The benchmark model is a Decision Tree. In the benchmark model, I only included the original features of the model as stated above. And on the normal models I used all original features plus the newly created ones.

Here are the performance results before tuning:

| Algorithm         |  MAE  |  RMSE  |   R2  |
|-------------------|:-----:|:------:|:-----:|
| Benchmark model   | 9.778 | 14.739 | 0.225 |
| Decision tree     | 8.534 | 14.011 | 0.308 |
| Random forest     | 7.426 | 13.212 | 0.385 |
| Gradient boosting | 8.388 | 13.378 | 0.369 |

The Random Forest model is selected to be tuned. Here are the best parameter values:
n_estimators: 1800
min_samples_split: 2
min_samples_leaf: 4
max_features: sqrt
max_depth: 300
bootstrap: True

Thought because of the high n_estimator value, I’ve decided to go with the parameter values that give the 2nd best performance, which is not very different than the best performance:
n_estimators: 200
min_samples_split: 10
min_samples_leaf: 2
max_features: sqrt
max_depth: 150
bootstrap: True

The performance compares to previous models is:
| Algorithm         |  MAE  |  RMSE  |   R2  |
|-------------------|:-----:|:------:|:-----:|
| Benchmark model   | 9.778 | 14.739 | 0.225 |
| Decision tree     | 8.534 | 14.011 | 0.308 |
| Random forest     | 7.426 | 13.212 | 0.385 |
| Gradient boosting | 8.388 | 13.378 | 0.369 |
| Tuned Random Forest | 7.337 | 12.662 | 0.435 |

Here is the True vs. Predicted value plot for the tuned random forest model. X-axis is the true values and y-axis the predicted values.

![Performance graph of tuned Random Forest](/images/tuned_random_forest_graph.png)

## Next steps
As you can see from the plot above, the performance can be improved. Three ways that wasn’t tried in this notebook:
* limiting the regions included in this analysis. Removing the regions that do not normally get a lot of taxi traffic can be omitted. This might be a good action to take depending on the problem at hand. (If the goal is to service all of NYC no matter what, we should keep those data points)
* hand selecting borough that should be included in the model. Again this should be decided based on the problem at hand and how this model is going to be used. But if the goal is solely to increase model performance, only including boroughs with the most transactions can increase the performance since likely most mistakes come from boroughs with fewer data points. Though this assumption should be checked before taking this action.
