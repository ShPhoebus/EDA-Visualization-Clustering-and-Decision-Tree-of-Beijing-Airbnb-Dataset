# EDA-Visualization-Clustering-and-Decision-Tree-of-Beijing-Airbnb-Dataset

-brief project description:
This paper analyzes the public Beijing Airbnb dataset, builds a decision tree model(including PCA dimensionality reduction processing) for predicting the price of listings and divides the apartments of listings into different groups via clustering. The exploratory data analysis performs a visualized summarization about the statics of listings and type of property, average prices and rating scores sorted by districts, occupancy rate in the future, word clouds of comments and demand changes overtime. Predicting the price of listing based on provided attributes via DecisionTreeRegressor model, and mean absolute deviation is the measure to evaluate the model. Meanwhile, PCA model is used for reducing dimensions to improve data quality and explained variance ratio is adopted as a measure of the quality of dimensionality reduction. When it comes to clustering, we use K-means model and find the optimal number of clusters based on the Silhouette Coefficient.
-research quality:
Any highly positive and negative thing about the project:
We learned and practiced EDA, clustering, and decision trees, and applied what we learned to a realistic topic. We achieved most of the expected goals, but some details and effects were not as perfect as expected.

Dataset quality and preprocessing:
The dataset we analyzed is from the website Inside Airbnb http://insideairbnb.com/index.html which provides with Airbnb information in cities round the world available to the public. 
The dataset consists of several tables and we mainly concentrated on four tables:
1.	Listings – Detailed listing data with 96 attributes
2.	Reviews – Rating scores with 6 attributes
3.	Calendar – The calendar data for reservation in the next year
We removed some samples whose values of some attribute is empty. And in order to make the results more accurate, we filled some null values with the average value of each attribute. Next, we also uniformed some values by removing the symbol (‘$’, ‘,’, ‘%’) and duplicate values.
methods:
EDA and visualization：
we selected 14 key attributes from 3 tables that are related to the subsequent work.
We used pyecharts and matplotlib to implement the visualization.

Decision Tree:
Here we use various attributes in the dataset to try to predict apartment prices(Which is the attribute 'price'). We use the DecisionTreeRegressor model of sklearn, set the random_state number to 30, and set the criterion to mse, maximum tree depth of 10.Then, we divide the data set, 30% is the test set, 70% is the train set. We calculate the Deviation of each data and calculate the total Mean Absolute Deviation, and we use this value as a measure of the effectiveness of the decision tree. 

Dimensionality reduction:
In order to improve that effect of the decision tree, we use the PCA method to reduce dimensions of attributes. We use the PCA model of sklearn, set svd_solver to 'full'. In order to get the best results, we choose to test the model from 1 to 20 when setting the number of dimensions (which is the parameter ‘n_components’ in sklearn). Here, we use the total value of ‘explained_variance_ratio_’ as a measure to evaluate how many dimensions the dimension reduction works best. 

Clustering:
Here we use clustering to try to divide the apartments in the data into different groups. We use the KMeans model of sklearn. In order to achieve the best clustering effect, we sequentially use from 2 to 8 as the parameter of the number of clusters for clustering, and use the Silhouette Coefficient as a measure of the clustering effect.
results:
Decision Tree:
We show each sample and the predicted value and the true value in a column chart(Using the sample serial number and house price as the x-axis and y-axis, respectively. And red line is the predict price, blue line is the real price), and save the output of the comparison result to csv file. We calculate the Deviation of each data and calculate the total Mean Absolute Deviation is 474.

Dimensionality reduction :
We have drawn graphics to show the results(Using the number of dimensions and the total value of ‘explained_variance_ratio_’ as the x-axis and y-axis, respectively).
Since this value is larger the better, and we found that when the dimension is reduced to almost 8 dimensions, this value no longer rises, so we choose 8 as the best dimension after dimension reduction. After the dimensionality reduction is completed, we model and train the decision tree on the dimensionality-reduced data. Here the decision tree training uses the same parameters as before, and also outputs the Mean Absolute Deviation result(which is 541), the Bar chart of the prediction result comparison, and the csv file of the comparison of the prediction result.

Clustering:
Here, we draw scatter plots of the effect of different clustering numbers (using regions and prices as the x-axis and y-axis, respectively. And ‘+’ sign of the same color represents the same cluster), and simultaneously plot the Silhouette Coefficient (using clustering numbers and Silhouette Coefficient value as the x-axis and y-axis respectively ).
As we all know, the larger the Silhouette Coefficient value, the better the clustering effect. According to the result of the diagram, we can see that the number of clusters is 5 when the Silhouette Coefficient value is maximum. At the end, we set the number of clusters to 5 to show the clustering effect at this time, which is the best clustering effect.
