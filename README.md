Comparison of House Price 
Prediction Models

  
 
-	Introduction and Background


The goal of this project is to build a model that predicts house prices. We wanted to compare 4 regression models to see which one of them is  better at predicting the price for houses, to do that first I had to explore the data and try to get as much information as possible in order to have a good understanding of what we can achieve with this data.
I used correlation heatmap to find the features that affect the price in order to use those features to build the prediction model and also compare it with the model that uses all features to see if there will be a noticeable difference.




-	Data & Experiment
We will be working with USA real estate data. This data is available for everyone on Kaggle, It has 900 thousand entries and 12 attributes.
•	street - Street name
•	city - City name
•	state - State name
•	zip_code - Zip Code
•	full_address - Full address
•	status - Housing Status (on sale or other option)
•	price - Price in USD
•	bed - Bedroom count
•	bath - Bathroom count
•	acre_lot - Acre lot
•	house_size - House size in sqft (square feet)
•	sold_date - The date when the house is sold

 
The end goal of this project is to build different machine learning models, evaluate them and compare them to see which one of them will better predict house prices.
The first part of the project will be cleaning the data and processing it, like finding outliers and removing them, removing duplicates, and replacing null values or removing them.
 

The second part of the project will focus on statistical analysis on the data and trying to answer some questions to further obtain information and increase our knowledge with the data, familiarize ourselves with it. 

•	What attributes affect house prices?


Based on the below correlation heatmap we can say that the features that affect house prices are:

	
1)	Number of Bathrooms


2)	Number of Bedrooms


3)	Square Feet Size
 



-	Models to predict house prices

First, I built a linear regression model using scikit-learn, I used the model on the correlated features and that gave me an accuracy of 0.99 and when I used it on all of the features, I got an accuracy of 0.95.
This means that the linear regression model works best on correlated features.

Second, I used Random Forest Regression model, when used on correlated features it gave me an accuracy of 0.92 compared to 0.96 when used on all features.
This means that random forest regression model works best when applied on all features instead of just using correlated features, which is the opposite result of what we got from the linear regression model. 


-	Conclusion


After using both Linear regression model and Random forest regression model we can conclude multiple things:

-	Linear regression model has a higher accuracy than the random forest model.

-	When using Linear regression model it is best for us to try to first find the correlated features and then use the model on them because that will give us a higher accuracy as we have already seen.  

-	When using Random forest model it is best to use it on all features instead of using it on the correlated data as that will give us a higher accuracy 



                
