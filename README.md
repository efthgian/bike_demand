# Relationship Between Rental Bike Demand and Weather Data

In this endeavor we used an open dataset to explore the relationship between weather data and rental bike demand.

First things first, we give credit to the source of this open dataset, UC Irvine Machine Learning Repository. The dataset can be found here: https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand

For the analysis we use Python and its libraries, mainly pandas and scikit-learn. Firstly we load the data and store them in a dataframe, called data. We then proceed with EDA.

data.info()

![Screenshot 2025-01-01 170107](https://github.com/user-attachments/assets/310b7f83-efaf-4dcc-893e-e6a703db2e12)

From the picture above, we can examine the dimensions of the dataset, the variables, their data types and if there are null values, in this case there are none. We then change the type of the date column from object to datetime64[ns], so that our date is consistent.

We go deeper to our analysis by examining first the Functioning Day variable, which takes the values 0 and 1. When this variable takes the value 0, the Rented Bike Count, which is our main focus, takes also the value 0, so we drop Functioning Day variable, since it gives us no useful information.

data[data['Functioning Day'] == 'No']['Rented Bike Count'].unique()

Afterwards via the Date variable, we create two more columns, namely Month and Day, and we drop the Year since the dataset includes only one year.

Below we can see the bar graphs for every variable

![bargraphs1](https://github.com/user-attachments/assets/481f990f-f734-4b44-82fc-9615560e6d66)
![bargraphs2](https://github.com/user-attachments/assets/eb9434fa-1a9c-4bf3-80ca-6f33c9add14a)
![bargraphs3](https://github.com/user-attachments/assets/b144329c-207e-4df8-9325-c370854ccfba)
![bargraphs4](https://github.com/user-attachments/assets/78453953-62f6-4468-a4f4-2054b7760181)
![bargraphs5](https://github.com/user-attachments/assets/b205eb0f-e0aa-44c8-b897-140041e48ecb)

Then we examine the relationship between our dependent variable Rented Bike Count and Hour with parameters the Holiday and Day variables, and we notice some spike during the working hours:

![woking1](https://github.com/user-attachments/assets/814c720e-c6b4-4caa-9562-597c183592d2)
![working2](https://github.com/user-attachments/assets/2f7f5136-9654-4f2c-b410-ce74ba8720de)

These spikes tend to appear when people go to work, so we decide to create a new variable according to the matrix below:

![combined](https://github.com/user-attachments/assets/65470c5b-8687-47b0-a145-0b979266807e)

And we notice that these spikes appear only when people go to work, namely Hday, so we created one more column with values Workday for Hday and Dayoff for the remaining 3 variables.

![combinedgraph](https://github.com/user-attachments/assets/7e54f28c-84b6-4d8e-bc9b-b4751129da65)

## Training models

For the training of our models we use the scikit-learn library. Because our dataset includes also text as values, it is required for our algorithms to use an encoding for those values and transform them into numerical ones. One popular options is the One-Hot encoding.
After this transformation we save our data into a new variable, called dataset and we split the new variable into the dependent vector y (Rented Bike Count) and the independent matrix X.
We further split these matrices into train and test matrices with a ratio of 80 and 20 respectively. (We train our models with 80% of the observations and we test them afterwards with the remaining 20%)
We use a Linear Regression model and a Random Forest model and we realise that Random Forest results in a lower error score.
Briefly:

![learn1](https://github.com/user-attachments/assets/bc12aec2-420e-472d-9397-675c01d190e5)
![learn2](https://github.com/user-attachments/assets/72c93796-4646-41bf-8839-1324c26d85b4)
![learn3](https://github.com/user-attachments/assets/8dab87b3-8c97-4ce1-87f8-c3506a0c3f24)

Afterwards we created every combination of the independent variables and calculate the error for every combination. We notice, that using more than 6 parameters gives us no better results:

![rmse](https://github.com/user-attachments/assets/7fef9451-eb3a-4b73-b520-87191029e116)

Finally we fine tune our Random Forest model and we compute the relative importance between the variables that we choose:

![importance](https://github.com/user-attachments/assets/747026e2-a284-410b-b49c-5ee1457687a2)

We conculed by saying that the error of our model (RMSE) is about 178, which is a large error. If we want to predict the rental bike demand more accurately, we must look for different types of data, for example one might look forpublic transportation utilization, to complement the data we analyzed.
