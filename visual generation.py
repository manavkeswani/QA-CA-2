# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv(r'C:\Users\keswa\Desktop\QA CA-@\Advertising Budget and Sales.csv')

# Display the first few rows of the dataset
print(data.head())

# Summary statistics of the dataset
print(data.describe())

# Scatter plots for each advertising budget vs. Sales
for column in ['TV Ad Budget', 'Radio Ad Budget', 'Newspaper Ad Budget']:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=column, y='Sales', data=data)
    plt.title(f'{column} vs. Sales')
    plt.xlabel(column)
    plt.ylabel('Sales')
    plt.show()

# Perform linear regression for each advertising budget vs. Sales
models = {}
for column in ['TV Ad Budget', 'Radio Ad Budget', 'Newspaper Ad Budget']:
    X = data[[column]]
    y = data['Sales']
    model = LinearRegression()
    model.fit(X, y)
    models[column] = {'Intercept': model.intercept_, 'Coefficient': model.coef_[0]}
    print(f'Intercept for {column}: {models[column]["Intercept"]}')
    print(f'Coefficient for {column}: {models[column]["Coefficient"]}')
    
# Create bar plots for intercepts
plt.figure(figsize=(10, 5))
plt.bar(models.keys(), [models[column]['Intercept'] for column in models], color='blue')
plt.title('Sales Intercept for Advertising Budgets')
plt.xlabel('Advertising Budget')
plt.ylabel('Sales (Intercept)')
plt.show()

# Create bar plots for coefficients
plt.figure(figsize=(10, 5))
plt.bar(models.keys(), [models[column]['Coefficient'] for column in models], color='green')
plt.title('Sales Coefficients for Advertising Budgets')
plt.xlabel('Advertising Budget')
plt.ylabel('Sales Coefficient')
plt.show()

# Calculate total budget for each channel
total_budgets = data[['TV Ad Budget', 'Radio Ad Budget', 'Newspaper Ad Budget']].sum()

# Create a pie chart
plt.figure(figsize=(8, 6))
plt.pie(total_budgets, labels=total_budgets.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Advertising Budgets')
plt.show()

# Calculate and visualize correlation matrices
correlation_matrix = data[['TV Ad Budget', 'Radio Ad Budget', 'Newspaper Ad Budget', 'Sales']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Analyze the obtained data and draw inferences
inferences = []

# Check if there is a positive or negative correlation between each advertising budget and sales
for column in ['TV Ad Budget', 'Radio Ad Budget', 'Newspaper Ad Budget']:
    correlation = data['Sales'].corr(data[column])
    if correlation > 0:
        inference = f"There is a positive correlation between {column} and sales (correlation coefficient: {correlation:.2f})."
    elif correlation < 0:
        inference = f"There is a negative correlation between {column} and sales (correlation coefficient: {correlation:.2f})."
    else:
        inference = f"There is no significant correlation between {column} and sales."
    inferences.append(inference)

# Check which advertising budget has the highest impact on sales based on coefficients
max_coefficient = max(models, key=lambda x: models[x]['Coefficient'])
inferences.append(f"The advertising budget with the highest impact on sales is {max_coefficient}.")

# Check if there are any outliers in the data
outliers = data[(data['Sales'] > 2 * data['Sales'].std() + data['Sales'].mean()) | 
                (data['Sales'] < data['Sales'].mean() - 2 * data['Sales'].std())]
if not outliers.empty:
    inferences.append("There are outliers present in the data.")
else:
    inferences.append("There are no outliers present in the data.")

# Print the generated inferences
for inference in inferences:
    print(inference)
