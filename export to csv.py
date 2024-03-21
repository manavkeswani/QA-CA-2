import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv(r'C:\Users\keswa\Desktop\QA CA-@\Advertising Budget and Sales.csv')

# Define the DataFrame to store regression results
regression_results = pd.DataFrame(columns=['Advertising Budget', 'Intercept', 'Coefficient'])

# Iterate over each advertising budget
for column in ['TV Ad Budget', 'Radio Ad Budget', 'Newspaper Ad Budget']:
    X = data[[column]]
    y = data['Sales']
    
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Append results to DataFrame
    regression_results = regression_results.append({'Advertising Budget': column,
                                                    'Intercept': model.intercept_,
                                                    'Coefficient': model.coef_[0]}, 
                                                    ignore_index=True)

# Save DataFrame to CSV
regression_results.to_csv('regression_results.csv', index=False)

print("Regression results saved successfully!")
