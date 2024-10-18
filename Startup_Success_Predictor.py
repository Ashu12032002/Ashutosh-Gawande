import pandas as pd
import numpy as np
import streamlit as st
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Set the title
st.title("Startup Success Prediction System")

# Load the data
data = pd.read_csv(r"C:\Users\admin\Downloads\investments_VC.csv",encoding="unicode_escape")

# Create input fields for user to enter data
input_data = {
    'Total_Funding': st.number_input('Total Funding'),
    'Funding_Rounds': st.number_input('Funding Rounds'),
    'seed': st.number_input('seed'),
    'venture': st.number_input('venture'),
    'equity_crowdfunding': st.number_input('equity_crowdfunding'),
    'undisclosed': st.number_input('undisclosed'),
    'convertible_note': st.number_input('convertible_note'),
    'debt_financing': st.number_input('debt_financing'),
    'private_equity': st.number_input('private_equity'),
    'angel': st.number_input('angel'),
    'grant': st.number_input('grant'),
    'post_ipo_equity': st.number_input('post ipo equity'),
    'post_ipo_debt': st.number_input('post ipo debt'),
    'secondary_market': st.number_input('secondary_market'),
    'product_crowdfunding': st.number_input('product_crowdfunding'),
    'round_A': st.number_input('round_A'),
    'round_B': st.number_input('round_B'),
    'round_C': st.number_input('round_C'),
    'round_D': st.number_input('round_D'),
    'round_E': st.number_input('round_E'),
    'round_F': st.number_input('round_F'),
    'round_G': st.number_input('round_G'),
    'round_H': st.number_input('round_H')
}

# Label encoding for the target variable
label_encoding = preprocessing.LabelEncoder()
data['status'] = label_encoding.fit_transform(data['status'])

# Drop rows with missing values
data.dropna(inplace=True)

# Define features and target variable
x = data.iloc[:, [7, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36]]
y = data.iloc[:, 35]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Create and train the logistic regression model
model = LogisticRegression(solver='liblinear')
model.fit(x_train, y_train)

# Make predictions on the test data
y_pred = model.predict(x_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)


# Predict startup success using user input
if st.button("Predict", key="predict_button"):
    # Create a NumPy array from the input data
    input_array = np.array(list(input_data.values())).reshape(1, -1)

    # Predict the success using the model
    prediction = model.predict(input_array)

    # Decode the prediction back to original labels
    predicted_label = label_encoding.inverse_transform(prediction)[0]

    # Display the prediction
    st.write(f"Predicted Startup Success: {predicted_label}")

    # Provide an explanation of the prediction
    explanation = {
        0: "The startup is predicted to be acquired.",
        1: "The startup is predicted to be closed.",
        2: "The startup is predicted to be operational."
   
    }
    st.write("Prediction Explanation:")
    st.write(explanation[prediction[0]])


    
    