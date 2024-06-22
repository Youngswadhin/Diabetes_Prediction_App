import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from PIL import Image

# Load the diabetes dataset
diabetes_df = pd.read_csv('diabetes.csv')

# Group the data by outcome to get a sense of the distribution
diabetes_mean_df = diabetes_df.groupby('Outcome').mean()

# Split the data into input and target variables
X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']

# Scale the input variables using StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create an SVM model with a linear kernel and balanced class weights
model = svm.SVC(kernel='linear', class_weight='balanced')

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the training and testing sets
train_y_pred = model.predict(X_train)
test_y_pred = model.predict(X_test)

# Calculate the accuracy of the model on the training and testing sets
train_acc = accuracy_score(train_y_pred, y_train)
test_acc = accuracy_score(test_y_pred, y_test)

# Create the Streamlit app
def app():
    st.set_page_config(page_title="Diabetes Prediction App", page_icon=":hospital:", layout="wide")

    # Custom CSS to change background color
    st.markdown(
        """
        <style>
        body {
            background-color: #00000; /* Light gray background */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Load and display a fixed image
    if 'step' not in st.session_state:
        st.session_state.step = 'start'

    if st.session_state.step == 'start':
        st.markdown('<div class="center-content">', unsafe_allow_html=True)
        image = Image.open("DR_iconw.png")
        st.image(image, width=400)
        st.title('iPredictDiabetes')
        name = st.text_input("Enter your name")
        if st.button('Next'):
            if not name:
                st.warning("Please enter your name to proceed.")
            else:
                st.session_state.name = name
                st.session_state.step = 'form'
                st.session_state.prediction = None
        st.markdown('</div>', unsafe_allow_html=True)

    elif st.session_state.step == 'form':
        st.markdown(
            """
            <style>
            .center-form {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 80vh;
                flex-direction: column;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        if st.session_state.prediction is None:
            st.title('Diabetes Prediction App - Step 2')
            st.header(f'Welcome {st.session_state.name}')

            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                with st.form(key='diabetes_form'):
                    st.subheader('Input Features')
                    preg = st.number_input('Pregnancies', min_value=0, max_value=17, value=3)
                    glucose = st.number_input('Glucose', min_value=0, max_value=199, value=117)
                    bp = st.number_input('Blood Pressure', min_value=0, max_value=122, value=72)
                    skinthickness = st.number_input('Skin Thickness', min_value=0, max_value=99, value=23)
                    insulin = st.number_input('Insulin', min_value=0, max_value=846, value=30)
                    bmi = st.number_input('BMI', min_value=0.0, max_value=67.1, value=32.0)
                    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.078, max_value=2.42, value=0.3725, step=0.001)
                    age = st.number_input('Age', min_value=21, max_value=81, value=29)

                    submit_button = st.form_submit_button(label='Predict')

                if submit_button:
                    # Make a prediction based on the user input
                    input_data = np.asarray([preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]).reshape(1, -1)
                    input_data_scaled = scaler.transform(input_data)
                    prediction = model.predict(input_data_scaled)
                    st.session_state.prediction = prediction[0]

                    st.experimental_rerun()
        else:
            st.sidebar.header('Input Features')
            st.sidebar.markdown('Please input the values for each feature below.')

            preg = st.sidebar.number_input('Pregnancies', min_value=0, max_value=17, value=3)
            glucose = st.sidebar.number_input('Glucose', min_value=0, max_value=199, value=117)
            bp = st.sidebar.number_input('Blood Pressure', min_value=0, max_value=122, value=72)
            skinthickness = st.sidebar.number_input('Skin Thickness', min_value=0, max_value=99, value=23)
            insulin = st.sidebar.number_input('Insulin', min_value=0, max_value=846, value=30)
            bmi = st.sidebar.number_input('BMI', min_value=0.0, max_value=67.1, value=32.0)
            dpf = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.078, max_value=2.42, value=0.3725, step=0.001)
            age = st.sidebar.number_input('Age', min_value=21, max_value=81, value=29)

            if st.sidebar.button('Predict'):
                # Make a prediction based on the user input
                input_data = np.asarray([preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]).reshape(1, -1)
                input_data_scaled = scaler.transform(input_data)
                prediction = model.predict(input_data_scaled)
                st.session_state.prediction = prediction[0]

            # Display the prediction to the user
            st.subheader('Prediction Result')
            st.markdown('**Based on the input features, the model predicts:**')
            if st.session_state.prediction == 1:
                st.warning('This person has diabetes.')
            else:
                st.success('This person does not have diabetes.')

        # Display some summary statistics about the dataset
        st.subheader('Dataset Summary')
        st.dataframe(diabetes_df.describe())

        # Display the distribution by outcome
        st.subheader('Distribution by Outcome')
        st.dataframe(diabetes_mean_df)

        # Display the model accuracy
        st.subheader('Model Accuracy')
        st.markdown(f'**Train set accuracy:** {train_acc:.2f}')
        st.markdown(f'**Test set accuracy:** {test_acc:.2f}')

if __name__ == '__main__':
    app()
