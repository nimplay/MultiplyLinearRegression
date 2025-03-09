import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Configuración de estilos personalizados
st.markdown("""
<style>
.big-font {
    font-size: 30px !important;
    font-weight: bold;
    color: #4CAF50;
    text-align: center;
}
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #4CAF50;
    color: white;
    text-align: center;
    padding: 10px;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
    border-radius: 5px;
    padding: 10px 20px;
}
.stNumberInput>div>div>input {
    border: 2px solid #4CAF50;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# Título de la aplicación
st.markdown('<p class="big-font">📈 Sales Forecasting Using Multiple Linear Regression</p>', unsafe_allow_html=True)

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the data
    df = pd.read_csv(uploaded_file)

    # Show the data
    st.write("### Sales Dataset")
    st.write(df.head())

    # Validate the dataset
    required_columns = ['TV', 'Radio', 'Social_Media', 'Sales']
    if all(column in df.columns for column in required_columns):
        st.success("✅ The dataset contains all the required columns.")

        # Training the model
        X = df[['TV', 'Radio', 'Social_Media']]
        y = df['Sales']
        model = LinearRegression()
        model.fit(X, y)

        # Show model coefficients
        st.write("### Model Coefficients")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Intercept (β₀):** {model.intercept_:.2f}")
        with col2:
            st.write(f"**Coefficients (β₁, β₂, β₃):** {model.coef_}")

        # Visualizations
        st.write("### Relationship Between Predictors and Sales")

        # Plot 1: TV vs Sales
        st.write("#### TV Budget vs Sales")
        fig1, ax1 = plt.subplots()
        ax1.scatter(df['TV'], df['Sales'], color='blue', alpha=0.6)
        ax1.plot(df['TV'], model.intercept_ + model.coef_[0] * df['TV'], color='red', label='Regression Line')
        ax1.set_xlabel('TV Budget (in thousands of dollars)')
        ax1.set_ylabel('Sales (in thousands of dollars)')
        ax1.set_title('TV Budget vs Sales')
        ax1.legend()
        st.pyplot(fig1)

        # Plot 2: Radio vs Sales
        st.write("#### Radio Budget vs Sales")
        fig2, ax2 = plt.subplots()
        ax2.scatter(df['Radio'], df['Sales'], color='green', alpha=0.6)
        ax2.plot(df['Radio'], model.intercept_ + model.coef_[1] * df['Radio'], color='red', label='Regression Line')
        ax2.set_xlabel('Radio Budget (in thousands of dollars)')
        ax2.set_ylabel('Sales (in thousands of dollars)')
        ax2.set_title('Radio Budget vs Sales')
        ax2.legend()
        st.pyplot(fig2)

        # Plot 3: Social Media vs Sales
        st.write("#### Social Media Budget vs Sales")
        fig3, ax3 = plt.subplots()
        ax3.scatter(df['Social_Media'], df['Sales'], color='red', alpha=0.6)
        ax3.plot(df['Social_Media'], model.intercept_ + model.coef_[2] * df['Social_Media'], color='blue', label='Regression Line')
        ax3.set_xlabel('Social Media Budget (in thousands of dollars)')
        ax3.set_ylabel('Sales (in thousands of dollars)')
        ax3.set_title('Social Media Budget vs Sales')
        ax3.legend()
        st.pyplot(fig3)

        # User interface for forecasting
        st.write("### Sales Forecasting in thousands of dollars")
        col1, col2, col3 = st.columns(3)
        with col1:
            tv = st.number_input("TV Budget ", min_value=0, max_value=200, value=100)
        with col2:
            radio = st.number_input("Radio Budget ", min_value=0, max_value=50, value=25)
        with col3:
            social_media = st.number_input("Social Budget ", min_value=0, max_value=100, value=50)

        # Predicting sales
        if st.button("Predict Sales"):
            prediction = model.predict([[tv, radio, social_media]])
            st.success(f"Estimated sales are: **{prediction[0]:.2f}** thousands of dollars.")
    else:
        st.error("❌ The dataset is missing one or more required columns. Please upload a CSV file with the following columns: 'TV', 'Radio', 'Social_Media', 'Sales'.")
else:
    st.write("Please upload a CSV file to get started.")

# Footer
st.markdown("""
<div class="footer">
    <p>© 2025 Sales Forecasting App. By Nimrod Acosta All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
