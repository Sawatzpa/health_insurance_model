import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# App-Titel
st.title("App: Prediction of Insurance Costs")
st.subheader(
    "(Note: This is a Portfolio Project. Dont use it for real world Predictions)"
)
st.markdown(
    "This app is for shwoing the result of a analysis of health insurance charges in realtion to other health related features.\
    Based on a Kaggle Dataset a GradientBoostingModel was trained to make predictions on insurance charges. Users can\
    predict health insurance Charges based on self chosen values."
)

st.subheader("Exploring the Dataset")
st.markdown(
    "The Dataset was obtained in Kaggle under this link: https://www.kaggle.com/datasets/willianoliveiragibin/healthcare-insurance?resource=download"
)
st.markdown(
    "The Dataset Contains the insurance Charges of a person and other related Attributes, like Age, Gender, if the Person smokes and so on"
)
st.markdown(
    "The nominal Features of the Dataset (Gender, Smoker, Region) got encoded beforehand into numericals"
)

# Loading and showing Data
df = pd.read_csv("health_insurance_model/data/insurance_encoded.csv")
df_show = pd.read_csv("health_insurance_model/data/insurance.csv")

st.write(df_show.head())

st.markdown("Here is a quick descriptive rundown of the Dataset")
col1, col2 = st.columns(2)
with col1:
    st.write(df_show.describe())

with col2:
    st.write(df_show.describe(include=["O"]))

# Plots showing realted Features and Dist. of Insurance Charges
df_corr = df.corr()
heatm = px.imshow(df_corr, title="Correlationmatrix of the present features")

st.plotly_chart(heatm)

st.markdown(
    "The heatmap shows, that the attribute Smoker is the by far most correlated feature to the insurance cost in our Dataset,\
            followed by age and bmi. The other features only show a marginal correlation to the insurance cost."
)

st.subheader("Mean Insurance Cost for smokers and non-smokers")
df_grouped = (
    df_show.groupby("smoker")
    .charges.mean()
    .reset_index(name="mean charges")
    .sort_values(by="mean charges", ascending=False)
)
bar = px.bar(df_grouped, x="smoker", y="mean charges")
bar.update_traces(width=0.4)
st.plotly_chart(bar)
diff = np.round(df_grouped["mean charges"].max() - df_grouped["mean charges"].min(), 2)
st.markdown(f"The mean difference between smoker and non-smokers is about {diff}$")

scatter = px.scatter(
    df,
    x="age",
    y="charges",
    color="smoker",
    title="Insurance cost by age and smoker/non-smoker",
)
scatter.update_yaxes(title="Insurance Charge in $")
scatter.update_xaxes(title="Age in years")


st.plotly_chart(scatter)

st.markdown(
    "The scatter shows the same kind of information. There is strong division in the data in terms of smokers and non-smokers,\
        while the insurance Cost increases more or less linearly with age."
)

hist = px.histogram(df, x="charges", title="Distribution of Insurance Charges")
hist.add_vline(
    df.charges.median(), annotation_text="Median Charges", line_color="white"
)
st.plotly_chart(hist)

# Input for Model
st.subheader("Predictive Model for Insurance Charges")
st.markdown(
    "In the following youre able to predict the Insurance Charges based on Inputs you choose.\
    A GradientBoostingRegressor Modell was trained to take in the attributes shown above and  predicht the insurance charge based on them"
)

age = st.number_input("Enter Age:", min_value=18, value=None)
sex = st.selectbox("Select your Gender:", df_show.sex.unique())
bmi = st.number_input("Enter BMI:", value=None)
children = st.number_input("Enter number of Children:", value=None)
smoker = st.selectbox("Select Smoker/non-Smoker:", df_show.smoker.unique())
region = st.selectbox("Select Region:", df_show.region.unique())

inputs = pd.Series([age, sex, bmi, children, smoker, region])

df_predictions = pd.DataFrame(
    data={
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region_northeast": False,
        "region_northwest": False,
        "region_southeast": False,
        "region_southwest": False,
    },
    index=[1],
)

if inputs.isna().sum() == 0:
    df_predictions["sex"] = df_predictions.sex.map({"female": 1, "male": 0})
    df_predictions["smoker"] = df_predictions.smoker.map({"yes": True, "no": False})

    for col in df_predictions.columns:
        if region in str(col):
            df_predictions[col] = True

    st.markdown("This is how the data is going to be presented to the model")
    st.write(df_predictions.head())
    st.subheader("Lets make some Predition !")

    gbr_model = joblib.load("health_insurance_model\model\gbr_insurance_model.pkl")

    if st.button("Predict"):
        prediction = gbr_model.predict(
            np.asarray(df_predictions.iloc[0, :]).reshape(1, -1)
        )
        st.success(f"Predicted Insurance Charge: {np.round(prediction[0],2)}$")
        df_predictions.insert(5, "charges", None)
        df_predictions["charges"] = prediction[0]

        df_show_prediction = pd.concat([df, df_predictions])
        percentile = (
            df_show_prediction[df_show_prediction["charges"] <= prediction[0]].shape[0]
            / len(df_show_prediction["charges"])
        ) * 100.0

        predict_hist = px.histogram(
            df_show_prediction, x="charges", title="Distribution of Charges"
        )
        predict_hist.add_vline(
            x=prediction[0],
            annotation_text=f"You pay less than {np.round(100-percentile,1)}% of the population!",
            line_color="white",
        )

        st.plotly_chart(predict_hist)

        if prediction and df_predictions["smoker"].iloc[0] == True:
            st.markdown("This is how much you would pay if you were a non-smoker:")
            df_predictions["smoker"] = False
            df_predictions = df_predictions.drop(columns=["charges"])
            prediction = gbr_model.predict(
                np.asarray(df_predictions.iloc[0, :]).reshape(1, -1)
            )
            st.write(f"Non-Smoker Insurance Charge: {np.round(prediction[0], 2)}$")
            df_show_prediction = pd.concat([df, df_predictions])
            percentile = (
                df_show_prediction[
                    df_show_prediction["charges"] <= prediction[0]
                ].shape[0]
                / len(df_show_prediction["charges"])
            ) * 100.0

            predict_hist = px.histogram(
                df_show_prediction,
                x="charges",
                title="Distribution of Charges (if you wouldnt smoke)",
            )
            predict_hist.add_vline(
                x=prediction[0],
                annotation_text=f"You pay less than {np.round(100-percentile,1)}% of the population!",
                line_color="white",
            )

            st.plotly_chart(predict_hist)
        elif prediction and df_predictions["smoker"].iloc[0] == False:
            st.markdown("This is how much you would if you were a smoker")
            df_predictions["smoker"] = True
            df_predictions = df_predictions.drop(columns=["charges"])
            prediction = gbr_model.predict(
                np.asarray(df_predictions.iloc[0, :]).reshape(1, -1)
            )
            st.write(f"Smoker Insurance Charge: {np.round(prediction[0], 2)}$")
            df_show_prediction = pd.concat([df, df_predictions])
            percentile = (
                df_show_prediction[
                    df_show_prediction["charges"] <= prediction[0]
                ].shape[0]
                / len(df_show_prediction["charges"])
            ) * 100.0

            predict_hist = px.histogram(
                df_show_prediction,
                x="charges",
                title="Distribution of Charges if you would smoke",
            )
            predict_hist.add_vline(
                x=prediction[0],
                annotation_text=f"You pay less than {np.round(100-percentile,1)}% of the population!",
                line_color="white",
            )

            st.plotly_chart(predict_hist)
        else:
            st.write("")
