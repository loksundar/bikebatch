import pickle
import streamlit as st
import pandas as pd
import numpy as np
#model = load_model('finalized_model')
model = pickle.load(open('finalized_model.pkl','rb'))




def predict(model, input_df):
    predictions_df = model.predict(input_df)
    predictions = predictions_df
    return predictions


def run():
    
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Realtime", "Batch"))
    st.sidebar.info('This app predicts the bike rental')
    st.title("Hourly bike prediction")
    if add_selectbox == 'Realtime':
        season=st.number_input('season' , min_value=1, max_value=4, value=1)
        holiday =st.number_input('holiday',min_value=0, max_value=1, value=0)
        workingday =st.number_input('workingday',min_value=0, max_value=1, value=0)
        weather =st.number_input('weather',min_value=1, max_value=4, value=1)
        temp= st.number_input('temp', min_value=0.0, max_value=40.0, value=30.0)
        humidity= st.number_input('humidity', min_value=0.0, max_value=100.0, value=50.0)
        windspeed= st.number_input('windspeed', min_value=0.0, max_value=100.0, value=50.0)
        casual= st.number_input('casual', min_value=0.0, max_value=400.0, value=200.0)
        registered= st.number_input('registered', min_value=0.0, max_value=700.0, value=200.0)
        hour= st.number_input('hour', min_value=0, max_value=10, value=0)
        month= st.number_input('month', min_value=1, max_value=12, value=1)
        
        output=""
        input_dict = {'season':season,'holiday':holiday,'workingday':workingday,'weather':weather,'temp':temp,'humidity': humidity,'windspeed':windspeed,'casual':casual,'registered' : registered,'hour':hour,'month':month}
        input_df = pd.DataFrame([input_dict])
        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = str(output)
        st.success('The output is {}'.format(output))

    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)

    
    
    
run()