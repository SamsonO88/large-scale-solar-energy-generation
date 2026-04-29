# import modules
import json
import numpy as np
import pandas as pd
from app.solar_irradiation import Get_Radiation

def Solar_Estimator():
        """
            Get inputs from users and return estimated solar radiation. 
        """

    results = Get_Radiation(raw_data)
        
    return results



def return_prediction(data):
    url = 'https://solarwebappmodule.herokuapp.com/'

    try:
        prediction = requests.post(url, data)
        result = prediction['solar irradiation']
            
    except:
        return None
    
    return result


    # = st.selectbox('Gender',("Male","Female"))
    # = st.selectbox('Marital Status',("Unmarried","Married"))
    # = st.selectbox('Credit_History',("Unclear Debts","No Unclear Debts"))