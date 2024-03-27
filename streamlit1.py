import streamlit as st
import pandas as pd 
import warnings 
warnings.filterwarnings('ignore')
import joblib

st.markdown("<h1 style = 'color: #FFC700; text-align: center; font-family: trebuchet ms'>EXPRESSO CHURN DATASET</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: tangerine'> Built By: AKERELE HAMZAT</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html= True)

st.image('pngwing.com (6).png', use_column_width=True)
st.header('Project Background Information', divider = True)
st.write("The overarching objective of this ambitious project is to meticulously engineer a highly sophisticated predictive model meticulously designed to meticulously assess the intricacies of startup profitability. By harnessing the unparalleled power and precision of cutting-edge machine learning methodologies, our ultimate aim is to furnish stakeholders with an unparalleled depth of insights meticulously delving into the myriad factors intricately interwoven with a startup's financial success. Through the comprehensive analysis of extensive and multifaceted datasets, our mission is to equip decision-makers with a comprehensive understanding of the multifarious dynamics shaping the trajectory of burgeoning enterprises. Our unwavering commitment lies in empowering stakeholders with the indispensable tools and knowledge requisite for making meticulously informed decisions amidst the ever-evolving landscape of entrepreneurial endeavors.")

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

data = pd.read_csv('expresso_processed.csv')
st.dataframe(data)

st.sidebar.image('pngwing.com (7).png', caption='Welcome User')

st.sidebar.markdown("<br>", unsafe_allow_html= True)
st.sidebar.markdown("<br>", unsafe_allow_html= True)

# Decleare user input variables
st.sidebar.subheader('Input Variable', divider=True)
tenure = st.sidebar.selectbox('TENURE', data['TENURE'].unique())
montant = st.sidebar.number_input('MONTANT')
freq_rech = st.sidebar.number_input('FREQUENCE_RECH')
rev = st.sidebar.number_input('REVENUE')
arpu_seg = st.sidebar.number_input('ARPU_SEGMENT')
freq = st.sidebar.number_input('FREQUENCE')
data_vol = st.sidebar.number_input('DATA_VOLUME')
on_net = st.sidebar.number_input('ON_NET')
mrg = st.sidebar.selectbox('MRG', ['NO'])
reg = st.sidebar.number_input('REGULARITY')

# Display the users-input
input_var = pd.DataFrame()
input_var['TENURE'] = [tenure]
input_var['MONTANT'] = [montant]
input_var['FREQUENCE_RECH'] = [freq_rech]
input_var['REVENUE'] = [rev]
input_var['ARPU_SEGMENT'] = [arpu_seg]
input_var['FREQUENCE'] = [freq]
input_var['DATA_VOLUME'] = [data_vol]
input_var['ON_NET'] = [on_net]
input_var['MRG'] = [mrg]
input_var['REGULARITY'] = [reg]

st.markdown("<br>", unsafe_allow_html= True)
st.divider()
st.subheader('User Input Variables', divider=True)
st.dataframe(input_var, use_container_width=True)

# Importing Transformers
tenures = joblib.load('TENURE_encoder.pkl')
montant = joblib.load('MONTANT_scaling.pkl')
arpu_seg = joblib.load('ARPU_SEGMENT_scaling.pkl')
rev = joblib.load('REVENUE_scaling.pkl')
mrg = joblib.load('MRG_encoder.pkl')

# Applying transformations to the user's
input_var['TENURE'] = tenures.transform(input_var[['TENURE']])
input_var['MONTANT'] = montant.transform(input_var[['MONTANT']])
input_var['ARPU_SEGMENT'] = arpu_seg.transform(input_var[['ARPU_SEGMENT']])
input_var['REVENUE'] = rev.transform(input_var[['REVENUE']])
input_var['MRG'] = mrg.transform(input_var[['MRG']])

# st.dataframe(input_var)

model = joblib.load('ExpressChurn.pkl')
prediction = model.predict(input_var)

if st.button('Check If Customer is Churned or not'):
    if prediction[0] == 0:
        st.error('Customer has been CHURNED')
        
    else:
        st.success('Customer is still Actively with us')
       