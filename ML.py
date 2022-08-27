import streamlit as st
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
# from imblearn.over_sampling import RandomOverSampler


#应用标题
st.set_page_config(page_title='Pred hypertension in  patients with OSA')
st.title('Prediction Model of Obstructive Sleep Apnea-related Hypertension: Machine Learning–Based Development and Interpretation Study')

st.sidebar.markdown('## Variables')
BQL = st.sidebar.selectbox('Berlin Questionnaire(BQL)',('Low risk','High risk'),index=1)
SBSL = st.sidebar.selectbox('STOP-Bang questionnaire(SBSL)',('Low risk','High risk'),index=1)
HHD = st.sidebar.selectbox('Heart disease(HHD)',('No','Yes'),index=1)
FHH = st.sidebar.selectbox('Family history of hypertension(FHH)',('No','Yes'),index=0)
diabetes = st.sidebar.selectbox('Diabetes',('No','Yes'),index=0)
memory = st.sidebar.selectbox('Memory decline',('No','Yes'),index=0)
HSD = st.sidebar.selectbox('High-salt diet(HSD)',('No','Yes'),index=0)
PS = st.sidebar.selectbox('Poor sleep quality(PS)',('No','Yes'),index=0)
# EM = st.sidebar.selectbox('EM',('No','Yes'),index=0)
# stress = st.sidebar.selectbox('stress',('No','Yes'),index=0)
# gender = st.sidebar.selectbox('gender',('male','female'),index=0)
# age = st.sidebar.slider("age(years)", 15, 95, value=30, step=1)
BMI = st.sidebar.slider("BMI(kg/㎡)", 15.0, 40.0, value=20.0, step=0.1)
waistline = st.sidebar.slider("Waist circumference(cm)", 50, 150, value=100, step=1)
NC = st.sidebar.slider("Neck circumference(cm)", 20.0, 60.0, value=30.0, step=0.1)
# DrT = st.sidebar.slider("DrT(单位)", 0, 50, value=30, step=1)
# SmT = st.sidebar.slider("SmT(单位)", 0, 50, value=30, step=1)
SmA = st.sidebar.slider("Smoking amount(SmA)(package)", 0, 5, value=3, step=1)
SnT = st.sidebar.slider("Course of snoring(SnT)(day)", 0, 50, value=30, step=1)
SuT = st.sidebar.slider("Course of choking(SuT)(day)", 0, 30, value=15, step=1)
ESS = st.sidebar.slider("Epworth sleepiness scale(ESS)(point)", 0, 25, value=10, step=1)
# SBS = st.sidebar.slider("SBS(单位)", 0, 10, value=5, step=1)
AHI = st.sidebar.slider("AHI(point)", 5.0, 150.0, value=50.0, step=0.1)
OAL = st.sidebar.slider("OAI(point)", 0.0, 130.0, value=50.0, step=0.1)
ageper10 = st.sidebar.slider("Age/10(year)", 1.0, 10.0, value=5.0, step=0.1)
minPper10 = st.sidebar.slider("Minimum SaO2/10(%)", 1.0, 10.0, value=5.0, step=0.1)
P90per10 = st.sidebar.slider("CT90 /10(%)", 0.00, 10.00, value=5.00, step=0.01)
# P90 = st.sidebar.slider("P90(单位)", 0.0, 100.0, value=70.0, step=0.1)
#分割符号
st.sidebar.markdown('#  ')
st.sidebar.markdown('#  ')
st.sidebar.markdown('##### All rights reserved') 
st.sidebar.markdown('##### For communication and cooperation, please contact wshinana99@163.com, Wu Shi-Nan, Nanchang university')
#传入数据
map = {'No':0,'Yes':1,'Low risk':0,'High risk':1}
# X_data = new_data[["BQL","SBSL","HHD","FHH","diabetes","memory","HSD","PS","EM","stress","gender"
#              ,"age","BMI","waistline","NC","DrT","SmT","SmA","SnT","SnT","ESS","SBS","AHI",
#              "OAHI","minP","P90"]]
BQL =map[BQL]
SBSL =map[SBSL]
HHD =map[HHD]
FHH =map[FHH]
diabetes =map[diabetes]
memory =map[memory]
HSD =map[HSD]
PS =map[PS]
# EM =map[EM]
# stress =map[stress]
# gender =map[gender]
# age =map[age]
# BMI =map[BMI]
# waistline =map[waistline]
# NC =map[NC]
# DrT =map[DrT]
# SmT =map[SmT]
# SmA =map[SmA]
# SnT =map[SnT]
# SuT =map[SuT]
# ESS =map[ESS]
# SBS =map[SBS]
# AHI =map[AHI]
# OAHI =map[OAHI]
# minP =map[minP]
# P90 =map[P90]


# 数据读取，特征标注
hp_train = pd.read_csv('data_new.csv')

hp_train['hypertension'] = hp_train['hypertension'].apply(lambda x : +1 if x==1 else 0)

features = ["BQL", "SBSL", "HHD", "FHH", "diabetes", "memory", "HSD", "PS", "BMI",
                   "waistline", "NC", "SmA", "SnT", "SuT", "ESS", "AHI",
                   "OAL", "ageper10", "minPper10", "P90per10"]
target = 'hypertension'
random_state_new = 1000
# ros = RandomOverSampler(random_state=random_state_new, sampling_strategy='auto')
# X_ros, y_ros = ros.fit_resample(hp_train[features], hp_train[target])
X_ros = np.array(hp_train[features])
y_ros = np.array(hp_train[target])
gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_depth=1, random_state=random_state_new)
gbm.fit(X_ros, y_ros)



sp = 0.5
#figure
is_t = (gbm.predict_proba(np.array([[BQL,SBSL,HHD,FHH,diabetes,memory,HSD,PS,
                                     BMI,waistline,NC,SmA,SnT,SuT,ESS,AHI,OAL,ageper10,minPper10,P90per10]]))[0][1])> sp
prob = (gbm.predict_proba(np.array([[BQL,SBSL,HHD,FHH,diabetes,memory,HSD,PS,
                                     BMI,waistline,NC,SmA,SnT,SuT,ESS,AHI,OAL,ageper10,minPper10,P90per10
                                     ]]))[0][1])*1000//1/10


if is_t:
    result = 'High Risk'
else:
    result = 'Low Risk'
if st.button('Predict'):
    st.markdown('## Risk grouping for OSA-related hypertension:  '+str(result))
    if result == 'Low Risk':
        st.balloons()
    st.markdown('## Probability of High risk group:  '+str(prob)+'%')
