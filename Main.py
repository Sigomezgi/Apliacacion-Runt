# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 17:00:00 2022

@author: usuario
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime as dte
import calendar
import numpy as np
import joblib
from PIL import Image
from streamlit_player import st_player


#image = Image.open('runt.PNG')

header = st.container()
Dataset = st.container()
Features = st.container()
Prediction = st.container()
Resultado  = st.container()

st.markdown(
    '''
    <style>
    .main{
        background-color: #F5F5F5;
    }
    </style>    
    ''',
    unsafe_allow_html=True
)
background_color = '#F5F5F5'

st.markdown("""
<style>
.big-font {
    font-size:100px !important;
    align : Center;
    color: #00aae4;
}
</style>
""", unsafe_allow_html=True)

def get_data(finename):
    txi_data = pd.read_csv(finename, parse_dates = ['Fecha'])
    return txi_data



with header:
    st.title('RUNTapp')
    st_player('https://www.youtube.com/watch?v=MPb90009yWk')
    
with Dataset:
    url = 'https://www.runt.com.co/'
    st.header('RUNT: Registro Único Nacional de [Tránsito](%s)'%url)
    
    Df0 = pd.read_csv('registros_autos.csv')
    
    Df0['Registros'] = Df0['Unidades']
    Df0 = Df0[['Fecha','Registros']]
    Df0 = Df0.head(60)
    
    fig1 = go.Figure(data = go.Table(
        header = dict(values = list(Df0[['Fecha','Registros']].columns),
                      fill_color = '#00aae4',#FD8E72
                      align = 'center',
                      font=dict( color = '#000000',size=22)),
        cells = dict(values = [Df0.Fecha, Df0.Registros],
                     fill_color = '#E5ECF6',
                     align = 'center',
                     font=dict( color = '#0a0a0a',size=14 ))))
    
    fig1.update_layout(margin = dict(l = 5,r = 5,b= 10,t=10),
                       paper_bgcolor = background_color)
    st.write(fig1)
    
    Df = get_data('registros_autos.csv')
    Df['Año']= pd.DatetimeIndex(Df['Fecha']).year
    Df['Mes']= pd.DatetimeIndex(Df['Fecha']).month
    
    #namemonth = {1 : 'Ene', 2:'Feb',3:'Mar', 4:'Abr',5:'May',6:'Jun',7:'Jul',8:'Ago',9:'Sep',10:'Oct',11:'Nov',12:'Dic'}
    
    #Df['NamMes'] = Df['Mes'].apply(lambda x: namemonth[x])
    
    df = Df.groupby(['Mes','Año'])['Unidades'].sum().unstack()
    
    fig = px.line(df)
    
    fig.update_layout(
    
    yaxis=dict(
        title_text="Registros",
        titlefont=dict(size=25),
        tickfont = dict(size=15)
    ),
    xaxis=dict(
        title_text="Mes",
        titlefont=dict(size=25),
        tickfont = dict(size=15)
    )
    )
    
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    
    st.write(fig)
    
with Prediction:
    
    
    st.header('Predicción')
    sel_col, col3 = st.columns(2)
    
    
    sel_col.info('¿Para que período de tiempo desea hacer la predicción?')
    intervalperiod = sel_col.selectbox( 'Período',options=['Día', 'Semana', 'Mes'], index=0)
    
    Fecha_inicial = sel_col.date_input('Inicio')

    if (intervalperiod != 'Día'):
        Fecha_final = sel_col.date_input('Final')
    else:
        Fecha_final = Fecha_inicial

    if(intervalperiod != 'Día' and Fecha_final < Fecha_inicial):

        sel_col.error('Revisa las fechas ingresadas')
        
    Df_festivos = pd.read_csv('festivos.csv',parse_dates = ['Fecha'])
    
    z = Fecha_inicial
    k = []
    while z <= Fecha_final:
        k.append(z)
        z = z + dte.timedelta(days=1)
    
    pred = pd.DataFrame(k, columns = ['Fecha'])
    
    pred['Festivo'] = pred['Fecha'].apply(lambda x : 1 if x in Df_festivos['Fecha'].unique() else 0)
    pred['Year']= pd.DatetimeIndex(pred['Fecha']).year
    pred['Month']= pd.DatetimeIndex(pred['Fecha']).month
    pred['Day']= pd.DatetimeIndex(pred['Fecha']).day
    pred['Dayweek'] = pred['Fecha'].apply(lambda x : calendar.day_name[x.weekday()])
    
    namemonth = {1 : 'Jan', 2:'Feb',3:'Mar', 4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    Vacaciones = [1,6,7,11,12]
    pred['NamMonth'] = pred['Month'].apply(lambda x: namemonth[x])
    pred['Holidays'] = pred['Month'].apply(lambda x: 1 if x in Vacaciones else 0)
    
    pred = pred[['Year','Month','Day','Dayweek','Festivo','Holidays']]
    
    num = ['Year', 'Month','Day', 'Festivo','Holidays','Dayweek_Friday','Dayweek_Monday','Dayweek_Saturday',
           'Dayweek_Sunday','Dayweek_Thursday','Dayweek_Tuesday','Dayweek_Wednesday']
           
    col3.markdown("![Alt Text](data:image/png;base64,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)")        
    pred = pd.get_dummies(pred)
    
    boton = sel_col.button('Predecir')
with Resultado:
    
    if boton:
        
        num2 =  list(pred.columns.values)
        dim = (pred.shape[0],1)
        for i in num:
            if i not in num2:
                pred[i] = np.zeros(dim)
        
        pred = pred[num]
        
        modelo = joblib.load('modelo_runt_entrenado.pkl')
        
        yrunt = modelo.predict(pred)
        st.balloons()
        yrunt = round(sum(yrunt))
        
        
        
        st.markdown('<p class="big-font"> %s Registros !!!</p>'%yrunt, unsafe_allow_html=True)
        #col3.markdown('<p class="big-font">Registros !!</p>', unsafe_allow_html=True)
        #st.image(image)
        #st.markdown("![Alt Text](https://giphy.com/gifs/mediamodifier-eivVnPUWgH6mzLORxn?utm_source=media-link&utm_medium=landing&utm_campaign=Media%20Links&utm_term=https://giphy.com/)") 
        
    
        
        
        
        