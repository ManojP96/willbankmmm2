import streamlit as st
from numerize.numerize import numerize
import pandas as pd
from utilities import (format_numbers,decimal_formater,
                       load_local_css,set_header,
                       initialize_data,
                       load_authenticator)
import pickle
import streamlit_authenticator as stauth
import yaml
from yaml import SafeLoader
from classes import class_from_dict
import plotly.express as px 
import numpy as np

def summary_plot(data, x, y, title, text_column, color, format_as_percent=False):
    fig = px.bar(data, x=x, y=y, orientation='h',
                 title=title, text=text_column, color=color)
    fig.update_layout(showlegend=False)
    data[text_column] = pd.to_numeric(data[text_column], errors='coerce')
    
    # Update the format of the displayed text based on magnitude
    if format_as_percent:
        fig.update_traces(texttemplate='%{text:.0%}', textposition='outside', hovertemplate='%{x:.0%}')
    else:
        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside', hovertemplate='%{x:.2s}')
    
    fig.update_layout(xaxis_title=x, yaxis_title='Channel Name', showlegend=False)
    return fig

    # Convert text_column to numeric values
    
st.set_page_config(layout='wide')
load_local_css('styles.css')
set_header()

# for k, v in st.session_state.items():
#     if k not in ['logout', 'login','config'] and not k.startswith('FormSubmitter'):
#         st.session_state[k] = v

st.empty()
st.header('Model Result Analysis')
spends_data=pd.read_excel('Overview_data_test.xlsx')

with open('summary_df.pkl', 'rb') as file:
  summary_df_sorted = pickle.load(file)

selected_scenario= st.selectbox('Select Saved Scenarios',['S1','S2']) 

st.header('Optimized Spends Overview')
___columns=st.columns(3)
with ___columns[2]:
    fig=summary_plot(summary_df_sorted, x='Delta_percent', y='Channel_name', title='Delta', text_column='Delta_percent',color='Channel_name')
    st.plotly_chart(fig,use_container_width=True)
with ___columns[0]:
    fig=summary_plot(summary_df_sorted, x='Actual_spend', y='Channel_name', title='Actual Spend', text_column='Actual_spend',color='Channel_name')
    st.plotly_chart(fig,use_container_width=True)         
with ___columns[1]:
    fig=summary_plot(summary_df_sorted, x='Optimized_spend', y='Channel_name', title='Planned Spend', text_column='Optimized_spend',color='Channel_name')
    st.plotly_chart(fig,use_container_width=False)

st.header(' Budget Allocation')
summary_df_sorted['Perc_alloted']=np.round(summary_df_sorted['Optimized_spend']/summary_df_sorted['Optimized_spend'].sum(),2)
columns2=st.columns(2)
with columns2[0]:
    fig=summary_plot(summary_df_sorted, x='Optimized_spend', y='Channel_name', title='Planned Spend', text_column='Optimized_spend',color='Channel_name')
    st.plotly_chart(fig,use_container_width=True)
with columns2[1]:
    fig=summary_plot(summary_df_sorted, x='Perc_alloted', y='Channel_name', title='% Split', text_column='Perc_alloted',color='Channel_name',format_as_percent=True)
    st.plotly_chart(fig,use_container_width=True)


if 'raw_data' not in st.session_state:
    st.session_state['raw_data']=pd.read_excel('raw_data_nov7_combined1.xlsx')
    st.session_state['raw_data']=st.session_state['raw_data'][st.session_state['raw_data']['MediaChannelName'].isin(summary_df_sorted['Channel_name'].unique())] 
    st.session_state['raw_data']=st.session_state['raw_data'][st.session_state['raw_data']['Date'].isin(spends_data["Date"].unique())]



#st.write(st.session_state['raw_data']['ResponseMetricName']) 
# st.write(st.session_state['raw_data'])


st.header('Response Forecast Overview')
raw_data=st.session_state['raw_data']
effectiveness_overall=raw_data.groupby('ResponseMetricName').agg({'ResponseMetricValue': 'sum'}).reset_index()
effectiveness_overall['Efficiency']=effectiveness_overall['ResponseMetricValue'].map(lambda x: x/raw_data['Media Spend'].sum() if x not in ["Total Approved Accounts - Revenue",'BAU approved clients - Revenue'] else raw_data['Media Spend'].sum()/x )
# st.write(effectiveness_overall)
columns4= st.columns(3)

with columns4[0]:
    fig=summary_plot(effectiveness_overall, x='ResponseMetricValue', y='ResponseMetricName', title='Effectiveness', text_column='ResponseMetricValue',color='ResponseMetricName')
    st.plotly_chart(fig,use_container_width=True) 

with columns4[1]:
    fig=summary_plot(effectiveness_overall, x='Efficiency', y='ResponseMetricName', title='Efficiency', text_column='ResponseMetricValue',color='ResponseMetricName')
    st.plotly_chart(fig,use_container_width=True) 

# st.header('Return Forecast by Media Channel')
with st.expander("Return Forecast by Media Channel"):
    metric_data=[val for val in list(st.session_state['raw_data']['ResponseMetricName'].unique()) if val!=np.NaN] 
    # st.write(metric_data)
    metric=st.selectbox('Select Metric',metric_data,index=1)

    selected_metric=st.session_state['raw_data'][st.session_state['raw_data']['ResponseMetricName']==metric]
    # st.dataframe(selected_metric.head(2))
    selected_metric=st.session_state['raw_data'][st.session_state['raw_data']['ResponseMetricName']==metric]
    effectiveness=selected_metric.groupby(by=['MediaChannelName'])['ResponseMetricValue'].sum()
    effectiveness_df=pd.DataFrame({'Channel':effectiveness.index,"ResponseMetricValue":effectiveness.values})

    summary_df_sorted=summary_df_sorted.merge(effectiveness_df,left_on="Channel_name",right_on='Channel')

    # st.dataframe(summary_df_sorted.head(2))
    summary_df_sorted['Efficiency']=summary_df_sorted['ResponseMetricValue']/summary_df_sorted['Optimized_spend']
# # # st.dataframe(summary_df_sorted.head(2))
# st.dataframe(summary_df_sorted.head(2))

    columns= st.columns(3)
    with columns[0]:
        fig=summary_plot(summary_df_sorted, x='Optimized_spend', y='Channel_name', title='', text_column='Optimized_spend',color='Channel_name')
        st.plotly_chart(fig,use_container_width=True)  
    with columns[1]: 
        
        # effectiveness=(selected_metric.groupby(by=['MediaChannelName'])['ResponseMetricValue'].sum()).values
        # effectiveness_df=pd.DataFrame({'Channel':st.session_state['raw_data']['MediaChannelName'].unique(),"ResponseMetricValue":effectiveness})
        # # effectiveness.reset_index(inplace=True)
        # # st.dataframe(effectiveness.head())
        fig=summary_plot(summary_df_sorted, x='ResponseMetricValue', y='Channel_name', title='Effectiveness', text_column='ResponseMetricValue',color='Channel_name')
        st.plotly_chart(fig,use_container_width=True)  

    with columns[2]:
        fig=summary_plot(summary_df_sorted, x='Efficiency', y='Channel_name', title='Efficiency', text_column='Efficiency',color='Channel_name')
        st.plotly_chart(fig,use_container_width=True) 


# st.dataframe(st.session_state['raw_data'].head())

