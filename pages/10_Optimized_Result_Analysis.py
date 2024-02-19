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
import plotly.graph_objects as go
import pandas as pd


def summary_plot(data, x, y, title, text_column, color, format_as_percent=False, format_as_decimal=False):
    fig = px.bar(data, x=x, y=y, orientation='h',
                 title=title, text=text_column, color=color)
    fig.update_layout(showlegend=False)
    data[text_column] = pd.to_numeric(data[text_column], errors='coerce')
    
    # Update the format of the displayed text based on the chosen format
    if format_as_percent:
        fig.update_traces(texttemplate='%{text:.0%}', textposition='outside', hovertemplate='%{x:.0%}')
    elif format_as_decimal:
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside', hovertemplate='%{x:.2f}')
    else:
        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside', hovertemplate='%{x:.2s}')
    
    fig.update_layout(xaxis_title=x, yaxis_title='Channel Name', showlegend=False)
    return fig


def stacked_summary_plot(data, x, y, title, text_column, color_column, stack_column=None, format_as_percent=False, format_as_decimal=False):
    fig = px.bar(data, x=x, y=y, orientation='h',
                 title=title, text=text_column, color=color_column, facet_col=stack_column)
    fig.update_layout(showlegend=False)
    data[text_column] = pd.to_numeric(data[text_column], errors='coerce')

    # Update the format of the displayed text based on the chosen format
    if format_as_percent:
        fig.update_traces(texttemplate='%{text:.0%}', textposition='outside', hovertemplate='%{x:.0%}')
    elif format_as_decimal:
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside', hovertemplate='%{x:.2f}')
    else:
        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside', hovertemplate='%{x:.2s}')

    fig.update_layout(xaxis_title=x, yaxis_title='', showlegend=False)
    return fig



def funnel_plot(data, x, y, title, text_column, color_column, format_as_percent=False, format_as_decimal=False):
    data[text_column] = pd.to_numeric(data[text_column], errors='coerce')

    # Round the numeric values in the text column to two decimal points
    data[text_column] = data[text_column].round(2)

    # Create a color map for categorical data
    color_map = {category: f'rgb({i * 30 % 255},{i * 50 % 255},{i * 70 % 255})' for i, category in enumerate(data[color_column].unique())}
    
    fig = go.Figure(go.Funnel(
        y=data[y],
        x=data[x],
        text=data[text_column],
        marker=dict(color=data[color_column].map(color_map)),
        textinfo="value",
        hoverinfo='y+x+text'
    ))

    # Update the format of the displayed text based on the chosen format
    if format_as_percent:
        fig.update_layout(title=title, funnelmode="percent")
    elif format_as_decimal:
        fig.update_layout(title=title, funnelmode="overlay")
    else:
        fig.update_layout(title=title, funnelmode="group")

    return fig


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
effectiveness_overall['Efficiency']=effectiveness_overall['ResponseMetricValue'].map(lambda x: x/raw_data['Media Spend'].sum() )
# st.write(effectiveness_overall)

columns6=st.columns(3)
columns4= st.columns([0.55,0.45])
effectiveness_overall.sort_values(by=['ResponseMetricValue'],ascending=False,inplace=True)
effectiveness_overall=np.round(effectiveness_overall,2)
effectiveness_overall['ResponseMetric'] = effectiveness_overall['ResponseMetricName'].apply(lambda x: 'BAU' if 'BAU' in x else ('Gamified' if 'Gamified' in x else x))
# effectiveness_overall=np.where(effectiveness_overall[effectiveness_overall['ResponseMetricName']=="Adjusted Account Approval BAU"],"Adjusted Account Approval BAU",effectiveness_overall['ResponseMetricName'])

effectiveness_overall.replace({'ResponseMetricName':{'BAU approved clients - Appsflyer':'Approved clients - Appsflyer',
                                                     'Gamified approved clients - Appsflyer':'Approved clients - Appsflyer'}},inplace=True)

# st.write(effectiveness_overall.sort_values(by=['ResponseMetricValue'],ascending=False))


condition = effectiveness_overall['ResponseMetricName'] == "Adjusted Account Approval BAU"
condition1= effectiveness_overall['ResponseMetricName'] == "Approved clients - Appsflyer"
effectiveness_overall['ResponseMetric'] = np.where(condition, "Adjusted Account Approval BAU", effectiveness_overall['ResponseMetric'])

effectiveness_overall['ResponseMetricName'] = np.where(condition1, "Approved clients - Appsflyer (BAU, Gamified)", effectiveness_overall['ResponseMetricName'])
# effectiveness_overall=pd.DataFrame({'ResponseMetricName':["App Installs - Appsflyer",'Account Requests - Appsflyer',
#                                                           'Total Adjusted Account Approval','Adjusted Account Approval BAU',
#                                                           'Approved clients - Appsflyer','Approved clients - Appsflyer'],
#                                     'ResponseMetricValue':[683067,367020,112315,79768,36661,16834],
#                                     'Efficiency':[1.24,0.67,0.2,0.14,0.07,0.03],
custom_colors = {
    'App Installs - Appsflyer': 'rgb(255, 135, 0)',       # Steel Blue (Blue)
    'Account Requests - Appsflyer': 'rgb(125, 239, 161)',  # Cornflower Blue (Blue)
    'Adjusted Account Approval': 'rgb(129, 200, 255)',      # Dodger Blue (Blue)
    'Adjusted Account Approval BAU': 'rgb(255, 207, 98)',  # Light Sky Blue (Blue)
    'Approved clients - Appsflyer': 'rgb(0, 97, 198)',  # Light Blue (Blue)
    "BAU": 'rgb(41, 176, 157)',                              # Steel Blue (Blue)
     "Gamified": 'rgb(213, 218, 229)'                      # Silver (Gray)
    # Add more categories and their respective shades of blue as needed
}






with columns6[0]:
    revenue=(effectiveness_overall[effectiveness_overall['ResponseMetricName']=='Total Approved Accounts - Revenue']['ResponseMetricValue']).iloc[0]
    revenue=round(revenue / 1_000_000, 2)

    st.metric('Total Revenue', f"${revenue} M")
with columns6[1]:
    BAU=(effectiveness_overall[effectiveness_overall['ResponseMetricName']=='BAU approved clients - Revenue']['ResponseMetricValue']).iloc[0]
    BAU=round(BAU / 1_000_000, 2)
    st.metric('BAU approved clients - Revenue', f"${BAU} M")
with columns6[2]:
    Gam=(effectiveness_overall[effectiveness_overall['ResponseMetricName']=='Gamified approved clients - Revenue']['ResponseMetricValue']).iloc[0]
    Gam=round(Gam / 1_000_000, 2)
    st.metric('Gamified approved clients - Revenue', f"${Gam} M")

# st.write(effectiveness_overall)
effectiveness_overall['Response Metric Name']=effectiveness_overall['ResponseMetricName']
with columns4[0]:
    fig=px.funnel(effectiveness_overall[~(effectiveness_overall['ResponseMetricName'].isin(['Total Approved Accounts - Revenue',
                                                                                          'BAU approved clients - Revenue',
                                                                                          'Gamified approved clients - Revenue',
                                                                                          "Total Approved Accounts - Appsflyer"]))],
                                                                                            x='ResponseMetricValue', y='Response Metric Name',color='ResponseMetric',
                                                                                            color_discrete_map=custom_colors,title='Effectiveness',
                                                                                            labels=None)
    custom_y_labels=['App Installs - Appsflyer','Account Requests - Appsflyer','Adjusted Account Approval','Adjusted Account Approval BAU',
                     "Approved clients - Appsflyer (BAU, Gamified)"
                     ]
    fig.update_layout(showlegend=False,
    yaxis=dict(
        tickmode='array',
        ticktext=custom_y_labels,
        )
        )
    fig.update_traces(textinfo='value', textposition='inside', texttemplate='%{x:.2s} ', hoverinfo='y+x+percent initial')

    last_trace_index = len(fig.data) - 1
    fig.update_traces(marker=dict(line=dict(color='black', width=2)), selector=dict(marker=dict(color='blue')))

    st.plotly_chart(fig,use_container_width=True)





with columns4[1]:

# Your existing code for creating the bar chart
    fig1 = px.bar((effectiveness_overall[~(effectiveness_overall['ResponseMetricName'].isin(['Total Approved Accounts - Revenue',
                                                                                            'BAU approved clients - Revenue',
                                                                                            'Gamified approved clients - Revenue',
                                                                                            "Total Approved Accounts - Appsflyer"]))]).sort_values(by='ResponseMetricValue'),
                x='Efficiency', y='Response Metric Name',
                color_discrete_map=custom_colors, color='ResponseMetric',
                labels=None,text_auto=True,title='Efficiency'
                )

    # Update layout and traces
    fig1.update_traces(customdata=effectiveness_overall['Efficiency'],
                   textposition='auto')
    fig1.update_layout(showlegend=False) 
    fig1.update_yaxes(title='',showticklabels=False)
    fig1.update_xaxes(title='',showticklabels=False)
    fig1.update_xaxes(tickfont=dict(size=20))
    fig1.update_yaxes(tickfont=dict(size=20))
    st.plotly_chart(fig1, use_container_width=True)


effectiveness_overall_revenue=pd.DataFrame({'ResponseMetricName':['Approved Clients','Approved Clients'],
                                            'ResponseMetricValue':[70201070,1768900],
                                            'Efficiency':[127.54,3.21],
                                            'ResponseMetric':['BAU','Gamified']
                                            })
# from plotly.subplots import make_subplots
# fig = make_subplots(rows=1, cols=2, 
#                     subplot_titles=["Effectiveness", "Efficiency"])

# # Add first plot as subplot
# fig.add_trace(go.Funnel(
#     x = fig.data[0].x,
#     y = fig.data[0].y,
#     textinfo = 'value+percent initial',
#     hoverinfo = 'x+y+percent initial'
# ), row=1, col=1)

# # Update layout for first subplot
# fig.update_xaxes(title_text="Response Metric Value", row=1, col=1) 
# fig.update_yaxes(ticktext = custom_y_labels, row=1, col=1)

# # Add second plot as subplot
# fig.add_trace(go.Bar(
#     x = fig1.data[0].x, 
#     y = fig1.data[0].y,
#     customdata = fig1.data[0].customdata, 
#     textposition = 'auto'
# ), row=1, col=2)

# # Update layout for second subplot
# fig.update_xaxes(title_text="Efficiency", showticklabels=False, row=1, col=2)
# fig.update_yaxes(title='', showticklabels=False, row=1, col=2)

# fig.update_layout(height=600, width=800, title_text="Key Metrics")
# st.plotly_chart(fig)


st.header('Return Forecast by Media Channel')
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
        fig=summary_plot(summary_df_sorted, x='Efficiency', y='Channel_name', title='Efficiency', text_column='Efficiency',color='Channel_name',format_as_decimal=True)
        st.plotly_chart(fig,use_container_width=True)

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create figure with subplots
# fig = make_subplots(rows=1, cols=2)

# # Add funnel plot to subplot 1
# fig.add_trace(
#     go.Funnel(
#         x=effectiveness_overall[~(effectiveness_overall['ResponseMetricName'].isin(['Total Approved Accounts - Revenue', 'BAU approved clients - Revenue', 'Gamified approved clients - Revenue', "Total Approved Accounts - Appsflyer"]))]['ResponseMetricValue'],
#         y=effectiveness_overall[~(effectiveness_overall['ResponseMetricName'].isin(['Total Approved Accounts - Revenue', 'BAU approved clients - Revenue', 'Gamified approved clients - Revenue', "Total Approved Accounts - Appsflyer"]))]['ResponseMetricName'],
#         textposition="inside",
#         texttemplate="%{x:.2s}",
#         customdata=effectiveness_overall['Efficiency'],
#         hovertemplate="%{customdata:.2f}<extra></extra>"
#     ),
#     row=1, col=1
# )

# # Add bar plot to subplot 2 
# fig.add_trace(
#     go.Bar(
#         x=effectiveness_overall.sort_values(by='ResponseMetricValue')['Efficiency'], 
#         y=effectiveness_overall.sort_values(by='ResponseMetricValue')['ResponseMetricName'],
#         marker_color=effectiveness_overall['ResponseMetric'], 
#         customdata=effectiveness_overall['Efficiency'],
#         hovertemplate="%{customdata:.2f}<extra></extra>",
#         textposition="outside"
#     ),
#     row=1, col=2
# )

# # Update layout
# fig.update_layout(title_text="Effectiveness")
# fig.update_yaxes(title_text="", row=1, col=1)
# fig.update_yaxes(title_text="", showticklabels=False, row=1, col=2) 
# fig.update_xaxes(title_text="Efficiency", showticklabels=False, row=1, col=2)

# # Show figure
# st.plotly_chart(fig)
