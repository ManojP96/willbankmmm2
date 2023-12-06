import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from utilities import set_header,initialize_data,load_local_css
from scipy.optimize import curve_fit
import statsmodels.api as sm
from plotly.subplots import make_subplots

st.set_page_config(
  page_title="Data Validation",
  page_icon=":shark:",
  layout="wide",
  initial_sidebar_state='collapsed'
)
load_local_css('styles.css')
set_header()

def format_numbers(x):
    if abs(x) >= 1e6:
        # Format as millions with one decimal place and commas
        return f'{x/1e6:,.1f}M'
    elif abs(x) >= 1e3:
        # Format as thousands with one decimal place and commas
        return f'{x/1e3:,.1f}K'
    else:
        # Format with one decimal place and commas for values less than 1000
        return f'{x:,.1f}'

def format_axis(x):
    if isinstance(x, tuple):
        x = x[0]  # Extract the numeric value from the tuple
    if abs(x) >= 1e6:
        return f'{x / 1e6:.0f}M'
    elif abs(x) >= 1e3:
        return f'{x / 1e3:.0f}k'
    else:
        return f'{x:.0f}'


attributred_app_installs=pd.read_csv("attributed_app_installs.csv")
attributred_app_installs_tactic=pd.read_excel('attributed_app_installs_tactic.xlsx')
data=pd.read_excel('Channel_wise_imp_click_spends.xlsx')
data['Date']=pd.to_datetime(data['Date'])
st.header('Saturation Curves')

# st.dataframe(data.head(2))
st.markdown('Data QC')

st.markdown('Channel wise summary')
summary_df=data.groupby(data['Date'].dt.strftime('%B %Y')).sum()
summary_df=summary_df.sort_index(key=lambda x: pd.to_datetime(x, format='%B %Y'))
st.dataframe(summary_df.applymap(format_numbers))



def line_plot_target(df,target,title):
    df=df
    df['Date_unix'] = df['Date'].apply(lambda x: x.timestamp())

# Perform polynomial fitting
    coefficients = np.polyfit(df['Date_unix'], df[target], 1)
    # st.dataframe(df)
    coefficients = np.polyfit(df['Date'].view('int64'), df[target], 1)
    trendline = np.poly1d(coefficients)
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df['Date'], y=df[target], mode='lines', name=target,line=dict(color='#11B6BD')))
    trendline_x = df['Date']
    trendline_y = trendline(df['Date'].view('int64'))


    fig.add_trace(go.Scatter(x=trendline_x, y=trendline_y, mode='lines', name='Trendline', line=dict(color='#739FAE')))

    fig.update_layout(
        title=title,
        xaxis=dict(type='date')
    )

    for year in df['Date'].dt.year.unique()[1:]:

        january_1 = pd.Timestamp(year=year, month=1, day=1)
        fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=january_1,
                x1=january_1,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(color="grey", width=1.5, dash="dash"),
            )
        )

    return fig
channels_d= data.columns[:28]
channels=list(set([col.replace('_impressions','').replace('_clicks','').replace('_spend','') for col in channels_d if col.lower()!='date']))
channel= st.selectbox('Select Channel_name',channels)
target_column = st.selectbox('Select Channel)',[col for col in data.columns if col.startswith(channel)])
fig=line_plot_target(data, target=str(target_column), title=f'{str(target_column)} Over Time')
st.plotly_chart(fig, use_container_width=True)

# st.markdown('## Saturation Curve')


st.header('Build saturation curve')

# Your data
# st.write(len(attributred_app_installs))
# st.write(len(data))
# col=st.columns(3)
# with col[0]:
col=st.columns(2)
with col[0]:
    if st.checkbox('Cap Outliers'):
        x = data[target_column]
        x.index=data['Date']
        # st.write(x)
        result = sm.tsa.seasonal_decompose(x, model='additive')
        x_resid=result.resid
        # fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        # trace_x = go.Scatter(x=data['Date'], y=x, mode='lines', name='x')
        # fig.add_trace(trace_x)
        # trace_x_resid = go.Scatter(x=data['Date'], y=x_resid, mode='lines', name='x_resid', yaxis='y2',line=dict(color='orange'))

        # fig.add_trace(trace_x_resid)
        # fig.update_layout(title='',
        #               xaxis=dict(title='Date'),
        #               yaxis=dict(title='x', side='left'),
        #               yaxis2=dict(title='x_resid', side='right'))
        # st.title('')
        # st.plotly_chart(fig)

        # x=result.resid
        # x=x.fillna(0)
        x_mean = np.mean(x)
        x_std = np.std(x)
        x_scaled = (x - x_mean) / x_std
        lower_threshold = -2.0 
        upper_threshold = 2.0   
        x_scaled = np.clip(x_scaled, lower_threshold, upper_threshold)
    else:
        x = data[target_column]
        x_mean = np.mean(x)
        x_std = np.std(x)
        x_scaled = (x - x_mean) / x_std
with col[1]:
    if st.checkbox('Attributed'):
        column=[col for col in attributred_app_installs.columns if col in target_column]
        data['app_installs_appsflyer']=attributred_app_installs[column]
        y=data['app_installs_appsflyer']
        title='Attributed-App_installs_appsflyer'
        # st.dataframe(y)
        # st.dataframe(x)
        # st.dataframe(x_scaled)
    else:
        y=data["app_installs_appsflyer"]
        title='App_installs_appsflyer'
        # st.write(len(y))
    # Curve fitting function
def sigmoid(x, K, a, x0):
    return K / (1 + np.exp(-a * (x - x0)))

initial_K = np.max(y)
initial_a = 1
initial_x0 = 0
columns=st.columns(3)


with columns[0]:
    K = st.number_input('K (Amplitude)', min_value=0.01, max_value=2.0 * np.max(y), value=float(initial_K), step=5.0)
with columns[1]:
    a = st.number_input('a (Slope)', min_value=0.01, max_value=5.0, value=float(initial_a), step=0.5)
with columns[2]:
    x0 = st.number_input('x0 (Center)', min_value=float(min(x_scaled)), max_value=float(max(x_scaled)), value=float(initial_x0), step=2.0)
params, _ = curve_fit(sigmoid, x_scaled, y, p0=[K, a, x0], maxfev=20000)


x_slider = st.slider('X Value', min_value=float(min(x)), max_value=float(max(x))+1, value=float(x_mean), step=1.)

# Calculate the corresponding value on the fitted curve
x_slider_scaled = (x_slider - x_mean) / x_std
y_slider_fit = sigmoid(x_slider_scaled, *params)

# Display the corresponding value
st.write(f'{target_column}: {format_numbers(x_slider)}')
st.write(f'Corresponding  App_installs: {format_numbers(y_slider_fit)}')

# Scatter plot of your data
fig = px.scatter(data_frame=data, x=x_scaled, y=y, labels={'x': f'{target_column}', 'y': 'App Installs'}, title=title)

# Add the fitted sigmoid curve to the plot
x_fit = np.linspace(min(x_scaled), max(x_scaled), 100)  # Generate x values for the curve
y_fit = sigmoid(x_fit, *params)
fig.add_trace(px.line(x=x_fit, y=y_fit).data[0])
fig.data[1].update(line=dict(color='orange'))
fig.add_vline(x=x_slider_scaled, line_dash='dash', line_color='red', annotation_text=f'{format_numbers(x_slider)}')

x_tick_labels = {format_axis(x_scaled[i]): format_axis(x[i]) for i in range(len(x_scaled))}
num_points = 30  # Number of points you want to select
keys = list(x_tick_labels.keys())
values = list(x_tick_labels.values())
spacing = len(keys) // num_points  # Calculate the spacing
if spacing==0:
    spacing=15
    selected_keys = keys[::spacing]
    selected_values = values[::spacing]
else:
    selected_keys = keys[::spacing]
    selected_values = values[::spacing]

# Update the x-axis ticks with the selected keys and values
fig.update_xaxes(tickvals=selected_keys, ticktext=selected_values)
fig.update_xaxes(tickvals=list(x_tick_labels.keys()), ticktext=list(x_tick_labels.values()))
# Show the plot using st.plotly_chart

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.update_layout(
    width=600,  # Adjust the width as needed
    height=600  # Adjust the height as needed
)
st.plotly_chart(fig)




st.markdown('Tactic level')
if channel=='paid_social':

    tactic_data=pd.read_excel("Tatcic_paid.xlsx",sheet_name='paid_social_impressions')
else:
    tactic_data=pd.read_excel("Tatcic_paid.xlsx",sheet_name='digital_app_display_impressions')  
target_column = st.selectbox('Select Channel)',[col for col in tactic_data.columns if col!='Date' and col!='app_installs_appsflyer'])
fig=line_plot_target(tactic_data, target=str(target_column), title=f'{str(target_column)} Over Time')
st.plotly_chart(fig, use_container_width=True)

if st.checkbox('Cap Outliers',key='tactic1'):
    x = tactic_data[target_column]
    x_mean = np.mean(x)
    x_std = np.std(x)
    x_scaled = (x - x_mean) / x_std
    lower_threshold = -2.0 
    upper_threshold = 2.0   
    x_scaled = np.clip(x_scaled, lower_threshold, upper_threshold)
else:
    x = tactic_data[target_column]
    x_mean = np.mean(x)
    x_std = np.std(x)
    x_scaled = (x - x_mean) / x_std

if st.checkbox('Attributed',key='tactic2'):
    column=[col for col in attributred_app_installs_tactic.columns if col in target_column]
    tactic_data['app_installs_appsflyer']=attributred_app_installs_tactic[column]
    y=tactic_data['app_installs_appsflyer']
    title='Attributed-App_installs_appsflyer'
    # st.dataframe(y)
    # st.dataframe(x)
    # st.dataframe(x_scaled)
else:
    y=data["app_installs_appsflyer"]
    title='App_installs_appsflyer'
    # st.write(len(y))
# Curve fitting function
def sigmoid(x, K, a, x0):
    return K / (1 + np.exp(-a * (x - x0)))

# Curve fitting
# st.dataframe(x_scaled.head(3))
# # y=y.astype(float)
# st.dataframe(y.head(3))
initial_K = np.max(y)
initial_a = 1
initial_x0 = 0
K = st.number_input('K (Amplitude)', min_value=0.01, max_value=2.0 * np.max(y), value=float(initial_K), step=5.0,key='tactic3')
a = st.number_input('a (Slope)', min_value=0.01, max_value=5.0, value=float(initial_a), step=2.0,key='tactic41')
x0 = st.number_input('x0 (Center)', min_value=float(min(x_scaled)), max_value=float(max(x_scaled)), value=float(initial_x0), step=2.0,key='tactic4')
params, _ = curve_fit(sigmoid, x_scaled, y, p0=[K, a, x0], maxfev=20000)

# Slider to vary x
x_slider = st.slider('X Value', min_value=float(min(x)), max_value=float(max(x)), value=float(x_mean), step=1.,key='tactic7')

# Calculate the corresponding value on the fitted curve
x_slider_scaled = (x_slider - x_mean) / x_std
y_slider_fit = sigmoid(x_slider_scaled, *params)

# Display the corresponding value
st.write(f'{target_column}: {format_axis(x_slider)}')
st.write(f'Corresponding  App_installs: {format_axis(y_slider_fit)}')

# Scatter plot of your data
fig = px.scatter(data_frame=data, x=x_scaled, y=y, labels={'x': f'{target_column}', 'y': 'App Installs'}, title=title)

# Add the fitted sigmoid curve to the plot
x_fit = np.linspace(min(x_scaled), max(x_scaled), 100)  # Generate x values for the curve
y_fit = sigmoid(x_fit, *params)
fig.add_trace(px.line(x=x_fit, y=y_fit).data[0])
fig.data[1].update(line=dict(color='orange'))
fig.add_vline(x=x_slider_scaled, line_dash='dash', line_color='red', annotation_text=f'{format_numbers(x_slider)}')



x_tick_labels = {format_axis((x_scaled[i],0)): format_axis(x[i]) for i in range(len(x_scaled))}
num_points = 50  # Number of points you want to select
keys = list(x_tick_labels.keys())
values = list(x_tick_labels.values())
spacing = len(keys) // num_points  # Calculate the spacing
if spacing==0:
    spacing=2
    selected_keys = keys[::spacing]
    selected_values = values[::spacing]
else:
    selected_keys = keys[::spacing]
    selected_values = values[::spacing]

# Update the x-axis ticks with the selected keys and values
fig.update_xaxes(tickvals=selected_keys, ticktext=selected_values)

# Round the x-axis and y-axis tick values to zero decimal places
fig.update_xaxes(tickformat=".f")  # Format x-axis ticks to zero decimal places
fig.update_yaxes(tickformat=".f")  # Format y-axis ticks to zero decimal places

# Show the plot using st.plotly_chart
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.update_layout(
    width=600,  # Adjust the width as needed
    height=600  # Adjust the height as needed
)
st.plotly_chart(fig)