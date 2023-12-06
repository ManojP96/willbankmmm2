import streamlit as st
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import r2_score
from collections import OrderedDict
import pickle
import json
import streamlit as st
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import r2_score
import pickle
import json
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_percentage_error
import sys
from utilities import (set_header, 
                       initialize_data,
                       load_local_css,
                       create_channel_summary,
                       create_contribution_pie,
                       create_contribuion_stacked_plot,
                       create_channel_spends_sales_plot,
                       format_numbers,
                       channel_name_formating,
                       load_authenticator)




def plot_actual_vs_predicted(date, y, predicted_values, model):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=date, y=y, mode='lines', name='Actual', line=dict(color='#6c757d')))
    fig.add_trace(go.Scatter(x=date, y=predicted_values, mode='lines', name='Predicted', line=dict(color='#FF3A3B')))
    
    # Calculate MAPE
    mape = mean_absolute_percentage_error(y, predicted_values)
    
    # Calculate AdjR2 # Assuming X is your feature matrix
    adjr2 = model.rsquared_adj

    # Create a table to display the metrics
    metrics_table = pd.DataFrame({
        'Metric': ['MAPE', 'R-squared', 'AdjR-squared'],
        'Value': [mape, model.rsquared, adjr2]
    })

    fig.update_layout(
        xaxis=dict(title='Date'),
        yaxis=dict(title='Value'),
        xaxis_tickangle=-30
    )
    #metrics_table.set_index(['Metric'],inplace=True)
    return metrics_table, fig 


X=pd.read_csv('actual_data.csv')
y=X['total_prospect_id']
date=X['date']
X=X.drop(['total_prospect_id','date','Unnamed: 0'],axis=1)

print(X.columns)
original_stdout = sys.stdout
sys.stdout = open('temp_stdout.txt', 'w')

# Perform linear regression
model = sm.OLS(y, X).fit()


sys.stdout.close()
sys.stdout = original_stdout

st.set_page_config(layout='wide')
load_local_css('styles.css')
set_header()
st.title('Analysis of Result')

st.write(model.summary(yname='Prospects'))

st.subheader('Actual vs Predicted Plot')
metrics_table,fig = plot_actual_vs_predicted(date, y, model.predict(X), model)

st.plotly_chart(fig,use_container_width=True)
#st.plotly_chart(fig)

# Display the metrics table

metrics_table=np.round(metrics_table,2)
metrics_table_html = metrics_table.to_html(index=False, escape=False)

# Display the metrics table in Streamlit as HTML
#st.subheader('Model Metrics')
#st.markdown(metrics_table_html, unsafe_allow_html=True)
# st.subheader('Model Metrics')
# st.table(metrics_table)

custom_css = """
<style>
table {
    width: 80%; /* Adjust the table width as needed */
    border-collapse: collapse;
}
th, td {
    padding: 8px;
    text-align: left;
    border-bottom: 1px solid #ddd;
}
</style>
"""

# Display the metrics table in Streamlit as HTML with custom CSS
st.subheader('Model Metrics')
st.markdown(custom_css, unsafe_allow_html=True)
st.markdown(metrics_table_html, unsafe_allow_html=True)
