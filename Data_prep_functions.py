import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pickle
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score,mean_absolute_percentage_error  
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
st.set_option('deprecation.showPyplotGlobalUse', False)
from datetime import datetime
import seaborn as sns

def calculate_discount(promo_price_series, non_promo_price_series):
    # Calculate the 4-week moving average of non-promo price
    window_size = 4
    base_price = non_promo_price_series.rolling(window=window_size).mean()
    
    # Calculate discount_raw
    discount_raw_series = (1 - promo_price_series / base_price) * 100
    
    # Calculate discount_final
    discount_final_series = discount_raw_series.where(discount_raw_series >= 5, 0)
    
    return base_price, discount_raw_series, discount_final_series


def create_dual_axis_line_chart(date_series, promo_price_series, non_promo_price_series, base_price_series, discount_series):
    # Create traces for the primary axis (price vars)
    trace1 = go.Scatter(
        x=date_series,
        y=promo_price_series,
        name='Promo Price',
        yaxis='y1'
    )
    
    trace2 = go.Scatter(
        x=date_series,
        y=non_promo_price_series,
        name='Non-Promo Price',
        yaxis='y1'
    )

    trace3 = go.Scatter(
        x=date_series,
        y=base_price_series,
        name='Base Price',
        yaxis='y1'
    )
    
    # Create a trace for the secondary axis (discount)
    trace4 = go.Scatter(
        x=date_series,
        y=discount_series,
        name='Discount',
        yaxis='y2'
    )

    # Create the layout with dual axes
    layout = go.Layout(
        title='Price and Discount Over Time',
        yaxis=dict(
            title='Price',
            side='left'
        ),
        yaxis2=dict(
            title='Discount',
            side='right',
            overlaying='y',
            showgrid=False
        ),
        xaxis=dict(title='Date'),
    )

    # Create the figure with the defined traces and layout
    fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)

    return fig


def to_percentage(value):
  return f'{value * 100:.1f}%'   

def plot_actual_vs_predicted(date, y, predicted_values, model,target_column=None, flag=None, repeat_all_years=False):
      fig = go.Figure()

      fig.add_trace(go.Scatter(x=date, y=y, mode='lines', name='Actual', line=dict(color='#08083B')))
      fig.add_trace(go.Scatter(x=date, y=predicted_values, mode='lines', name='Predicted', line=dict(color='#11B6BD')))
      line_values=[]
      if flag:
          min_date, max_date = flag[0], flag[1]
          min_week = datetime.strptime(str(min_date), "%Y-%m-%d").strftime("%U")
          max_week = datetime.strptime(str(max_date), "%Y-%m-%d").strftime("%U")
          month=pd.to_datetime(min_date).month
          day=pd.to_datetime(min_date).day
          #st.write(pd.to_datetime(min_date).week)
          #st.write(min_week) 
          # Initialize an empty list to store line values
           
          if repeat_all_years:
            #line_values=list(pd.to_datetime((pd.Series(date)).dt.week).map(lambda x: 10000 if x==min_week else 0  ))
            #st.write(pd.Series(date).map(lambda x: pd.Timestamp(x).week))
            line_values=list(pd.Series(date).map(lambda x: 5000000 if (pd.Timestamp(x).week >=int(min_week)) & (pd.Timestamp(x).week <=int(max_week)) else 0))
            
            #st.write(line_values)
            fig.add_trace(go.Scatter(x=date, y=line_values, mode='lines', name='Flag', line=dict(color='#FF5733')))
          else:
            line_values = []
            line_values = list(pd.Series(date).map(lambda x: 5000000 if (pd.Timestamp(x) >= pd.Timestamp(min_date)) and (pd.Timestamp(x) <= pd.Timestamp(max_date)) else 0))

            #st.write(line_values)
            fig.add_trace(go.Scatter(x=date, y=line_values, mode='lines', name='Flag', line=dict(color='#FF5733')))             
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
          yaxis=dict(title=target_column),
          xaxis_tickangle=-30
      )
      fig.add_annotation(
      text=f"MAPE: {mape*100:0.1f}%,  Adjr2: {adjr2 *100:.1f}%",
      xref="paper",
      yref="paper",
      x=0.95,  # Adjust these values to position the annotation
      y=1.2,
      showarrow=False,
      )

      #metrics_table.set_index(['Metric'],inplace=True)
      return metrics_table,line_values, fig

def plot_residual_predicted(actual, predicted, df):
        df_=df.copy()
        df_['Residuals'] = actual - pd.Series(predicted)
        df_['StdResidual'] = (df_['Residuals'] - df_['Residuals'].mean()) / df_['Residuals'].std()
        
        # Create a Plotly scatter plot
        fig = px.scatter(df_, x=predicted, y='StdResidual', opacity=0.5,color_discrete_sequence=["#11B6BD"])
        
        # Add horizontal lines
        fig.add_hline(y=0, line_dash="dash", line_color="darkorange")
        fig.add_hline(y=2, line_color="red")
        fig.add_hline(y=-2, line_color="red")
        
        fig.update_xaxes(title='Predicted')
        fig.update_yaxes(title='Standardized Residuals (Actual - Predicted)')
        
        # Set the same width and height for both figures
        fig.update_layout(title='2.3.1 Residuals over Predicted Values', autosize=False, width=600, height=400)
        
        return fig

def residual_distribution(actual, predicted):
        Residuals = actual - pd.Series(predicted)
        
        # Create a Seaborn distribution plot
        sns.set(style="whitegrid")
        plt.figure(figsize=(6, 4))
        sns.histplot(Residuals, kde=True, color="#11B6BD")
        
        plt.title('2.3.3 Distribution of Residuals')
        plt.xlabel('Residuals')
        plt.ylabel('Probability Density')
        
        return plt


def qqplot(actual, predicted):
        Residuals = actual - pd.Series(predicted)
        Residuals = pd.Series(Residuals)
        Resud_std = (Residuals - Residuals.mean()) / Residuals.std()
        
        # Create a QQ plot using Plotly with custom colors
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sm.ProbPlot(Resud_std).theoretical_quantiles,
                                y=sm.ProbPlot(Resud_std).sample_quantiles,
                                mode='markers',
                                marker=dict(size=5, color="#11B6BD"),
                                name='QQ Plot'))
        
        # Add the 45-degree reference line
        diagonal_line = go.Scatter(
            x=[-2, 2],  # Adjust the x values as needed to fit the range of your data
            y=[-2, 2],  # Adjust the y values accordingly
            mode='lines',
            line=dict(color='red'),  # Customize the line color and style
            name=' '
        )
        fig.add_trace(diagonal_line)
        
        # Customize the layout
        fig.update_layout(title='2.3.2 QQ Plot of Residuals',title_x=0.5, autosize=False, width=600, height=400,
                          xaxis_title='Theoretical Quantiles', yaxis_title='Sample Quantiles')
        
        return fig
