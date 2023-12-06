import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from Eda_functions import format_numbers
import numpy as np
import pickle
from st_aggrid import AgGrid
from st_aggrid import GridOptionsBuilder,GridUpdateMode
from utilities import set_header,load_local_css
from st_aggrid import GridOptionsBuilder
import time
import itertools
import statsmodels.api as sm
import numpy as np
import re
import itertools
from sklearn.metrics import mean_absolute_error, r2_score,mean_absolute_percentage_error  
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
st.set_option('deprecation.showPyplotGlobalUse', False)
from datetime import datetime
import seaborn as sns
from Data_prep_functions import *

st.set_page_config(
  page_title="Model Build",
  page_icon=":shark:",
  layout="wide",
  initial_sidebar_state='collapsed'
)

load_local_css('styles.css')
set_header()


st.title('1. Build Your Model')

media_data=pd.read_csv('Media_data_for_model.csv')
date=media_data['Date']
st.session_state['date']=date
revenue=media_data['Total Approved Accounts - Revenue']
media_data.drop(['Total Approved Accounts - Revenue'],axis=1,inplace=True)
media_data.drop(['Date'],axis=1,inplace=True)
media_data.reset_index(drop=True,inplace=True)
columns = st.columns(2)
old_shape=media_data.shape

if "old_shape" not in st.session_state:
   st.session_state['old_shape']=old_shape

with columns[0]:
  slider_value_adstock  = st.slider('Select Adstock Range (only applied to media)', 0.0, 1.0, (0.2, 0.4), step=0.1, format="%.2f")
with columns[1]:
  slider_value_lag = st.slider('Select Lag Range (applied to media, seasonal, macroeconomic variables)', 1, 7, (1, 3), step=1)

def lag(X, features, min_lag=0,max_lag=6):
    for i in features:
        for lag in range(min_lag, max_lag + 1):
            X[f'{i}_lag{lag}'] = X[i].shift(periods=lag)
    return X.fillna(method='bfill')

def adstock_variable(X,variable_name,decay):
    adstock = [0] * len(X[variable_name])  

    for t in range(len(X[variable_name])):
        if t == 0:
            adstock[t] = X[variable_name][t] 
        else:
            adstock[t] = X[variable_name][t] + adstock[t-1] * decay
    return adstock

start=slider_value_adstock[0]
end=slider_value_adstock[1]
if 'media_data' not in st.session_state:
   
  st.session_state['media_data']=pd.DataFrame()

with columns[0]:
  if st.button('Apply Transformations'):

    st.spinner('Applying Transformations')
    
    media_data.reset_index(drop=True,inplace=True)
    media_data=lag(media_data,media_data.columns,min_lag=slider_value_lag[0],max_lag=slider_value_lag[1])
    for i in media_data.columns: 
        for j in np.arange(start,end+0.1,0.1):#adding adstock
          media_data[f'{i}_adst.{np.round(j,2)}']=adstock_variable(media_data,i,j)
    st.session_state['media_data']=media_data

    with st.spinner('Applying Transformations'):
      time.sleep(2)
      st.success("Transformations complete!")

if st.session_state['media_data'].shape[1]>old_shape[1]:
  with columns[0]:
    st.write(f'Total no.of variables before transformation: {old_shape[1]}, Total no.of variables after transformation: {st.session_state["media_data"].shape[1]}')
  #st.write(f'Total no.of variables after transformation: {st.session_state["media_data"].shape[1]}')


bucket=['paid_search', 'kwai','indicacao','infleux', 'influencer','FB: Level Achieved - Tier 1 Impressions',
      ' FB: Level Achieved - Tier 2 Impressions','paid_social_others',
        ' GA App: Will And Cid Pequena Baixo Risco Clicks',
      'digital_tactic_others',"programmatic"
      ]
   
with columns[1]:
  if st.button('Create Combinations of Variables'):

    top_3_correlated_features=[]
    for col in st.session_state['media_data'].columns[:19]:
        corr_df=pd.concat([st.session_state['media_data'].filter(regex=col),
                  revenue],axis=1).corr()['Total Approved Accounts - Revenue'].iloc[:-1]
        top_3_correlated_features.append(list(corr_df.sort_values(ascending=False).head(2).index))
    flattened_list = [item for sublist in top_3_correlated_features for item in sublist]
    all_features_set={var:[col for col in flattened_list if var in col] for var in bucket}
    channels_all=[values for values in all_features_set.values()]
    st.session_state['combinations'] = list(itertools.product(*channels_all))

  # if 'combinations' not in st.session_state:  
  #   st.session_state['combinations']=combinations_all

    st.session_state['final_selection']=st.session_state['combinations']



revenue.reset_index(drop=True,inplace=True)
if 'Model_results' not in st.session_state:
      st.session_state['Model_results']={'Model_object':[],
    'Model_iteration':[],
    'Feature_set':[],
    'MAPE':[],
    'R2':[],
    'ADJR2':[]
    }

#if st.button('Build Model'):
if 'iterations' not in st.session_state:
   st.session_state['iterations']=1
save_path = r"Model"
with columns[1]:
  if "final_selection" in st.session_state:
    st.write(f'Total combinations created {format_numbers(len(st.session_state["final_selection"]))}')
    
    st.success('Done')
if st.checkbox('Build all iterations'):
   iterations=len(st.session_state['final_selection'])
else:
   iterations = st.number_input('Select the number of iterations to perform', min_value=1, step=100, value=st.session_state['iterations'])  

st.session_state['iterations']=iterations


st.session_state['media_data']=st.session_state['media_data'].fillna(method='ffill')
if st.button("Build Models"):  
  st.markdown('Data Split -- Training Period: May 9th, 2023 - October 5th,2023 , Testing Period: October 6th, 2023 - November 7th, 2023 ')
  progress_bar = st.progress(0)  # Initialize the progress bar
  #time_remaining_text = st.empty()  # Create an empty space for time remaining text
  start_time = time.time()  # Record the start time
  progress_text = st.empty()
  #time_elapsed_text = st.empty()
  for i, selected_features in enumerate(st.session_state["final_selection"][40000:40000+int(iterations)]):
      df = st.session_state['media_data']

      fet = [var for var in selected_features if len(var) > 0]
      X = df[fet]
      y = revenue
      ss = MinMaxScaler()
      X = pd.DataFrame(ss.fit_transform(X), columns=X.columns)
      X = sm.add_constant(X)
      X_train=X.iloc[:150]
      X_test=X.iloc[150:]
      y_train=y.iloc[:150]
      y_test=y.iloc[150:]


      model = sm.OLS(y_train, X_train).fit()
      # st.write(fet)
      positive_coeff=X.columns
      negetive_coeff=[]
      coefficients=model.params.to_dict()
      model_possitive=[col for col in coefficients.keys() if coefficients[col]>0]
      # st.write(positive_coeff)
      # st.write(model_possitive)
      pvalues=[var for var in list(model.pvalues) if var<=0.06]
      if (len(model_possitive)/len(selected_features))>0.9 and (len(pvalues)/len(selected_features))>=0.8:


          predicted_values = model.predict(X_train)
          mape = mean_absolute_percentage_error(y_train, predicted_values)
          adjr2 = model.rsquared_adj
          r2 = model.rsquared
          filename = os.path.join(save_path, f"model_{i}.pkl")
          with open(filename, "wb") as f:
            pickle.dump(model, f)
          # with open(r"C:\Users\ManojP\Documents\MMM\simopt\Model\model.pkl", 'rb') as file:
          #   model = pickle.load(file)

          st.session_state['Model_results']['Model_object'].append(filename)
          st.session_state['Model_results']['Model_iteration'].append(i)
          st.session_state['Model_results']['Feature_set'].append(fet)
          st.session_state['Model_results']['MAPE'].append(mape)
          st.session_state['Model_results']['R2'].append(r2)
          st.session_state['Model_results']['ADJR2'].append(adjr2)

      current_time = time.time()
      time_taken = current_time - start_time
      time_elapsed_minutes = time_taken / 60
      completed_iterations_text = f"{i + 1}/{iterations}"
      progress_bar.progress((i + 1) / int(iterations))
      progress_text.text(f'Completed iterations: {completed_iterations_text},Time Elapsed (min): {time_elapsed_minutes:.2f}')
  
st.write(f'Out of {st.session_state["iterations"]} iterations : {len(st.session_state["Model_results"]["Model_object"])} valid models')


def to_percentage(value):
  return f'{value * 100:.1f}%'   

st.title('2. Select Models')
if 'tick' not in st.session_state:
   st.session_state['tick']=False
if st.checkbox('Show results of top 10 models (based on MAPE and Adj. R2)',value=st.session_state['tick']):
  st.session_state['tick']=True
  st.write('Select one model iteration to generate performance metrics for it:')
  data=pd.DataFrame(st.session_state['Model_results'])
  data.sort_values(by=['MAPE'],ascending=False,inplace=True)
  data.drop_duplicates(subset='Model_iteration',inplace=True)
  top_10=data.head(10)
  top_10['Rank']=np.arange(1,len(top_10)+1,1)
  top_10[['MAPE','R2','ADJR2']]=np.round(top_10[['MAPE','R2','ADJR2']],4).applymap(to_percentage)
  top_10_table = top_10[['Rank','Model_iteration','MAPE','ADJR2','R2']]
  #top_10_table.columns=[['Rank','Model Iteration Index','MAPE','Adjusted R2','R2']]
  gd=GridOptionsBuilder.from_dataframe(top_10_table)
  gd.configure_pagination(enabled=True)
  gd.configure_selection(use_checkbox=True)

  
  gridoptions=gd.build()

  table = AgGrid(top_10,gridOptions=gridoptions,update_mode=GridUpdateMode.SELECTION_CHANGED)
  
  selected_rows=table.selected_rows
  # if st.session_state["selected_rows"] != selected_rows:
  #   st.session_state["build_rc_cb"] = False
  st.session_state["selected_rows"] = selected_rows
  if 'Model' not in st.session_state:
    st.session_state['Model']={}

  if len(selected_rows)>0:
    st.header('2.1 Results Summary')

    model_object=data[data['Model_iteration']==selected_rows[0]['Model_iteration']]['Model_object']
    features_set=data[data['Model_iteration']==selected_rows[0]['Model_iteration']]['Feature_set']
    
    with open(str(model_object.values[0]), 'rb') as file:
            model = pickle.load(file)
    st.write(model.summary())        
    st.header('2.2 Actual vs. Predicted Plot')
  
    df=st.session_state['media_data']
    X=df[features_set.values[0]]
    X = sm.add_constant(X)
    y=revenue
    X_train=X.iloc[:150]
    X_test=X.iloc[150:]
    y_train=y.iloc[:150]
    y_test=y.iloc[150:]
    ss = MinMaxScaler()
    X_train = pd.DataFrame(ss.fit_transform(X_train), columns=X_train.columns)
    st.session_state['X']=X_train
    st.session_state['features_set']=features_set.values[0]

    metrics_table,line,actual_vs_predicted_plot=plot_actual_vs_predicted(date, y_train, model.predict(X_train), model,target_column='Revenue')

    st.plotly_chart(actual_vs_predicted_plot,use_container_width=True)


     
    st.markdown('## 2.3 Residual Analysis')
    columns=st.columns(2)
    with columns[0]:
      fig=plot_residual_predicted(y_train,model.predict(X_train),X_train)
      st.plotly_chart(fig)
    
    with columns[1]:
      st.empty()
      fig = qqplot(y_train,model.predict(X_train))
      st.plotly_chart(fig)

    with columns[0]:
      fig=residual_distribution(y_train,model.predict(X_train))
      st.pyplot(fig)
    


    vif_data = pd.DataFrame()
    # X=X.drop('const',axis=1)
    vif_data["Variable"] = X_train.columns
    vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
    vif_data.sort_values(by=['VIF'],ascending=False,inplace=True)
    vif_data=np.round(vif_data)
    vif_data['VIF']=vif_data['VIF'].astype(float)
    st.header('2.4 Variance Inflation Factor (VIF)')
    #st.dataframe(vif_data)
    color_mapping = {
    'darkgreen': (vif_data['VIF'] < 3),
    'orange': (vif_data['VIF'] >= 3) & (vif_data['VIF'] <= 10),
    'darkred': (vif_data['VIF'] > 10)
    }

# Create a horizontal bar plot
    fig, ax = plt.subplots()
    fig.set_figwidth(10)  # Adjust the width of the figure as needed

    # Sort the bars by descending VIF values
    vif_data = vif_data.sort_values(by='VIF', ascending=False)

    # Iterate through the color mapping and plot bars with corresponding colors
    for color, condition in color_mapping.items():
        subset = vif_data[condition]
        bars = ax.barh(subset["Variable"], subset["VIF"], color=color, label=color)
        
        # Add text annotations on top of the bars
        for bar in bars:
            width = bar.get_width()
            ax.annotate(f'{width:}', xy=(width, bar.get_y() + bar.get_height() / 2), xytext=(5, 0),
                        textcoords='offset points', va='center')

    # Customize the plot
    ax.set_xlabel('VIF Values')
    #ax.set_title('2.4 Variance Inflation Factor (VIF)')
    #ax.legend(loc='upper right')

    # Display the plot in Streamlit
    st.pyplot(fig)

    with st.expander('Results Summary Test data'):
      ss = MinMaxScaler()
      X_test = pd.DataFrame(ss.fit_transform(X_test), columns=X_test.columns)
      st.header('2.2 Actual vs. Predicted Plot')

      metrics_table,line,actual_vs_predicted_plot=plot_actual_vs_predicted(date, y_test, model.predict(X_test), model,target_column='Revenue')

      st.plotly_chart(actual_vs_predicted_plot,use_container_width=True)

      st.markdown('## 2.3 Residual Analysis')
      columns=st.columns(2)
      with columns[0]:
        fig=plot_residual_predicted(revenue,model.predict(X_test),X_test)
        st.plotly_chart(fig)
      
      with columns[1]:
        st.empty()
        fig = qqplot(revenue,model.predict(X_test))
        st.plotly_chart(fig)

      with columns[0]:
        fig=residual_distribution(revenue,model.predict(X_test))
        st.pyplot(fig)
        
    value=False
    if st.checkbox('Save this model to tune',key='build_rc_cb'):
      mod_name=st.text_input('Enter model name')
      if len(mod_name)>0:
        st.session_state['Model'][mod_name]={"Model_object":model,'feature_set':st.session_state['features_set'],'X_train':X_train}
        st.session_state['X_train']=X_train
        st.session_state['X_test']=X_test
        st.session_state['y_train']=y_train
        st.session_state['y_test']=y_test
        with open("best_models.pkl", "wb") as f:
          pickle.dump(st.session_state['Model'], f)  
          st.success('Model saved!, Proceed  next page to tune the model')
        value=False
