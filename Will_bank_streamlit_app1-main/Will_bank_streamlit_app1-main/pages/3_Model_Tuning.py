import streamlit as st
import pandas as pd
from Eda_functions import format_numbers
import pickle
from utilities import set_header,load_local_css
import statsmodels.api as sm
import re
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
st.set_option('deprecation.showPyplotGlobalUse', False)
from Data_prep_functions import *

st.set_page_config(
  page_title="Model Tuning",
  page_icon=":shark:",
  layout="wide",
  initial_sidebar_state='collapsed'
)
load_local_css('styles.css')
set_header()


st.title('1. Model Tuning')


if "X_train" not in st.session_state:
   st.error(
"Oops! It seems there are no saved models available. Please build and save a model from the previous page to proceed.")
   st.stop()
X_train=st.session_state['X_train']
X_test=st.session_state['X_test']
y_train=st.session_state['y_train']
y_test=st.session_state['y_test']
df=st.session_state['media_data']

with open("best_models.pkl", 'rb') as file:
  model_dict= pickle.load(file)

if 'selected_model' not in st.session_state:
   st.session_state['selected_model']=0


st.markdown('### 1.1 Event Flags')
st.markdown('Helps in quantifying the impact of specific occurrences of events')   
with st.expander('Apply Event Flags'):
  st.session_state["selected_model"]=st.selectbox('Select Model to apply flags',model_dict.keys())
  model =model_dict[st.session_state["selected_model"]]['Model_object']
  date=st.session_state['date']
  date=pd.to_datetime(date)
  X_train =model_dict[st.session_state["selected_model"]]['X_train']
  features_set= model_dict[st.session_state["selected_model"]]['feature_set']

  col=st.columns(3)  
  min_date=min(date)
  max_date=max(date)
  with col[0]:
    start_date=st.date_input('Select Start Date',min_date,min_value=min_date,max_value=max_date)
  with col[1]:
    end_date=st.date_input('Select End Date',max_date,min_value=min_date,max_value=max_date)
  with col[2]:
    repeat=st.selectbox('Repeat Annually',['Yes','No'],index=1)
  if repeat =='Yes':
      repeat=True
  else: 
      repeat=False
  # X_train=sm.add_constant(X_train)

  if 'Flags' not in st.session_state:
    st.session_state['Flags']={}

  met,line_values,fig_flag=plot_actual_vs_predicted(date[:150], y_train, model.predict(X_train), model,flag=(start_date,end_date),repeat_all_years=repeat)
  st.plotly_chart(fig_flag,use_container_width=True)
  flag_name='f1'
  flag_name=st.text_input('Enter Flag Name')
  if st.button('Update flag'):
    st.session_state['Flags'][flag_name]=line_values
    st.success(f'{flag_name} stored')

  options=list(st.session_state['Flags'].keys())
  selected_options = []
  num_columns = 4
  num_rows = -(-len(options) // num_columns)  


tick=False
if st.checkbox('Select all'):
    tick=True
selected_options = []
for row in range(num_rows):
    cols = st.columns(num_columns)
    for col in cols:
        if options:
            option = options.pop(0) 
            selected = col.checkbox(option,value=tick)
            if selected:
                selected_options.append(option)

st.markdown('### 1.2 Select Parameters to Apply')
parameters=st.columns(3)
with parameters[0]:
   Trend=st.checkbox("**Trend**")
   st.markdown('Helps account for long-term trends or seasonality that could influence advertising effectiveness')
with parameters[1]:
   week_number=st.checkbox('**Week_number**')
   st.markdown('Assists in detecting and incorporating weekly patterns or seasonality')
with parameters[2]:
   sine_cosine=st.checkbox('**Sine and Cosine Waves**')
   st.markdown('Helps in capturing cyclical patterns or seasonality in the data')
if st.button('Build model with Selected Parameters and Flags'):
  st.header('2.1 Results Summary')
  # date=list(df.index)
  # df = df.reset_index(drop=True)
  # st.write(df.head(2))
  # X_train=df[features_set]
  ss = MinMaxScaler()
  X_train_tuned = pd.DataFrame(ss.fit_transform(X_train), columns=X_train.columns)
  X_train_tuned=sm.add_constant(X_train_tuned)
  for flag in selected_options:
    X_train_tuned[flag]=st.session_state['Flags'][flag]
  if Trend:
     X_train_tuned['Trend']=np.arange(1,len(X_train_tuned)+1,1)
  # if week_number:
  #    st.write(date)
     date=pd.to_datetime(date.values)
     X_train_tuned['Week_number']=date.day_of_week[:150]
  model_tuned = sm.OLS(y_train, X_train_tuned).fit()

  metrics_table,line,actual_vs_predicted_plot=plot_actual_vs_predicted(date[:150], y_train, model.predict(X_train), model,target_column='Revenue')
  metrics_table_tuned,line,actual_vs_predicted_plot_tuned=plot_actual_vs_predicted(date[:150], y_train, model_tuned.predict(X_train_tuned), model_tuned,target_column='Revenue')
  
  # st.write(metrics_table)
  mape=np.round(metrics_table.iloc[0,1],2)
  r2=np.round(metrics_table.iloc[1,1],2)
  adjr2=np.round(metrics_table.iloc[2,1],2)
  mape_tuned=np.round(metrics_table_tuned.iloc[0,1],2)
  r2_tuned=np.round(metrics_table_tuned.iloc[1,1],2)
  adjr2_tuned=np.round(metrics_table_tuned.iloc[2,1],2)
  parameters_=st.columns(3)
  with parameters_[0]:
     st.metric('R2',r2_tuned,np.round(r2_tuned-r2,2))
  with parameters_[1]:
     st.metric('Adjusted R2',adjr2_tuned,np.round(adjr2_tuned-adjr2,2))
  with parameters_[2]:
     st.metric('MAPE',mape_tuned,np.round(mape_tuned-mape,2),'inverse')

  st.header('2.2 Actual vs. Predicted Plot')
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

# if st.checkbox('Use this model to build response curves',key='123'):

#   raw_data=df[features_set]
#   columns_raw=[re.split(r"(_lag|_adst)",col)[0] for col in raw_data.columns]
#   raw_data.columns=columns_raw
#   columns_media=[col for col in columns_raw if Categorised_data[col]['BB']=='Media']
#   raw_data=raw_data[columns_media]

#   raw_data['Date']=list(df.index)

#   spends_var=[col for col in df.columns if "spends" in col.lower() and 'adst' not in col.lower() and 'lag' not in col.lower()]
#   spends_df=df[spends_var]
#   spends_df['Week']=list(df.index)
  
 
#   j=0
#   X1=X.copy()
#   col=X1.columns
#   for i in model.params.values:
#       X1[col[j]]=X1.iloc[:,j]*i
#       j+=1
#   contribution_df=X1
#   contribution_df['Date']=list(df.index)
#   excel_file='Overview_data.xlsx'

#   with pd.ExcelWriter(excel_file,engine='xlsxwriter') as writer:
#      raw_data.to_excel(writer,sheet_name='RAW DATA MMM',index=False)
#      spends_df.to_excel(writer,sheet_name='SPEND INPUT',index=False)
#      contribution_df.to_excel(writer,sheet_name='CONTRIBUTION MMM') 
