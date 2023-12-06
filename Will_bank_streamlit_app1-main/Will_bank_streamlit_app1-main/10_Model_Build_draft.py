import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from Eda_functions import format_numbers,line_plot,summary
import numpy as np
from Transformation_functions import check_box
from Transformation_functions import apply_lag,apply_adstock,top_correlated_feature
import pickle
from st_aggrid import AgGrid
from st_aggrid import GridOptionsBuilder,GridUpdateMode
from utilities import set_header,initialize_data,load_local_css
from st_aggrid import GridOptionsBuilder
import time
import itertools
import statsmodels.api as sm
import numpy as np
import re
import itertools
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error  
from PIL import Image
import os
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(
  page_title="Model Build",
  page_icon=":shark:",
  layout="wide",
  initial_sidebar_state='collapsed'
)
load_local_css('styles.css')
set_header()

# logo = Image.open("Full_Logo_Blue.png")

# # Set the logo size
# logo = logo.resize((100, 100))
# st.image(logo)
# st.markdown("""
#     <style>
#     .logo {
#         position: absolute;
#         top: 10px;
#         right: 10px;
#     }
#     </style>
#     """,unsafe_allow_html=True)



# st.image(logo, use_column_width=True, top=0.95, right=0.05)

# Use CSS to position the logo in the top right corner
# st.write(
#     """
#     <style>
#     .logo {
#         position: absolute;
#         top: 10px;
#         right: 10px;
#     }
#     </style>
#     """
# )


st.title('Model Build')
with open("filtered_variables.pkl", 'rb') as file:
    filtered_variables = pickle.load(file)

with open('Categorised_data.pkl', 'rb') as file:
  Categorised_data = pickle.load(file)

with open("target_column.pkl", 'rb') as file:
  target_column= pickle.load(file)

with open("df.pkl", 'rb') as file:
  df= pickle.load(file)

#st.markdown('### Generating all the possible combinations of variables')

if 'final_selection' not in st.session_state:
    st.session_state['final_selection']=None

keywords = ['Digital (Impressions)', 'Streaming (Impressions)']

  # Use list comprehension to filter columns
  #drop_columns = [col for col in df.columns if any(keyword in col for keyword in keywords)]
  #st.write(drop_columns)
  #df.drop(drop_columns,axis=1,inplace=True)
if st.button('Create all Possibile combinations of Variables'):
  with st.spinner('Wait for it'):
    multiple_col=[col for col in filtered_variables.keys() if Categorised_data[col]['VB']=='Holiday']
    #st.write(multiple_col)




    for var in multiple_col:  
      all_combinations_hol = []
      for r in range(1, len(filtered_variables[var]) + 1):
          combinations = itertools.combinations(filtered_variables[var], r)
          all_combinations_hol.extend(combinations)
      all_combinations_hol.append([])
      all_combinations_hol = [list(comb) for comb in all_combinations_hol] 
      filtered_variables[var]=all_combinations_hol


    # st.write(filtered_variables)
    price=[col for col in df.columns if Categorised_data[re.split(r'_adst|_lag', col )[0]]['VB']=='Price']
    price.append("Non Promo Price")
    
    price.append('Promo Price') #tempfix
    
    
    #st.write(price)
    Distribution=[col for col in df.columns if Categorised_data[re.split(r'_adst|_lag', col )[0]]['VB']=='Distribution']
    Promotion=[col for col in df.columns if  Categorised_data[re.split(r'_adst|_lag', col )[0]]['VB']=='Promotion']
    Promotion.remove("Non Promo Price")
    price.append('')
    Distribution.append('')
    
    
    Promotion.remove('Promo Price')  #temp fi------


    filtered_variables['Price']=price
    filtered_variables['Distribution']=Distribution
    filtered_variables['Promotion']=Promotion

    variable_names = list(filtered_variables.keys())
    variable_values = list(filtered_variables.values())

    combinations = list(itertools.product(*variable_values))


    # for combo in combinations:
    #     flattened_combo = [item for sublist in combo for item in (sublist if isinstance(sublist, list) else [sublist])]
    #     print(flattened_combo)
    # st.text(flattened_combo)



    final_selection=[]
    for comb in combinations:
      nested_tuple = comb

      flattened_list = [item for sublist in nested_tuple for item in (sublist if isinstance(sublist, list) else [sublist])]
      final_selection.append(flattened_list)
    #st.write(final_selection[:15])

    st.session_state['final_selection']=final_selection

    st.success('Done')
    st.write(f'Total combinations created {format_numbers(len(final_selection))}')

if 'Model_results' not in st.session_state:
      st.session_state['Model_results']={'Model_object':[],
    'Model_iteration':[],
    'Feature_set':[],
    'MAPE':[],
    'R2':[],
    'ADJR2':[]
    }

#if st.button('Build Model'):
save_path = r"C:\Users\ManojP\Documents\MMM\simopt\Model"
iterations = st.number_input('Select the number of iterations to perform', min_value=1, step=1, value=1)
if st.button("Build Model"):
  
  progress_bar = st.progress(0)  # Initialize the progress bar
  #time_remaining_text = st.empty()  # Create an empty space for time remaining text
  start_time = time.time()  # Record the start time
  progress_text = st.empty()
  #time_elapsed_text = st.empty()

  for i, selected_features in enumerate(st.session_state["final_selection"][:int(iterations)]):
      df = df.reset_index(drop=True)

      fet = [var for var in selected_features if len(var) > 0]
      X = df[fet]
      y = df['Prospects']
      ss = MinMaxScaler()
      X = pd.DataFrame(ss.fit_transform(X), columns=X.columns)
      X = sm.add_constant(X)
      model = sm.OLS(y, X).fit()
      # st.write(fet)
      positive_coeff=[col for col in fet if Categorised_data[re.split(r'_adst|_lag', col )[0]]['VB'] in ["Distribution","Promotion	TV"	,"Display",	"Video"	,"Facebook",	"Twitter"	,"Instagram"	,"Pintrest",	"YouTube"	,"Paid Search"	,"OOH	Radio"	,"Audio Streaming",'Digital']]  
      negetive_coeff=[col for col in fet  if Categorised_data[re.split(r'_adst|_lag', col )[0]]['VB'] in ["Price"]]
      coefficients=model.params.to_dict()
      model_possitive=[col for col in coefficients.keys() if coefficients[col]>0]
      model_negatives=[col for col in coefficients.keys() if coefficients[col]<0]
      # st.write(positive_coeff)
      # st.write(model_possitive)
      pvalues=[var for var in list(model.pvalues) if var<=0.06]
      if (set(positive_coeff).issubset(set(model_possitive))) and (set(negetive_coeff).issubset(model_negatives)) and (len(pvalues)/len(selected_features))>=0.5:


          predicted_values = model.predict(X)
          mape = mean_absolute_percentage_error(y, predicted_values)
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
      progress_text.text(f'Completed iterations: {completed_iterations_text}   Time Elapsed (min): {time_elapsed_minutes:.2f}')

  st.write(f'Out of {iterations} iterations : {len(st.session_state["Model_results"]["Model_object"])} valid models')


def to_percentage(value):
  return f'{value * 100:.1f}%'   

st.title('Analysis of Results')
if st.checkbox('Show Results of Top 10 Models'):
  st.write('Click on the Row to Generate Model Result')
  data=pd.DataFrame(st.session_state['Model_results'])
  data.sort_values(by=['MAPE'],ascending=False,inplace=True)
  top_10=data.head(10)
  top_10['Row_number']=np.arange(1,11,1)
  top_10[['MAPE','R2','ADJR2']]=np.round(top_10[['MAPE','R2','ADJR2']],4).applymap(to_percentage)

  gd=GridOptionsBuilder.from_dataframe(top_10[['Row_number','MAPE','R2','ADJR2','Model_iteration']])
  gd.configure_pagination(enabled=True)
  gd.configure_selection(use_checkbox=True)

  #gd.configure_columns_auto_size_mode(GridOptionsBuilder.configure_columns)
  gridoptions=gd.build()

  table = AgGrid(top_10,gridOptions=gridoptions,update_mode=GridUpdateMode.SELECTION_CHANGED)

  selected_rows=table.selected_rows
  if len(selected_rows)>0:
    st.header('Model Summary')
    #st.text(selected_rows[0]['Model_iteration'])

    model_object=data[data['Model_iteration']==selected_rows[0]['Model_iteration']]['Model_object']
    features_set=data[data['Model_iteration']==selected_rows[0]['Model_iteration']]['Feature_set']
    #st.write(features_set.values)

    with open(str(model_object.values[0]), 'rb') as file:
            model = pickle.load(file)
    st.write(model.summary())        
    # st.write(df.index)


    def plot_actual_vs_predicted(date, y, predicted_values, model):
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=date, y=y, mode='lines', name='Actual', line=dict(color='#08083B')))
        fig.add_trace(go.Scatter(x=date, y=predicted_values, mode='lines', name='Predicted', line=dict(color='#11B6BD')))
        
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
        #metrics_table.set_index(['Metric'],inplace=True)
        return metrics_table, fig 

  # st.text(features_set.values[0])
  # st.dataframe(df[features_set.values[0]])

    date=list(df.index)
    df = df.reset_index(drop=True)
    X=df[features_set.values[0]]
    ss = MinMaxScaler()
    X = pd.DataFrame(ss.fit_transform(X), columns=X.columns)
    X=sm.add_constant(X)
#st.write(model.predict(X))

  #st.write(df[target_column])
    metrics_table,fig=plot_actual_vs_predicted(date, df[target_column], model.predict(X), model)

    st.plotly_chart(fig,use_container_width=True)

    def plot_residual_predicted(actual, predicted, df_):
        df_['Residuals'] = actual - pd.Series(predicted)
        df_['StdResidual'] = (df_['Residuals'] - df_['Residuals'].mean()) / df_['Residuals'].std()
        
        # Create a Plotly scatter plot
        fig = px.scatter(df_, x=predicted, y='StdResidual', opacity=0.5)
        
        # Add horizontal lines
        fig.add_hline(y=0, line_dash="dash", line_color="darkorange")
        fig.add_hline(y=2, line_color="red")
        fig.add_hline(y=-2, line_color="red")
        
        fig.update_xaxes(title='Predicted')
        fig.update_yaxes(title='Standardized Residuals (Actual - Predicted)')
        
        # Set the same width and height for both figures
        fig.update_layout(title='Residuals over Predicted values', autosize=False, width=600, height=400)
        
        return fig

    def residual_distribution(actual, predicted):
        Residuals = actual - pd.Series(predicted)
        
        # Create a Plotly histogram and distribution curve with custom colors
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=Residuals, name='Residuals', histnorm='probability',
                                  marker_color="#11B6BD"))
        fig.add_trace(go.Histogram(x=Residuals, histnorm='probability', showlegend=False,
                                  marker_color="#11B6BD"))
        
        fig.update_layout(title='Distribution of Residuals',title_x=0.5, autosize=False, width=600, height=400)
        
        return fig
    
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
        fig.update_layout(title='QQ Plot of Residuals',title_x=0.5, autosize=False, width=600, height=400,
                          xaxis_title='Theoretical Quantiles', yaxis_title='Sample Quantiles')
        
        return fig

    st.markdown('## Residual Analysis')
    columns=st.columns(2)
    with columns[0]:
      fig=plot_residual_predicted(df[target_column],model.predict(X),df)
      st.plotly_chart(fig)
    
    with columns[1]:
      st.empty()
      fig = qqplot(df[target_column],model.predict(X))
      st.plotly_chart(fig)

    with columns[0]:
      fig=residual_distribution(df[target_column],model.predict(X))
      st.plotly_chart(fig)
    


    vif_data = pd.DataFrame()
    X=X.drop('const',axis=1)
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_data.sort_values(by=['VIF'],ascending=False,inplace=True)
    st.dataframe(vif_data)