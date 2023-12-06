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
import seaborn as sns
import matplotlib.pyplot as plt
import sweetviz as sv
import tempfile

original_stdout = sys.stdout
sys.stdout = open('temp_stdout.txt', 'w')
sys.stdout.close()
sys.stdout = original_stdout

st.set_page_config(layout='wide')
load_local_css('styles.css')
set_header()

for k, v in st.session_state.items():
    if k not in ['logout', 'login','config'] and not k.startswith('FormSubmitter'):
        st.session_state[k] = v

authenticator = st.session_state.get('authenticator')
if authenticator is None:
    authenticator = load_authenticator()

name, authentication_status, username = authenticator.login('Login', 'main')
auth_status = st.session_state.get('authentication_status')

if auth_status == True:
    is_state_initiaized = st.session_state.get('initialized',False)
    if not is_state_initiaized:
        a=1
      

    def plot_residual_predicted(actual, predicted, df_):
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
            fig.update_layout(title='Residuals over Predicted Values', autosize=False, width=600, height=400)
            
            return fig

    def residual_distribution(actual, predicted):
            Residuals = actual - pd.Series(predicted)
            
            # Create a Seaborn distribution plot
            sns.set(style="whitegrid")
            plt.figure(figsize=(6, 4))
            sns.histplot(Residuals, kde=True, color="#11B6BD")
            
            plt.title(' Distribution of Residuals')
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
            fig.update_layout(title='QQ Plot of Residuals',title_x=0.5, autosize=False, width=600, height=400,
                            xaxis_title='Theoretical Quantiles', yaxis_title='Sample Quantiles')
            
            return fig


    def plot_actual_vs_predicted(date, y, predicted_values, model):
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=date, y=y, mode='lines', name='Actual', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=date, y=predicted_values, mode='lines', name='Predicted', line=dict(color='orange')))
        
        # Calculate MAPE
        mape = mean_absolute_percentage_error(y, predicted_values)*100
        
        # Calculate R-squared
        rss = np.sum((y - predicted_values) ** 2)
        tss = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (rss / tss)
        
        # Get the number of predictors
        num_predictors = model.df_model
        
        # Get the number of samples
        num_samples = len(y)
        
        # Calculate Adjusted R-squared
        adj_r_squared = 1 - ((1 - r_squared) * ((num_samples - 1) / (num_samples - num_predictors - 1)))
        metrics_table = pd.DataFrame({
        'Metric': ['MAPE', 'R-squared', 'AdjR-squared'],
        'Value': [mape, r_squared, adj_r_squared]})
        fig.update_layout(
            xaxis=dict(title='Date'),
            yaxis=dict(title='Value'),
            title=f'MAPE : {mape:.2f}%, AdjR2: {adj_r_squared:.2f}',
            xaxis_tickangle=-30
        )

        return metrics_table,fig




    # # Perform linear regression
    # model = sm.OLS(y, X).fit()
    eda_columns=st.columns(3)
    with eda_columns[0]:
        tactic=st.checkbox('Tactic Level Model')
    if tactic:
        with open('mastercard_mmm_model.pkl', 'rb') as file:
            model = pickle.load(file)
        train=pd.read_csv('train_mastercard.csv')
        test=pd.read_csv('test_mastercard.csv')
        train['Date']=pd.to_datetime(train['Date'])
        test['Date']=pd.to_datetime(test['Date'])
        train.set_index('Date',inplace=True)
        test.set_index('Date',inplace=True)
        test.dropna(inplace=True)
        X_train=train.drop(["total_approved_accounts_revenue"],axis=1)
        y_train=train['total_approved_accounts_revenue']
        X_test=test.drop(["total_approved_accounts_revenue"],axis=1)
        X_train=sm.add_constant(X_train)
        X_test=sm.add_constant(X_test)
        y_test=test['total_approved_accounts_revenue']

        # sys.stdout.close()
        # sys.stdout = original_stdout

        # st.set_page_config(layout='wide')
        # load_local_css('styles.css')
        # set_header()

        channel_data=pd.read_excel("Channel_wise_imp_click_spends_new.xlsx",sheet_name='Sheet3')
        target_column='Total Approved Accounts - Revenue'

  
        with eda_columns[1]:
          if st.button('Generate EDA Report'):
              def generate_report_with_target(channel_data, target_feature):
                  report = sv.analyze([channel_data, "Dataset"], target_feat=target_feature,verbose=False)
                  temp_dir = tempfile.mkdtemp()
                  report_path = os.path.join(temp_dir, "report.html")
                  report.show_html(filepath=report_path, open_browser=False)  # Generate the report as an HTML file
                  return report_path
          
              report_file = generate_report_with_target(channel_data, target_column)
          
              if os.path.exists(report_file):
                  with open(report_file, 'rb') as f:
                      st.download_button(
                          label="Download EDA Report",
                          data=f.read(),
                          file_name="report.html",
                          mime="text/html"
                      )
              else:
                  st.warning("Report generation failed. Unable to find the report file.")


        st.title('Analysis of Result')

        st.write(model.summary(yname='Revenue'))

        metrics_table_train,fig_train= plot_actual_vs_predicted(X_train.index, y_train, model.predict(X_train), model)
        metrics_table_test,fig_test= plot_actual_vs_predicted(X_test.index, y_test, model.predict(X_test), model)

        metrics_table_train=metrics_table_train.set_index('Metric').transpose()
        metrics_table_train.index=['Train']
        metrics_table_test=metrics_table_test.set_index('Metric').transpose()
        metrics_table_test.index=['test']
        metrics_table=np.round(pd.concat([metrics_table_train,metrics_table_test]),2)

        st.markdown('Result Overview')
        st.dataframe(np.round(metrics_table,2),use_container_width=True)

        st.subheader('Actual vs Predicted Plot Train')

        st.plotly_chart(fig_train,use_container_width=True)
        st.subheader('Actual vs Predicted Plot Test')
        st.plotly_chart(fig_test,use_container_width=True)

        st.markdown('## Residual Analysis')
        columns=st.columns(2)
        Xtrain1=X_train.copy()
        with columns[0]:
            fig=plot_residual_predicted(y_train,model.predict(Xtrain1),Xtrain1)
            st.plotly_chart(fig)

        with columns[1]:
            st.empty()
            fig = qqplot(y_train,model.predict(X_train))
            st.plotly_chart(fig)

        with columns[0]:
            fig=residual_distribution(y_train,model.predict(X_train))
            st.pyplot(fig)
    else:
        with open('mastercard_mmm_model_channel.pkl', 'rb') as file:
            model = pickle.load(file)
        train=pd.read_csv('train_mastercard_channel.csv')
        test=pd.read_csv('test_mastercard_channel.csv')
        # train['Date']=pd.to_datetime(train['Date'])
        # test['Date']=pd.to_datetime(test['Date'])
        # train.set_index('Date',inplace=True)
        # test.set_index('Date',inplace=True)
        test.dropna(inplace=True)
        X_train=train.drop(["total_approved_accounts_revenue"],axis=1)
        y_train=train['total_approved_accounts_revenue']
        X_test=test.drop(["total_approved_accounts_revenue"],axis=1)
        X_train=sm.add_constant(X_train)
        X_test=sm.add_constant(X_test)
        y_test=test['total_approved_accounts_revenue']

    

        channel_data=pd.read_excel("Channel_wise_imp_click_spends_new.xlsx",sheet_name='Sheet3')
        target_column='Total Approved Accounts - Revenue'
        with eda_columns[1]:
          if st.button('Generate EDA Report'):
              def generate_report_with_target(channel_data, target_feature):
                  report = sv.analyze([channel_data, "Dataset"], target_feat=target_feature)
                  temp_dir = tempfile.mkdtemp()
                  report_path = os.path.join(temp_dir, "report.html")
                  report.show_html(filepath=report_path, open_browser=False)  # Generate the report as an HTML file
                  return report_path
          
              report_file = generate_report_with_target(channel_data, target_column)
          
              # Provide a link to download the generated report
              with open(report_file, 'rb') as f:
                  st.download_button(
                      label="Download EDA Report",
                      data=f.read(),
                      file_name="report.html",
                      mime="text/html"
                  )


        st.title('Analysis of Result')

        st.write(model.summary(yname='Revenue'))

        metrics_table_train,fig_train= plot_actual_vs_predicted(X_train.index, y_train, model.predict(X_train), model)
        metrics_table_test,fig_test= plot_actual_vs_predicted(X_test.index, y_test, model.predict(X_test), model)

        metrics_table_train=metrics_table_train.set_index('Metric').transpose()
        metrics_table_train.index=['Train']
        metrics_table_test=metrics_table_test.set_index('Metric').transpose()
        metrics_table_test.index=['test']
        metrics_table=np.round(pd.concat([metrics_table_train,metrics_table_test]),2)

        st.markdown('Result Overview')
        st.dataframe(np.round(metrics_table,2),use_container_width=True)

        st.subheader('Actual vs Predicted Plot Train')

        st.plotly_chart(fig_train,use_container_width=True)
        st.subheader('Actual vs Predicted Plot Test')
        st.plotly_chart(fig_test,use_container_width=True)

        st.markdown('## Residual Analysis')
        columns=st.columns(2)
        Xtrain1=X_train.copy()
        with columns[0]:
            fig=plot_residual_predicted(y_train,model.predict(Xtrain1),Xtrain1)
            st.plotly_chart(fig)

        with columns[1]:
            st.empty()
            fig = qqplot(y_train,model.predict(X_train))
            st.plotly_chart(fig)

        with columns[0]:
            fig=residual_distribution(y_train,model.predict(X_train))
            st.pyplot(fig)

elif auth_status == False:
    st.error('Username/Password is incorrect')
    
if auth_status != True:
    try:
        username_forgot_pw, email_forgot_password, random_password = authenticator.forgot_password('Forgot password')
        if username_forgot_pw:
            st.success('New password sent securely')
            # Random password to be transferred to user securely
        elif username_forgot_pw == False:
            st.error('Username not found')
    except Exception as e:
        st.error(e)
