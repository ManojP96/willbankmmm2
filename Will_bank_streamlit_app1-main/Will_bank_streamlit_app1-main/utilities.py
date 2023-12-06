from numerize.numerize import numerize
import streamlit as st
import pandas as pd
import json
from classes import Channel, Scenario
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from classes import class_to_dict
from collections import OrderedDict
import io
import plotly
from pathlib import Path
import pickle
import streamlit_authenticator as stauth
import yaml
from yaml import SafeLoader
from streamlit.components.v1 import html
import smtplib
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from classes import class_from_dict
import os
import base64




color_palette = ['#003059','#00EDED', '#A2F3F3',"#304550","#F3F3F0"]

CURRENCY_INDICATOR = '$'

def load_authenticator():
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
        st.session_state['config'] = config
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )
    st.session_state['authenticator'] = authenticator
    return authenticator

def nav_page(page_name, timeout_secs=3):
    nav_script = """
        <script type="text/javascript">
            function attempt_nav_page(page_name, start_time, timeout_secs) {
                var links = window.parent.document.getElementsByTagName("a");
                for (var i = 0; i < links.length; i++) {
                    if (links[i].href.toLowerCase().endsWith("/" + page_name.toLowerCase())) {
                        links[i].click();
                        return;
                    }
                }
                var elasped = new Date() - start_time;
                if (elasped < timeout_secs * 1000) {
                    setTimeout(attempt_nav_page, 100, page_name, start_time, timeout_secs);
                } else {
                    alert("Unable to navigate to page '" + page_name + "' after " + timeout_secs + " second(s).");
                }
            }
            window.addEventListener("load", function() {
                attempt_nav_page("%s", new Date(), %d);
            });
        </script>
    """ % (page_name, timeout_secs)
    html(nav_script)


# def load_local_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# def set_header():
#     return st.markdown(f"""<div class='main-header'>
#                     <h1>MMM LiME</h1>
#                     <img src="https://assets-global.website-files.com/64c8fffb0e95cbc525815b79/64df84637f83a891c1473c51_Vector%20(Stroke).svg   ">
#             </div>""", unsafe_allow_html=True)

path = os.path.dirname(__file__)

file_ = open(f"{path}/mastercard_logo.png", "rb")

contents = file_.read()

data_url = base64.b64encode(contents).decode("utf-8")

file_.close()

 

DATA_PATH = './data'

IMAGES_PATH = './data/images_224_224'

 

def load_local_css(file_name):

    with open(file_name) as f:

        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

 

 

# def set_header():

#     return st.markdown(f"""<div class='main-header'>

#                     <h1>H & M Recommendations</h1>

#                     <img src="data:image;base64,{data_url}", alt="Logo">

#             </div>""", unsafe_allow_html=True)
path1 = os.path.dirname(__file__)

file_1 = open(f"{path}/willbank.png", "rb")

contents1 = file_1.read()

data_url1 = base64.b64encode(contents1).decode("utf-8")

file_1.close()

 

DATA_PATH1 = './data'

IMAGES_PATH1 = './data/images_224_224'





def set_header():
    return st.markdown(f"""<div class='main-header'>
                    <!-- <h1></h1> -->
                       <div >
                       <img class='blend-logo' src="data:image;base64,{data_url1}", alt="Logo">
                       </div>
                    <img class='blend-logo' src="data:image;base64,{data_url}", alt="Logo">
            </div>""", unsafe_allow_html=True)

# def set_header():
#     logo_path = "./path/to/your/local/LIME_logo.png"  # Replace with the actual file path
#     text = "LiME"
#     return st.markdown(f"""<div class='main-header'>
#                     <img src="data:image/png;base64,{data_url}" alt="Logo" style="float: left; margin-right: 10px; width: 100px; height: auto;">
#                     <h1>{text}</h1>
#             </div>""", unsafe_allow_html=True)


def s_curve(x,K,b,a,x0):
    return K / (1 + b * np.exp(-a*(x-x0)))

def initialize_data():
    # uopx_conv_rates = {'streaming_impressions' : 0.007,'digital_impressions' : 0.007,'search_clicks' : 0.00719,'tv_impressions' : 0.000173,
    #                    "digital_clicks":0.005,"streaming_clicks":0.004,'streaming_spends':1,"tv_spends":1,"search_spends":1,
    #                    "digital_spends":1}
    print('State initialized')
    excel = pd.read_excel("Overview_data_test.xlsx",sheet_name=None)
    raw_df = excel['RAW DATA MMM']

    spend_df = excel['SPEND INPUT']
    contri_df = excel['CONTRIBUTION MMM']
    #Revenue_df = excel['Revenue']
    
    ## remove sesonalities, indices etc ... 
    exclude_columns = ['Date',
                       'Region',
                       'Controls_Grammarly_Index_SeasonalAVG',
                       'Controls_Quillbot_Index',
                       'Daily_Positive_Outliers',
                       'External_RemoteClass_Index',
                       'Intervals ON 20190520-20190805 | 20200518-20200803 | 20210517-20210802',
                       'Intervals ON 20190826-20191209 | 20200824-20201207 | 20210823-20211206',
                       'Intervals ON 20201005-20201019',
                       'Promotion_PercentOff',
                       'Promotion_TimeBased',
                       'Seasonality_Indicator_Chirstmas',
                       'Seasonality_Indicator_NewYears_Days',
                       'Seasonality_Indicator_Thanksgiving',
                       'Trend 20200302 / 20200803',
                  ]
    raw_df['Date']=pd.to_datetime(raw_df['Date'])
    contri_df['Date']=pd.to_datetime(contri_df['Date'])
    input_df = raw_df.sort_values(by='Date')
    output_df = contri_df.sort_values(by='Date')
    spend_df['Week'] = pd.to_datetime(spend_df['Week'], format='%Y-%m-%d', errors='coerce')
    spend_df.sort_values(by='Week', inplace=True)

    # spend_df['Week'] = pd.to_datetime(spend_df['Week'], errors='coerce')
    # spend_df = spend_df.sort_values(by='Week')
    

    channel_list = [col for col in input_df.columns if col not in exclude_columns]
    
    response_curves = {}
    mapes = {}
    rmses = {}
    upper_limits = {}
    powers = {}
    r2 = {}
    conv_rates = {}
    output_cols = []
    channels = {}
    sales = None
    dates = input_df.Date.values
    actual_output_dic = {}
    actual_input_dic = {}
    
    for inp_col in channel_list:
        #st.write(inp_col)
        spends = input_df[inp_col].values
        x = spends.copy()
        # upper limit for penalty   
        upper_limits[inp_col] = 2*x.max()
        
        # contribution
        out_col = [_col for _col in output_df.columns if _col.startswith(inp_col)][0]
        y = output_df[out_col].values.copy()
        actual_output_dic[inp_col] = y.copy()
        actual_input_dic[inp_col] = x.copy()
        ##output cols aggregation
        output_cols.append(out_col)
        
        ## scale the input
        power = (np.ceil(np.log(x.max()) / np.log(10) )- 3)
        if power >= 0 :
            x = x / 10**power
        
            
        x = x.astype('float64')
        y = y.astype('float64')
        print('Printing yyyyyyyyy')
        print(inp_col)
        print(x.max())
        print(y.max())
        bounds = ((0, 0, 0, 0), (3*y.max(), 1000, 1, x.max()))

        #bounds = ((y.max(), 3*y.max()),(0,1000),(0,1),(0,x.max()))
        params,_ = curve_fit(s_curve,x,y,p0=(2*y.max(),0.01,1e-5,x.max()),
                                bounds=bounds,
                                maxfev=int(1e5))
        mape = (100 * abs(1 - s_curve(x, *params) / y.clip(min=1))).mean()
        rmse =  np.sqrt(((y - s_curve(x,*params))**2).mean())
        r2_ = r2_score(y, s_curve(x,*params))

        response_curves[inp_col] = {'K' : params[0], 'b' : params[1], 'a' : params[2], 'x0' : params[3]}
        mapes[inp_col] = mape
        rmses[inp_col] = rmse
        r2[inp_col] = r2_
        powers[inp_col] = power
        
        
        ## conversion rates
        spend_col = [_col for _col in spend_df.columns if _col.startswith(inp_col.rsplit('_',1)[0])][0]

        print('printing spendssss')
        print(spend_col)
        conv = (spend_df.set_index('Week')[spend_col] / input_df.set_index('Date')[inp_col].clip(lower=1)).reset_index()
        conv.rename(columns={'index':'Week'},inplace=True)
        conv['year'] = conv.Week.dt.year
        conv_rates[inp_col] = list(conv.drop('Week',axis=1).mean().to_dict().values())[0]
        #print('Before',conv_rates[inp_col])
        # conv_rates[inp_col] = uopx_conv_rates[inp_col]
        #print('After',(conv_rates[inp_col]))
        
        
        channel = Channel(name=inp_col,dates=dates,
                            spends=spends,
                            # conversion_rate = np.mean(list(conv_rates[inp_col].values())),
                            conversion_rate = conv_rates[inp_col],
                            response_curve_type='s-curve',
                            response_curve_params={'K' : params[0], 'b' : params[1], 'a' : params[2], 'x0' : params[3]},
                            bounds=np.array([-10,10]))
        channels[inp_col] = channel
        if sales is None:
            sales = channel.actual_sales
        else:
            sales += channel.actual_sales
    other_contributions = output_df.drop([*output_cols], axis=1).sum(axis=1, numeric_only = True).values
    correction = output_df.drop('Date',axis=1).sum(axis=1).values -  (sales + other_contributions)
    scenario = Scenario(name='default', channels=channels, constant=other_contributions, correction = correction)
    ## setting session variables
    st.session_state['initialized'] = True
    st.session_state['actual_df'] = input_df
    st.session_state['raw_df'] = raw_df
    st.session_state['contri_df'] = output_df
    default_scenario_dict = class_to_dict(scenario)
    st.session_state['default_scenario_dict'] = default_scenario_dict
    st.session_state['scenario'] = scenario
    st.session_state['channels_list'] = channel_list
    st.session_state['optimization_channels'] = {channel_name : False for channel_name in channel_list}
    st.session_state['rcs'] = response_curves
    st.session_state['powers'] = powers
    st.session_state['actual_contribution_df'] = pd.DataFrame(actual_output_dic)
    st.session_state['actual_input_df'] = pd.DataFrame(actual_input_dic)
    
    for channel in channels.values():
        st.session_state[channel.name] = numerize(channel.actual_total_spends * channel.conversion_rate,1)
    
    st.session_state['xlsx_buffer'] = io.BytesIO()

    
    if Path('../saved_scenarios.pkl').exists():
        with open('../saved_scenarios.pkl','rb') as f:
            st.session_state['saved_scenarios'] = pickle.load(f)
    else:
        st.session_state['saved_scenarios'] = OrderedDict()
        
    st.session_state['total_spends_change'] = 0
    st.session_state['optimization_channels'] = {channel_name : False for channel_name in channel_list}
    st.session_state['disable_download_button'] = True
    
# def initialize_data():
#     # fetch data from excel
#     output = pd.read_excel('data.xlsx',sheet_name=None)
#     raw_df = output['RAW DATA MMM']
#     contribution_df = output['CONTRIBUTION MMM']
#     Revenue_df = output['Revenue']

#     ## channels to be shows
#     channel_list = []
#     for col in raw_df.columns:
#         if 'click' in col.lower() or 'spend' in col.lower() or 'imp' in col.lower():
#             #print(col)
#             channel_list.append(col)
#         else:
#             pass
    
#     ## NOTE : Considered only Desktop spends for all calculations
#     acutal_df = raw_df[raw_df.Region == 'Desktop'].copy()
#     ## NOTE : Considered one year of data
#     acutal_df = acutal_df[acutal_df.Date>'2020-12-31']
#     actual_df = acutal_df.drop('Region',axis=1).sort_values(by='Date')[[*channel_list,'Date']]
    
#     ##load response curves
#     with open('./grammarly_response_curves.json','r') as f:
#         response_curves = json.load(f)
        
#     ## create channel dict for scenario creation
#     dates = actual_df.Date.values
#     channels = {}
#     rcs = {}
#     constant = 0.
#     for i,info_dict in enumerate(response_curves):
#         name = info_dict.get('name')
#         response_curve_type = info_dict.get('response_curve')
#         response_curve_params = info_dict.get('params')
#         rcs[name] = response_curve_params
#         if name != 'constant':
#             spends = actual_df[name].values
#             channel = Channel(name=name,dates=dates,
#                             spends=spends,
#                             response_curve_type=response_curve_type,
#                             response_curve_params=response_curve_params,
#                             bounds=np.array([-30,30]))
            
#             channels[name] = channel
#         else:
#             constant = info_dict.get('value',0.) * len(dates)
            
#     ## create scenario
#     scenario = Scenario(name='default', channels=channels, constant=constant)
#     default_scenario_dict = class_to_dict(scenario)
    

#     ## setting session variables
#     st.session_state['initialized'] = True
#     st.session_state['actual_df'] = actual_df
#     st.session_state['raw_df'] = raw_df
#     st.session_state['default_scenario_dict'] = default_scenario_dict
#     st.session_state['scenario'] = scenario
#     st.session_state['channels_list'] = channel_list
#     st.session_state['optimization_channels'] = {channel_name : False for channel_name in channel_list}
#     st.session_state['rcs'] = rcs
#     for channel in channels.values():
#         if channel.name not in st.session_state:
#             st.session_state[channel.name] = float(channel.actual_total_spends)
    
#     if 'xlsx_buffer' not in st.session_state:
#         st.session_state['xlsx_buffer'] = io.BytesIO()

#     ## for saving scenarios
#     if 'saved_scenarios' not in st.session_state:
#         if Path('../saved_scenarios.pkl').exists():
#             with open('../saved_scenarios.pkl','rb') as f:
#                 st.session_state['saved_scenarios'] = pickle.load(f)
        
#         else:
#             st.session_state['saved_scenarios'] = OrderedDict()

#     if 'total_spends_change' not in st.session_state:
#         st.session_state['total_spends_change'] = 0
        
#     if 'optimization_channels' not in st.session_state:
#         st.session_state['optimization_channels'] = {channel_name : False for channel_name in channel_list}
    
#     if 'disable_download_button' not in st.session_state:
#         st.session_state['disable_download_button'] = True
        
       
def create_channel_summary(scenario):
    summary_columns = []
    actual_spends_rows = []
    actual_sales_rows = []
    actual_roi_rows = []
    for channel in scenario.channels.values():
        name_mod = channel.name.replace('_', ' ')
        if name_mod.lower().endswith(' imp'):
            name_mod = name_mod.replace('Imp',' Impressions')
        print(name_mod,channel.actual_total_spends,channel.conversion_rate,channel.actual_total_spends * channel.conversion_rate)
        summary_columns.append(name_mod)
        actual_spends_rows.append(format_numbers(float(channel.actual_total_spends * channel.conversion_rate)))
        actual_sales_rows.append(format_numbers((float(channel.actual_total_sales))))  
        actual_roi_rows.append(decimal_formater(format_numbers((channel.actual_total_sales ) / (channel.actual_total_spends * channel.conversion_rate),include_indicator=False,n_decimals=4),n_decimals=4))
        
    actual_summary_df = pd.DataFrame([summary_columns,actual_spends_rows,actual_sales_rows,actual_roi_rows]).T
    actual_summary_df.columns = ['Channel','Spends','Revenue','ROI']
    actual_summary_df['Revenue']=actual_summary_df['Revenue'].map(lambda x: str(x)[1:])
    
    return actual_summary_df


@st.cache(allow_output_mutation=True)
# def create_contribution_pie(scenario):
#     #c1f7dc
#     colors_map = {col:color for col,color in zip(st.session_state['channels_list'],plotly.colors.n_colors(plotly.colors.hex_to_rgb('#BE6468'), plotly.colors.hex_to_rgb('#E7B8B7'),23))}
#     total_contribution_fig = make_subplots(rows=1, cols=2,subplot_titles=['Spends','Revenue'],specs=[[{"type": "pie"}, {"type": "pie"}]])
#     total_contribution_fig.add_trace(
#                 go.Pie(labels=[channel_name_formating(channel_name) for channel_name in st.session_state['channels_list']] + ['Non Media'],
#                     values= [round(scenario.channels[channel_name].actual_total_spends * scenario.channels[channel_name].conversion_rate,1) for channel_name in st.session_state['channels_list']] + [0],
#                     marker=dict(colors = [plotly.colors.label_rgb(colors_map[channel_name]) for channel_name in st.session_state['channels_list']] + ['#F0F0F0']),
#                         hole=0.3),
#                 row=1, col=1)

#     total_contribution_fig.add_trace(
#                 go.Pie(labels=[channel_name_formating(channel_name) for channel_name in st.session_state['channels_list']] + ['Non Media'],
#                     values= [scenario.channels[channel_name].actual_total_sales for channel_name in st.session_state['channels_list']] + [scenario.correction.sum() + scenario.constant.sum()],
#                         hole=0.3),
#                 row=1, col=2)

#     total_contribution_fig.update_traces(textposition='inside',texttemplate='%{percent:.1%}')
#     total_contribution_fig.update_layout(uniformtext_minsize=12,title='Channel contribution', uniformtext_mode='hide')
#     return total_contribution_fig

# @st.cache(allow_output_mutation=True)

# def create_contribuion_stacked_plot(scenario):
#     weekly_contribution_fig = make_subplots(rows=1, cols=2,subplot_titles=['Spends','Revenue'],specs=[[{"type": "bar"}, {"type": "bar"}]])
#     raw_df = st.session_state['raw_df']
#     df = raw_df.sort_values(by='Date')
#     x = df.Date
#     weekly_spends_data = []
#     weekly_sales_data = []
#     for channel_name in st.session_state['channels_list']:
#         weekly_spends_data.append((go.Bar(x=x, 
#                                           y=scenario.channels[channel_name].actual_spends * scenario.channels[channel_name].conversion_rate,
#                                           name=channel_name_formating(channel_name), 
#                                           hovertemplate="Date:%{x}<br>Spend:%{y:$.2s}",
#                                           legendgroup=channel_name)))
#         weekly_sales_data.append((go.Bar(x=x, 
#                                          y=scenario.channels[channel_name].actual_sales,
#                                          name=channel_name_formating(channel_name), 
#                                          hovertemplate="Date:%{x}<br>Revenue:%{y:$.2s}",
#                                          legendgroup=channel_name, showlegend=False)))
#     for _d in weekly_spends_data:
#         weekly_contribution_fig.add_trace(_d, row=1, col=1)
#     for _d in weekly_sales_data:
#         weekly_contribution_fig.add_trace(_d, row=1, col=2)
#     weekly_contribution_fig.add_trace(go.Bar(x=x, 
#                                          y=scenario.constant + scenario.correction,
#                                          name='Non Media', 
#                                          hovertemplate="Date:%{x}<br>Revenue:%{y:$.2s}"), row=1, col=2)
#     weekly_contribution_fig.update_layout(barmode='stack', title='Channel contribuion by week', xaxis_title='Date')
#     weekly_contribution_fig.update_xaxes(showgrid=False)
#     weekly_contribution_fig.update_yaxes(showgrid=False)
#     return weekly_contribution_fig

# @st.cache(allow_output_mutation=True)
# def create_channel_spends_sales_plot(channel):
#     if channel is not None:
#         x = channel.dates
#         _spends = channel.actual_spends * channel.conversion_rate
#         _sales = channel.actual_sales
#         channel_sales_spends_fig = make_subplots(specs=[[{"secondary_y": True}]])
#         channel_sales_spends_fig.add_trace(go.Bar(x=x, y=_sales,marker_color='#c1f7dc',name='Revenue', hovertemplate="Date:%{x}<br>Revenue:%{y:$.2s}"), secondary_y = False)
#         channel_sales_spends_fig.add_trace(go.Scatter(x=x, y=_spends,line=dict(color='#005b96'),name='Spends',hovertemplate="Date:%{x}<br>Spend:%{y:$.2s}"), secondary_y = True)
#         channel_sales_spends_fig.update_layout(xaxis_title='Date',yaxis_title='Revenue',yaxis2_title='Spends ($)',title='Channel spends and Revenue week wise')
#         channel_sales_spends_fig.update_xaxes(showgrid=False)
#         channel_sales_spends_fig.update_yaxes(showgrid=False)
#     else:
#         raw_df = st.session_state['raw_df']
#         df = raw_df.sort_values(by='Date')
#         x = df.Date
#         scenario = class_from_dict(st.session_state['default_scenario_dict'])
#         _sales = scenario.constant + scenario.correction
#         channel_sales_spends_fig = make_subplots(specs=[[{"secondary_y": True}]])
#         channel_sales_spends_fig.add_trace(go.Bar(x=x, y=_sales,marker_color='#c1f7dc',name='Revenue', hovertemplate="Date:%{x}<br>Revenue:%{y:$.2s}"), secondary_y = False)
#         # channel_sales_spends_fig.add_trace(go.Scatter(x=x, y=_spends,line=dict(color='#15C39A'),name='Spends',hovertemplate="Date:%{x}<br>Spend:%{y:$.2s}"), secondary_y = True)
#         channel_sales_spends_fig.update_layout(xaxis_title='Date',yaxis_title='Revenue',yaxis2_title='Spends ($)',title='Channel spends and Revenue week wise')
#         channel_sales_spends_fig.update_xaxes(showgrid=False)
#         channel_sales_spends_fig.update_yaxes(showgrid=False)
#     return channel_sales_spends_fig


# Define a shared color palette


def create_contribution_pie(scenario):
    total_contribution_fig = make_subplots(rows=1, cols=2, subplot_titles=['Spends', 'Revenue'], specs=[[{"type": "pie"}, {"type": "pie"}]])
    
    colors_map = {col: color_palette[i % len(color_palette)] for i, col in enumerate(st.session_state['channels_list'])}
    colors_map['Non Media'] = color_palette[-1]

    total_contribution_fig.add_trace(
        go.Pie(labels=[channel_name_formating(channel_name) for channel_name in st.session_state['channels_list']] + ['Non Media'],
               values=[round(scenario.channels[channel_name].actual_total_spends * scenario.channels[channel_name].conversion_rate, 1) for channel_name in st.session_state['channels_list']] + [0],
               marker=dict(colors=[colors_map[channel_name] for channel_name in st.session_state['channels_list']] + [color_palette[-1]]),
               hole=0.3),
        row=1, col=1)

    total_contribution_fig.add_trace(
        go.Pie(labels=[channel_name_formating(channel_name) for channel_name in st.session_state['channels_list']] + ['Non Media'],
               values=[scenario.channels[channel_name].actual_total_sales for channel_name in st.session_state['channels_list']] + [scenario.correction.sum() + scenario.constant.sum()],
               hole=0.3),
        row=1, col=2)

    total_contribution_fig.update_traces(textposition='inside', texttemplate='%{percent:.1%}')
    total_contribution_fig.update_layout(uniformtext_minsize=12, title='Channel contribution', uniformtext_mode='hide')
    return total_contribution_fig

def create_contribuion_stacked_plot(scenario):
    weekly_contribution_fig = make_subplots(rows=1, cols=2, subplot_titles=['Spends', 'Revenue'], specs=[[{"type": "bar"}, {"type": "bar"}]])
    raw_df = st.session_state['raw_df']
    df = raw_df.sort_values(by='Date')
    x = df.Date
    weekly_spends_data = []
    weekly_sales_data = []
    
    for i, channel_name in enumerate(st.session_state['channels_list']):
        color = color_palette[i % len(color_palette)]
        
        weekly_spends_data.append(go.Bar(
            x=x,
            y=scenario.channels[channel_name].actual_spends * scenario.channels[channel_name].conversion_rate,
            name=channel_name_formating(channel_name),
            hovertemplate="Date:%{x}<br>Spend:%{y:$.2s}",
            legendgroup=channel_name,
            marker_color=color,
        ))
        
        weekly_sales_data.append(go.Bar(
            x=x,
            y=scenario.channels[channel_name].actual_sales,
            name=channel_name_formating(channel_name),
            hovertemplate="Date:%{x}<br>Revenue:%{y:$.2s}",
            legendgroup=channel_name,
            showlegend=False,
            marker_color=color,
        ))
    
    for _d in weekly_spends_data:
        weekly_contribution_fig.add_trace(_d, row=1, col=1)
    for _d in weekly_sales_data:
        weekly_contribution_fig.add_trace(_d, row=1, col=2)
    
    weekly_contribution_fig.add_trace(go.Bar(
        x=x,
        y=scenario.constant + scenario.correction,
        name='Non Media',
        hovertemplate="Date:%{x}<br>Revenue:%{y:$.2s}",
        marker_color=color_palette[-1],
    ), row=1, col=2)

    weekly_contribution_fig.update_layout(barmode='stack', title='Channel contribution by week', xaxis_title='Date')
    weekly_contribution_fig.update_xaxes(showgrid=False)
    weekly_contribution_fig.update_yaxes(showgrid=False)
    return weekly_contribution_fig

def create_channel_spends_sales_plot(channel):
    if channel is not None:
        x = channel.dates
        _spends = channel.actual_spends * channel.conversion_rate
        _sales = channel.actual_sales
        channel_sales_spends_fig = make_subplots(specs=[[{"secondary_y": True}]])
        channel_sales_spends_fig.add_trace(go.Bar(
            x=x,
            y=_sales,
            marker_color=color_palette[3],  # You can choose a color from the palette
            name='Revenue',
            hovertemplate="Date:%{x}<br>Revenue:%{y:$.2s}",
        ), secondary_y=False)
        
        channel_sales_spends_fig.add_trace(go.Scatter(
            x=x,
            y=_spends,
            line=dict(color=color_palette[2]),  # You can choose another color from the palette
            name='Spends',
            hovertemplate="Date:%{x}<br>Spend:%{y:$.2s}",
        ), secondary_y=True)
        
        channel_sales_spends_fig.update_layout(xaxis_title='Date', yaxis_title='Revenue', yaxis2_title='Spends ($)', title='Channel spends and Revenue week-wise')
        channel_sales_spends_fig.update_xaxes(showgrid=False)
        channel_sales_spends_fig.update_yaxes(showgrid=False)
    else:
        raw_df = st.session_state['raw_df']
        df = raw_df.sort_values(by='Date')
        x = df.Date
        scenario = class_from_dict(st.session_state['default_scenario_dict'])
        _sales = scenario.constant + scenario.correction
        channel_sales_spends_fig = make_subplots(specs=[[{"secondary_y": True}]])
        channel_sales_spends_fig.add_trace(go.Bar(
            x=x,
            y=_sales,
            marker_color=color_palette[0],  # You can choose a color from the palette
            name='Revenue',
            hovertemplate="Date:%{x}<br>Revenue:%{y:$.2s}",
        ), secondary_y=False)
        
        channel_sales_spends_fig.update_layout(xaxis_title='Date', yaxis_title='Revenue', yaxis2_title='Spends ($)', title='Channel spends and Revenue week-wise')
        channel_sales_spends_fig.update_xaxes(showgrid=False)
        channel_sales_spends_fig.update_yaxes(showgrid=False)
    
    return channel_sales_spends_fig

def format_numbers(value, n_decimals=1,include_indicator = True):
    if include_indicator:
        return f'{CURRENCY_INDICATOR} {numerize(value,n_decimals)}'
    else:
        return f'{numerize(value,n_decimals)}'


def decimal_formater(num_string,n_decimals=1):
    parts = num_string.split('.')
    if len(parts) == 1:
        return num_string+'.' + '0'*n_decimals
    else:
        to_be_padded = n_decimals - len(parts[-1])
        if to_be_padded > 0 :
            return num_string+'0'*to_be_padded
        else:
            return num_string
        
        
def channel_name_formating(channel_name):
    name_mod = channel_name.replace('_', ' ')
    if name_mod.lower().endswith(' imp'):
        name_mod = name_mod.replace('Imp','Spend')
    elif name_mod.lower().endswith(' clicks'):
        name_mod = name_mod.replace('Clicks','Spend')
    return name_mod


def send_email(email,message):
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login("geethu4444@gmail.com", "jgydhpfusuremcol")
    s.sendmail("geethu4444@gmail.com", email, message)
    s.quit()

if __name__ == "__main__":
    initialize_data()