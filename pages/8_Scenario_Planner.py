import streamlit as st
from numerize.numerize import numerize
import numpy as np
from functools import partial
from collections import OrderedDict
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from utilities import (
    format_numbers,
    load_local_css,
    set_header,
    initialize_data,
    load_authenticator,
    send_email,
    channel_name_formating,
)
from classes import class_from_dict, class_to_dict
import pickle
import streamlit_authenticator as stauth
import yaml
from yaml import SafeLoader
import re
import pandas as pd
import plotly.express as px

target = "Revenue"
st.set_page_config(layout="wide")
load_local_css("styles.css")
set_header()

for k, v in st.session_state.items():
    if k not in ["logout", "login", "config"] and not k.startswith(
        "FormSubmitter"
    ):
        st.session_state[k] = v
# ======================================================== #
# ======================= Functions ====================== #
# ======================================================== #


def optimize(key):
    """
    Optimize the spends for the sales
    """

    channel_list = [
        key
        for key, value in st.session_state["optimization_channels"].items()
        if value
    ]
    # print('channel_list')
    # print(channel_list)
    # print('@@@@@@@@')
    if len(channel_list) > 0:
        scenario = st.session_state["scenario"]
        if key.lower() == "spends":
            with status_placeholder:
                with st.spinner("Optimizing Sales"):
                    result = st.session_state["scenario"].optimize(
                        st.session_state["total_spends_change"], channel_list
                    )
        elif key.lower() == "sales":
            with status_placeholder:
                with st.spinner("Optimizing Spends"):

                    result = st.session_state["scenario"].optimize_spends(
                        st.session_state["total_sales_change"], channel_list
                    )
        for channel_name, modified_spends in result:

            st.session_state[channel_name] = numerize(
                modified_spends
                * scenario.channels[channel_name].conversion_rate,
                1,
            )
            prev_spends = (
                st.session_state["scenario"]
                .channels[channel_name]
                .actual_total_spends
            )
            st.session_state[f"{channel_name}_change"] = round(
                100 * (modified_spends - prev_spends) / prev_spends, 2
            )


def save_scenario(scenario_name):
    """
    Save the current scenario with the mentioned name in the session state

    Parameters
    ----------
    scenario_name
        Name of the scenario to be saved
    """
    if "saved_scenarios" not in st.session_state:
        st.session_state = OrderedDict()

    # st.session_state['saved_scenarios'][scenario_name] = st.session_state['scenario'].save()
    st.session_state["saved_scenarios"][scenario_name] = class_to_dict(
        st.session_state["scenario"]
    )
    st.session_state["scenario_input"] = ""
    # print(type(st.session_state['saved_scenarios']))
    with open("../saved_scenarios.pkl", "wb") as f:
        pickle.dump(st.session_state["saved_scenarios"], f)


def update_sales_abs():
    actual_sales = _scenario.actual_total_sales
    if validate_input(st.session_state["total_sales_change_abs"]):
        modified_sales = extract_number_for_string(
            st.session_state["total_sales_change_abs"]
        )
        st.session_state["total_sales_change"] = round(
            ((modified_sales / actual_sales) - 1) * 100
        )


def update_sales():
    st.session_state["total_sales_change_abs"] = numerize(
        (1 + st.session_state["total_sales_change"] / 100)
        * _scenario.actual_total_sales,
        1,
    )


def update_all_spends_abs():
    actual_spends = _scenario.actual_total_spends
    if validate_input(st.session_state["total_spends_change_abs"]):
        modified_spends = extract_number_for_string(
            st.session_state["total_spends_change_abs"]
        )
        print(modified_spends)
        print(actual_spends)

        st.session_state["total_spends_change"] = (
            (modified_spends / actual_spends) - 1
        ) * 100

        update_all_spends()


def update_all_spends():
    """
    Updates spends for all the channels with the given overall spends change
    """
    percent_change = st.session_state["total_spends_change"]
    st.session_state["total_spends_change_abs"] = numerize(
        (1 + percent_change / 100) * _scenario.actual_total_spends, 1
    )
    for channel_name in st.session_state["channels_list"]:
        channel = st.session_state["scenario"].channels[channel_name]
        current_spends = channel.actual_total_spends
        modified_spends = (1 + percent_change / 100) * current_spends
        st.session_state["scenario"].update(channel_name, modified_spends)
        st.session_state[channel_name] = numerize(
            modified_spends * channel.conversion_rate, 1
        )
        st.session_state[f"{channel_name}_change"] = percent_change


def extract_number_for_string(string_input):
    string_input = string_input.upper()
    if string_input.endswith("K"):
        return float(string_input[:-1]) * 10**3
    elif string_input.endswith("M"):
        return float(string_input[:-1]) * 10**6
    elif string_input.endswith("B"):
        return float(string_input[:-1]) * 10**9


def validate_input(string_input):
    pattern = r"\d+\.?\d*[K|M|B]$"
    match = re.match(pattern, string_input)
    if match is None:
        return False
    return True


def update_data_by_percent(channel_name):
    prev_spends = (
        st.session_state["scenario"].channels[channel_name].actual_total_spends
        * st.session_state["scenario"].channels[channel_name].conversion_rate
    )
    modified_spends = prev_spends * (
        1 + st.session_state[f"{channel_name}_change"] / 100
    )
    st.session_state[channel_name] = numerize(modified_spends, 1)
    st.session_state["scenario"].update(
        channel_name,
        modified_spends
        / st.session_state["scenario"].channels[channel_name].conversion_rate,
    )


def update_data(channel_name):
    """
    Updates the spends for the given channel
    """

    if validate_input(st.session_state[channel_name]):
        modified_spends = extract_number_for_string(
            st.session_state[channel_name]
        )
        prev_spends = (
            st.session_state["scenario"]
            .channels[channel_name]
            .actual_total_spends
            * st.session_state["scenario"]
            .channels[channel_name]
            .conversion_rate
        )
        st.session_state[f"{channel_name}_change"] = round(
            100 * (modified_spends - prev_spends) / prev_spends, 2
        )
        st.session_state["scenario"].update(
            channel_name,
            modified_spends
            / st.session_state["scenario"]
            .channels[channel_name]
            .conversion_rate,
        )
    # st.session_state['scenario'].update(channel_name, modified_spends)
    # else:
    #     try:
    #         modified_spends = float(st.session_state[channel_name])
    #         prev_spends = st.session_state['scenario'].channels[channel_name].actual_total_spends * st.session_state['scenario'].channels[channel_name].conversion_rate
    #         st.session_state[f'{channel_name}_change'] = round(100*(modified_spends - prev_spends) / prev_spends,2)
    #         st.session_state['scenario'].update(channel_name, modified_spends/st.session_state['scenario'].channels[channel_name].conversion_rate)
    #         st.session_state[f'{channel_name}'] = numerize(modified_spends,1)
    #     except ValueError:
    #         st.write('Invalid input')


def select_channel_for_optimization(channel_name):
    """
    Marks the given channel for optimization
    """
    st.session_state["optimization_channels"][channel_name] = st.session_state[
        f"{channel_name}_selected"
    ]


def select_all_channels_for_optimization():
    """
    Marks all the channel for optimization
    """
    for channel_name in st.session_state["optimization_channels"].keys():
        st.session_state[f"{channel_name}_selected"] = st.session_state[
            "optimze_all_channels"
        ]
        st.session_state["optimization_channels"][channel_name] = (
            st.session_state["optimze_all_channels"]
        )


def update_penalty():
    """
    Updates the penalty flag for sales calculation
    """
    st.session_state["scenario"].update_penalty(
        st.session_state["apply_penalty"]
    )


def reset_scenario():
    # #print(st.session_state['default_scenario_dict'])
    # st.session_state['scenario']  = class_from_dict(st.session_state['default_scenario_dict'])
    # for channel in st.session_state['scenario'].channels.values():
    #     st.session_state[channel.name] = float(channel.actual_total_spends * channel.conversion_rate)
    initialize_data()
    for channel_name in st.session_state["channels_list"]:
        st.session_state[f"{channel_name}_selected"] = False
        st.session_state[f"{channel_name}_change"] = 0
    st.session_state["optimze_all_channels"] = False


def format_number(num):
    if num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.0f}K"
    else:
        return f"{num:.2f}"


def summary_plot(data, x, y, title, text_column):
    fig = px.bar(
        data,
        x=x,
        y=y,
        orientation="h",
        title=title,
        text=text_column,
        color="Channel_name",
    )

    # Convert text_column to numeric values
    data[text_column] = pd.to_numeric(data[text_column], errors="coerce")

    # Update the format of the displayed text based on magnitude
    fig.update_traces(
        texttemplate="%{text:.2s}",
        textposition="outside",
        hovertemplate="%{x:.2s}",
    )

    fig.update_layout(
        xaxis_title=x, yaxis_title="Channel Name", showlegend=False
    )
    return fig


def s_curve(x, K, b, a, x0):
    return K / (1 + b * np.exp(-a * (x - x0)))


def find_segment_value(x, roi, mroi):
    start_value = x[0]
    end_value = x[len(x) - 1]

    # Condition for green region: Both MROI and ROI > 1
    green_condition = (roi > 1) & (mroi > 1)
    left_indices = np.where(green_condition)[0]
    left_value = x[left_indices[0]] if left_indices.size > 0 else x[0]

    right_indices = np.where(green_condition)[0]
    right_value = x[right_indices[-1]] if right_indices.size > 0 else x[0]

    return start_value, end_value, left_value, right_value


def calculate_rgba(
    start_value, end_value, left_value, right_value, current_channel_spends
):
    # Initialize alpha to None for clarity
    alpha = None

    # Determine the color and calculate relative_position and alpha based on the point's position
    if start_value <= current_channel_spends <= left_value:
        color = "yellow"
        relative_position = (current_channel_spends - start_value) / (
            left_value - start_value
        )
        alpha = 0.8 - (
            0.6 * relative_position
        )  # Alpha decreases from start to end

    elif left_value < current_channel_spends <= right_value:
        color = "green"
        relative_position = (current_channel_spends - left_value) / (
            right_value - left_value
        )
        alpha = 0.8 - (
            0.6 * relative_position
        )  # Alpha decreases from start to end

    elif right_value < current_channel_spends <= end_value:
        color = "red"
        relative_position = (current_channel_spends - right_value) / (
            end_value - right_value
        )
        alpha = 0.2 + (
            0.6 * relative_position
        )  # Alpha increases from start to end

    else:
        # Default case, if the spends are outside the defined ranges
        return "rgba(136, 136, 136, 0.5)"  # Grey for values outside the range

    # Ensure alpha is within the intended range in case of any calculation overshoot
    alpha = max(0.2, min(alpha, 0.8))

    # Define color codes for RGBA
    color_codes = {
        "yellow": "255, 255, 0",  # RGB for yellow
        "green": "0, 128, 0",  # RGB for green
        "red": "255, 0, 0",  # RGB for red
    }

    rgba = f"rgba({color_codes[color]}, {alpha})"
    return rgba


def debug_temp(x_test, power, K, b, a, x0):
    print("*" * 100)
    # Calculate the count of bins
    count_lower_bin = sum(1 for x in x_test if x <= 2524)
    count_center_bin = sum(1 for x in x_test if x > 2524 and x <= 3377)
    count_ = sum(1 for x in x_test if x > 3377)

    print(
        f"""
            lower : {count_lower_bin}
            center : {count_center_bin}
            upper : {count_}
          """
    )


# @st.cache
def plot_response_curves():
    cols = 4
    rows = (
        len(channels_list) // cols
        if len(channels_list) % cols == 0
        else len(channels_list) // cols + 1
    )
    rcs = st.session_state["rcs"]
    shapes = []
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=channels_list)
    for i in range(0, len(channels_list)):
        col = channels_list[i]
        x_actual = st.session_state["scenario"].channels[col].actual_spends
        # x_modified = st.session_state["scenario"].channels[col].modified_spends

        power = np.ceil(np.log(x_actual.max()) / np.log(10)) - 3

        K = rcs[col]["K"]
        b = rcs[col]["b"]
        a = rcs[col]["a"]
        x0 = rcs[col]["x0"]

        x_plot = np.linspace(0, 5 * x_actual.sum(), 50)

        x, y, marginal_roi = [], [], []
        for x_p in x_plot:
            x.append(x_p * x_actual / x_actual.sum())

        for index in range(len(x_plot)):
            y.append(s_curve(x[index] / 10**power, K, b, a, x0))

        for index in range(len(x_plot)):
            marginal_roi.append(
                a
                * y[index]
                * (1 - y[index] / np.maximum(K, np.finfo(float).eps))
            )

        x = (
            np.sum(x, axis=1)
            * st.session_state["scenario"].channels[col].conversion_rate
        )
        y = np.sum(y, axis=1)
        marginal_roi = (
            np.average(marginal_roi, axis=1)
            / st.session_state["scenario"].channels[col].conversion_rate
        )

        roi = y / np.maximum(x, np.finfo(float).eps)

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=col,
                customdata=np.stack((roi, marginal_roi), axis=-1),
                hovertemplate="Spend:%{x:$.2s}<br>Sale:%{y:$.2s}<br>ROI:%{customdata[0]:.3f}<br>MROI:%{customdata[1]:.3f}",
                line=dict(color="blue"),
            ),
            row=1 + (i) // cols,
            col=i % cols + 1,
        )

        x_optimal = (
            st.session_state["scenario"].channels[col].modified_total_spends
            * st.session_state["scenario"].channels[col].conversion_rate
        )
        y_optimal = (
            st.session_state["scenario"].channels[col].modified_total_sales
        )

        # if col == "Paid_social_others":
        #     debug_temp(x_optimal * x_actual / x_actual.sum(), power, K, b, a, x0)

        fig.add_trace(
            go.Scatter(
                x=[x_optimal],
                y=[y_optimal],
                name=col,
                legendgroup=col,
                showlegend=False,
                marker=dict(color=["black"]),
            ),
            row=1 + (i) // cols,
            col=i % cols + 1,
        )

        shapes.append(
            go.layout.Shape(
                type="line",
                x0=0,
                y0=y_optimal,
                x1=x_optimal,
                y1=y_optimal,
                line_width=1,
                line_dash="dash",
                line_color="black",
                xref=f"x{i+1}",
                yref=f"y{i+1}",
            )
        )

        shapes.append(
            go.layout.Shape(
                type="line",
                x0=x_optimal,
                y0=0,
                x1=x_optimal,
                y1=y_optimal,
                line_width=1,
                line_dash="dash",
                line_color="black",
                xref=f"x{i+1}",
                yref=f"y{i+1}",
            )
        )

        start_value, end_value, left_value, right_value = find_segment_value(
            x,
            roi,
            marginal_roi,
        )

        # Adding background colors
        y_max = y.max() * 1.3  # 30% extra space above the max

        # Yellow region
        shapes.append(
            go.layout.Shape(
                type="rect",
                x0=start_value,
                y0=0,
                x1=left_value,
                y1=y_max,
                line=dict(width=0),
                fillcolor="rgba(255, 255, 0, 0.3)",
                layer="below",
                xref=f"x{i+1}",
                yref=f"y{i+1}",
            )
        )

        # Green region
        shapes.append(
            go.layout.Shape(
                type="rect",
                x0=left_value,
                y0=0,
                x1=right_value,
                y1=y_max,
                line=dict(width=0),
                fillcolor="rgba(0, 255, 0, 0.3)",
                layer="below",
                xref=f"x{i+1}",
                yref=f"y{i+1}",
            )
        )

        # Red region
        shapes.append(
            go.layout.Shape(
                type="rect",
                x0=right_value,
                y0=0,
                x1=end_value,
                y1=y_max,
                line=dict(width=0),
                fillcolor="rgba(255, 0, 0, 0.3)",
                layer="below",
                xref=f"x{i+1}",
                yref=f"y{i+1}",
            )
        )

    fig.update_layout(
        # height=1000,
        # width=1000,
        title_text="Response Curves (X: Spends Vs Y: Revenue)",
        showlegend=False,
        shapes=shapes,
    )
    fig.update_annotations(font_size=10)
    # fig.update_xaxes(title="Spends")
    # fig.update_yaxes(title=target)
    fig.update_yaxes(
        gridcolor="rgba(136, 136, 136, 0.5)", gridwidth=0.5, griddash="dash"
    )

    return fig


# @st.cache
# def plot_response_curves():
#     cols = 4
#     rcs = st.session_state["rcs"]
#     shapes = []
#     fig = make_subplots(rows=6, cols=cols, subplot_titles=channels_list)
#     for i in range(0, len(channels_list)):
#         col = channels_list[i]
#         x = st.session_state["actual_df"][col].values
#         spends = x.sum()
#         power = np.ceil(np.log(x.max()) / np.log(10)) - 3
#         x = np.linspace(0, 3 * x.max(), 200)

#         K = rcs[col]["K"]
#         b = rcs[col]["b"]
#         a = rcs[col]["a"]
#         x0 = rcs[col]["x0"]

#         y = s_curve(x / 10**power, K, b, a, x0)
#         roi = y / x
#         marginal_roi = a * (y) * (1 - y / K)
#         fig.add_trace(
#             go.Scatter(
#                 x=52
#                 * x
#                 * st.session_state["scenario"].channels[col].conversion_rate,
#                 y=52 * y,
#                 name=col,
#                 customdata=np.stack((roi, marginal_roi), axis=-1),
#                 hovertemplate="Spend:%{x:$.2s}<br>Sale:%{y:$.2s}<br>ROI:%{customdata[0]:.3f}<br>MROI:%{customdata[1]:.3f}",
#             ),
#             row=1 + (i) // cols,
#             col=i % cols + 1,
#         )

#         fig.add_trace(
#             go.Scatter(
#                 x=[
#                     spends
#                     * st.session_state["scenario"]
#                     .channels[col]
#                     .conversion_rate
#                 ],
#                 y=[52 * s_curve(spends / (10**power * 52), K, b, a, x0)],
#                 name=col,
#                 legendgroup=col,
#                 showlegend=False,
#                 marker=dict(color=["black"]),
#             ),
#             row=1 + (i) // cols,
#             col=i % cols + 1,
#         )

#         shapes.append(
#             go.layout.Shape(
#                 type="line",
#                 x0=0,
#                 y0=52 * s_curve(spends / (10**power * 52), K, b, a, x0),
#                 x1=spends
#                 * st.session_state["scenario"].channels[col].conversion_rate,
#                 y1=52 * s_curve(spends / (10**power * 52), K, b, a, x0),
#                 line_width=1,
#                 line_dash="dash",
#                 line_color="black",
#                 xref=f"x{i+1}",
#                 yref=f"y{i+1}",
#             )
#         )

#         shapes.append(
#             go.layout.Shape(
#                 type="line",
#                 x0=spends
#                 * st.session_state["scenario"].channels[col].conversion_rate,
#                 y0=0,
#                 x1=spends
#                 * st.session_state["scenario"].channels[col].conversion_rate,
#                 y1=52 * s_curve(spends / (10**power * 52), K, b, a, x0),
#                 line_width=1,
#                 line_dash="dash",
#                 line_color="black",
#                 xref=f"x{i+1}",
#                 yref=f"y{i+1}",
#             )
#         )

#     fig.update_layout(
#         height=1500,
#         width=1000,
#         title_text="Response Curves",
#         showlegend=False,
#         shapes=shapes,
#     )
#     fig.update_annotations(font_size=10)
#     fig.update_xaxes(title="Spends")
#     fig.update_yaxes(title=target)
#     return fig


# ======================================================== #
# ==================== HTML Components =================== #
# ======================================================== #


def generate_spending_header(heading):
    return st.markdown(
        f"""<h2 class="spends-header">{heading}</h2>""", unsafe_allow_html=True
    )


# ======================================================== #
# =================== Session variables ================== #
# ======================================================== #

with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)
    st.session_state["config"] = config

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
    config["preauthorized"],
)
st.session_state["authenticator"] = authenticator
name, authentication_status, username = authenticator.login("Login", "main")
auth_status = st.session_state.get("authentication_status")
if auth_status == True:
    authenticator.logout("Logout", "main")
    is_state_initiaized = st.session_state.get("initialized", False)
    if not is_state_initiaized:
        initialize_data()

    channels_list = st.session_state["channels_list"]

    # ======================================================== #
    # ========================== UI ========================== #
    # ======================================================== #

    # print(list(st.session_state.keys()))

    st.header("Simulation")
    main_header = st.columns((2, 2))
    sub_header = st.columns((1, 1, 1, 1))
    _scenario = st.session_state["scenario"]

    if "total_spends_change_abs" not in st.session_state:
        st.session_state["total_spends_change_abs"] = numerize(
            _scenario.actual_total_spends, 1
        )

    if "total_sales_change_abs" not in st.session_state:
        st.session_state["total_sales_change_abs"] = numerize(
            _scenario.actual_total_sales, 1
        )

    with main_header[0]:
        st.subheader("Actual")

    with main_header[-1]:
        st.subheader("Simulated")

    with sub_header[0]:
        st.metric(
            label="Spends", value=format_numbers(_scenario.actual_total_spends)
        )

    with sub_header[1]:
        st.metric(
            label=target,
            value=format_numbers(
                float(_scenario.actual_total_sales), include_indicator=False
            ),
        )

    with sub_header[2]:
        st.metric(
            label="Spends",
            value=format_numbers(_scenario.modified_total_spends),
            delta=numerize(_scenario.delta_spends, 1),
        )

    with sub_header[3]:
        st.metric(
            label=target,
            value=format_numbers(
                float(_scenario.modified_total_sales), include_indicator=False
            ),
            delta=numerize(_scenario.delta_sales, 1),
        )

    with st.expander("Channel Spends Simulator"):
        _columns1 = st.columns((2, 2, 1, 1))
        with _columns1[0]:

            optimization_selection = st.selectbox(
                "Optimize", options=["Spends", "Sales"], key="optimization_key"
            )
        with _columns1[1]:
            st.markdown("#")
            st.checkbox(
                label="Optimize all Channels",
                key=f"optimze_all_channels",
                value=False,
                on_change=select_all_channels_for_optimization,
            )

        with _columns1[2]:
            st.markdown("#")
            st.button(
                "Optimize",
                on_click=optimize,
                args=(st.session_state["optimization_key"],),
            )

        with _columns1[3]:
            st.markdown("#")
            st.button("Reset", on_click=reset_scenario)

        _columns2 = st.columns((2, 2, 2))
        if st.session_state["optimization_key"] == "Spends":
            with _columns2[0]:
                spend_input = st.text_input(
                    "Absolute",
                    key="total_spends_change_abs",
                    # label_visibility="collapsed",
                    on_change=update_all_spends_abs,
                )
            with _columns2[1]:

                st.number_input(
                    "Percent",
                    key=f"total_spends_change",
                    step=1,
                    on_change=update_all_spends,
                )
        elif st.session_state["optimization_key"] == "Sales":
            with _columns2[0]:

                sales_input = st.text_input(
                    "Absolute",
                    key="total_sales_change_abs",
                    on_change=update_sales_abs,
                )
            with _columns2[1]:
                st.number_input(
                    "Percent change",
                    key=f"total_sales_change",
                    step=1,
                    on_change=update_sales,
                )

        with _columns2[2]:
            st.markdown("#")
            status_placeholder = st.empty()

        st.markdown(
            """<hr class="spends-heading-seperator">""", unsafe_allow_html=True
        )
        _columns = st.columns((2.5, 2, 1.5, 1.5, 1))
        with _columns[0]:
            generate_spending_header("Channel")
        with _columns[1]:
            generate_spending_header("Spends Input")
        with _columns[2]:
            generate_spending_header("Spends")
        with _columns[3]:
            generate_spending_header(target)
        with _columns[4]:
            generate_spending_header("Optimize")

        st.markdown(
            """<hr class="spends-heading-seperator">""", unsafe_allow_html=True
        )

        if "acutual_predicted" not in st.session_state:
            st.session_state["acutual_predicted"] = {
                "Channel_name": [],
                "Actual_spend": [],
                "Optimized_spend": [],
                "Delta": [],
            }
        for i, channel_name in enumerate(channels_list):
            _channel_class = st.session_state["scenario"].channels[
                channel_name
            ]
            _columns = st.columns((2.5, 1.5, 1.5, 1.5, 1))
            with _columns[0]:
                st.write(channel_name_formating(channel_name))
                bin_placeholder = st.container()

            with _columns[1]:
                channel_bounds = _channel_class.bounds
                channel_spends = float(_channel_class.actual_total_spends)
                min_value = float(
                    (1 + channel_bounds[0] / 100) * channel_spends
                )
                max_value = float(
                    (1 + channel_bounds[1] / 100) * channel_spends
                )
                ##print(st.session_state[channel_name])
                spend_input = st.text_input(
                    channel_name,
                    key=channel_name,
                    label_visibility="collapsed",
                    on_change=partial(update_data, channel_name),
                )
                if not validate_input(spend_input):
                    st.error("Invalid input")

                st.number_input(
                    "Percent change",
                    key=f"{channel_name}_change",
                    step=1,
                    on_change=partial(update_data_by_percent, channel_name),
                )

            with _columns[2]:
                # spends
                current_channel_spends = float(
                    _channel_class.modified_total_spends
                    * _channel_class.conversion_rate
                )
                actual_channel_spends = float(
                    _channel_class.actual_total_spends
                    * _channel_class.conversion_rate
                )
                spends_delta = float(
                    _channel_class.delta_spends
                    * _channel_class.conversion_rate
                )
                st.session_state["acutual_predicted"]["Channel_name"].append(
                    channel_name
                )
                st.session_state["acutual_predicted"]["Actual_spend"].append(
                    actual_channel_spends
                )
                st.session_state["acutual_predicted"][
                    "Optimized_spend"
                ].append(current_channel_spends)
                st.session_state["acutual_predicted"]["Delta"].append(
                    spends_delta
                )
                ## REMOVE
                st.metric(
                    "Spends",
                    format_numbers(current_channel_spends),
                    delta=numerize(spends_delta, 1),
                    label_visibility="collapsed",
                )

            with _columns[3]:
                # sales
                current_channel_sales = float(
                    _channel_class.modified_total_sales
                )
                actual_channel_sales = float(_channel_class.actual_total_sales)
                sales_delta = float(_channel_class.delta_sales)
                st.metric(
                    target,
                    format_numbers(
                        current_channel_sales, include_indicator=False
                    ),
                    delta=numerize(sales_delta, 1),
                    label_visibility="collapsed",
                )

            with _columns[4]:

                st.checkbox(
                    label="select for optimization",
                    key=f"{channel_name}_selected",
                    value=False,
                    on_change=partial(
                        select_channel_for_optimization, channel_name
                    ),
                    label_visibility="collapsed",
                )

            st.markdown(
                """<hr class="spends-child-seperator">""",
                unsafe_allow_html=True,
            )

            # Bins
            col = channels_list[i]
            x_actual = st.session_state["scenario"].channels[col].actual_spends
            x_modified = (
                st.session_state["scenario"].channels[col].modified_spends
            )

            x_total = x_modified.sum()
            power = np.ceil(np.log(x_actual.max()) / np.log(10)) - 3

            K = st.session_state["rcs"][col]["K"]
            b = st.session_state["rcs"][col]["b"]
            a = st.session_state["rcs"][col]["a"]
            x0 = st.session_state["rcs"][col]["x0"]

            x_plot = np.linspace(0, 5 * x_actual.sum(), 200)

            x, y, marginal_roi = [], [], []
            for x_p in x_plot:
                x.append(x_p * x_actual / x_actual.sum())

            for index in range(len(x_plot)):
                y.append(s_curve(x[index] / 10**power, K, b, a, x0))

            for index in range(len(x_plot)):
                marginal_roi.append(
                    a
                    * y[index]
                    * (1 - y[index] / np.maximum(K, np.finfo(float).eps))
                )

            x = (
                np.sum(x, axis=1)
                * st.session_state["scenario"].channels[col].conversion_rate
            )
            y = np.sum(y, axis=1)
            marginal_roi = (
                np.average(marginal_roi, axis=1)
                / st.session_state["scenario"].channels[col].conversion_rate
            )

            roi = y / np.maximum(x, np.finfo(float).eps)

            start_value, end_value, left_value, right_value = (
                find_segment_value(
                    x,
                    roi,
                    marginal_roi,
                )
            )

            rgba = calculate_rgba(
                start_value,
                end_value,
                left_value,
                right_value,
                current_channel_spends,
            )

            # Protecting division by zero by adding a small epsilon to denominators
            roi_current = current_channel_sales / np.maximum(
                current_channel_spends, np.finfo(float).eps
            )
            marginal_roi_current = (
                st.session_state["scenario"]
                .channels[col]
                .get_marginal_roi("modified")
            )

            with bin_placeholder:
                st.markdown(
                    f"""
                    <div style="
                        border-radius: 12px;
                        background-color: {rgba};
                        padding: 10px;
                        text-align: center;
                        color: #006EC0;
                        ">
                        <p style="margin: 0; font-size: 20px;">ROI: {round(roi_current,1)}</p>
                        <p style="margin: 0; font-size: 20px;">Marginal ROI: {round(marginal_roi_current,1)}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with st.expander("See Response Curves"):
        fig = plot_response_curves()
        st.plotly_chart(fig, use_container_width=True)

    _columns = st.columns(2)
    with _columns[0]:
        st.subheader("Save Scenario")
        scenario_name = st.text_input(
            "Scenario name",
            key="scenario_input",
            placeholder="Scenario name",
            label_visibility="collapsed",
        )
        st.button(
            "Save",
            on_click=lambda: save_scenario(scenario_name),
            disabled=len(st.session_state["scenario_input"]) == 0,
        )

    summary_df = pd.DataFrame(st.session_state["acutual_predicted"])
    summary_df.drop_duplicates(
        subset="Channel_name", keep="last", inplace=True
    )

    summary_df_sorted = summary_df.sort_values(by="Delta", ascending=False)
    summary_df_sorted["Delta_percent"] = np.round(
        (
            (
                summary_df_sorted["Optimized_spend"]
                / summary_df_sorted["Actual_spend"]
            )
            - 1
        )
        * 100,
        2,
    )

    with open("summary_df.pkl", "wb") as f:
        pickle.dump(summary_df_sorted, f)
        # st.dataframe(summary_df_sorted)
        # ___columns=st.columns(3)
        # with ___columns[2]:
        #     fig=summary_plot(summary_df_sorted, x='Delta_percent', y='Channel_name', title='Delta', text_column='Delta_percent')
        #     st.plotly_chart(fig,use_container_width=True)
        # with ___columns[0]:
        #     fig=summary_plot(summary_df_sorted, x='Actual_spend', y='Channel_name', title='Actual Spend', text_column='Actual_spend')
        #     st.plotly_chart(fig,use_container_width=True)
        # with ___columns[1]:
        #     fig=summary_plot(summary_df_sorted, x='Optimized_spend', y='Channel_name', title='Planned Spend', text_column='Optimized_spend')
        #     st.plotly_chart(fig,use_container_width=True)

elif auth_status == False:
    st.error("Username/Password is incorrect")

if auth_status != True:
    try:
        username_forgot_pw, email_forgot_password, random_password = (
            authenticator.forgot_password("Forgot password")
        )
        if username_forgot_pw:
            st.session_state["config"]["credentials"]["usernames"][
                username_forgot_pw
            ]["password"] = stauth.Hasher([random_password]).generate()[0]
            send_email(email_forgot_password, random_password)
            st.success("New password sent securely")
            # Random password to be transferred to user securely
        elif username_forgot_pw == False:
            st.error("Username not found")
    except Exception as e:
        st.error(e)
