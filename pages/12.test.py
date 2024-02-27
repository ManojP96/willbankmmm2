import pandas as pd
import time
import streamlit as st

count=0
def test():
    global count
    while count<5:
        time.sleep(60)
        count+=1
        st.write(count)

st.button('test',on_click=test)
