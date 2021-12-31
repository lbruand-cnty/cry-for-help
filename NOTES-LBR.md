

## Add a keypress option.

We need a fork of streamlit that here.

Succeeding in building that fork is quite hard.

https://github.com/louislva/streamlit/tree/add-keypress


```
import streamlit as st

key = st.keypress()

if(key = "k"):
    # Do something, just like you would with a button
elif(key = "j"):
    # Do something else, because st.keypress() triggers on any key
```