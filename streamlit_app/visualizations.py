
import plotly.express as px 
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

action_string_to_id = {"left": 0, "right": 1, "forward": 2, "pickup": 3, "drop": 4, "toggle": 5, "done": 6}
action_id_to_string = {v: k for k, v in action_string_to_id.items()}


def plot_action_preds(action_preds):
     # make bar chart of action_preds
    action_preds = action_preds[-1][-1]
    action_preds = action_preds.detach().numpy()
    # softmax
    action_preds = np.exp(action_preds) / np.sum(np.exp(action_preds), axis=0)
    action_preds = pd.DataFrame(
        action_preds, 
        index=list(action_id_to_string.values())[:3]
        )
    st.bar_chart(action_preds)

def plot_attention_pattern(cache, layer):
    n_tokens = st.session_state.dt.n_ctx - 1
    attention_pattern = cache["pattern", layer, "attn"]
    fig = px.imshow(
        attention_pattern[:,:n_tokens,:n_tokens], 
        facet_col=0, range_color=[0,1])
    st.plotly_chart(fig)

def render_env(env):
    img = env.render()
    # use matplotlib to render the image
    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.axis('off')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    return fig