import pandas as pd
import plotly.express as px
import streamlit as st
import torch as t
from einops import rearrange
from fancy_einsum import einsum
from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT

from src.decision_transformer.utils import get_max_len_from_model_type

from .analysis import get_residual_decomp
from .constants import (
    IDX_TO_ACTION,
    IDX_TO_STATE,
    three_channel_schema,
    twenty_idx_format_func,
)
from .utils import fancy_histogram, fancy_imshow
from .visualizations import (
    plot_attention_pattern_single,
    plot_logit_diff,
    plot_dendrogram_heatmap,
)

RTG_SCAN_BATCH_SIZE = 256


def show_attention_pattern(dt, cache):
    with st.expander("Attention Pattern at at current Reward-to-Go"):
        st.latex(
            r"""
            h(x)=\left(A \otimes W_O W_V\right) \cdot x \newline
            """
        )

        st.latex(
            r"""
            A=\operatorname{softmax}\left(x^T W_Q^T W_K x\right)
            """
        )
        n_heads = dt.transformer_config.n_heads
        n_layers = dt.transformer_config.n_layers
        softmax = st.checkbox("softmax", value=True)
        heads = st.multiselect(
            "Select Heads",
            options=list(range(n_heads)),
            default=list(range(n_heads)),
            key="heads attention",
        )

        if n_layers == 1:
            plot_attention_pattern_single(
                cache, 0, softmax=softmax, specific_heads=heads
            )
        else:
            layer = st.slider(
                "Layer",
                min_value=0,
                max_value=n_layers - 1,
                value=0,
                step=1,
            )
            plot_attention_pattern_single(
                cache, layer, softmax=softmax, specific_heads=heads
            )


def show_residual_stream_contributions_single(dt, cache, logit_dir):
    with st.expander(
        "Show Residual Stream Contributions at current Reward-to-Go"
    ):
        layertab, componenttab, headtab = st.tabs(
            ["Layer", "Component", "Head"]
        )

        with layertab:
            tab1, tab2 = st.tabs(["Layerwise", "Accumulated"])
            with tab1:
                results, labels = cache.decompose_resid(
                    apply_ln=True, return_labels=True
                )
                attribution = results[:, 0, -1] @ logit_dir
                fig = px.line(
                    attribution.detach(),
                    title="Logit Difference From Residual Stream",
                    labels={"index": "Layer", "value": "Logit Difference"},
                )
                fig.update_layout(
                    hovermode="x unified",
                    showlegend=False,
                    xaxis_tickvals=list(range(len(labels))),
                    xaxis_ticktext=labels,
                    xaxis_tickangle=45,
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                results, labels = cache.accumulated_resid(
                    apply_ln=True, return_labels=True
                )
                attribution = results[:, 0, -1] @ logit_dir
                fig = px.line(
                    attribution.detach(),
                    title="Logit Difference From Accumulated Residual Stream",
                    labels={"index": "Layer", "value": "Logit Difference"},
                )
                fig.update_layout(
                    hovermode="x unified",
                    showlegend=False,
                    xaxis_tickvals=list(range(len(labels))),
                    xaxis_ticktext=labels,
                    xaxis_tickangle=45,
                )
                st.plotly_chart(fig, use_container_width=True)

        with componenttab:
            result, labels = cache.get_full_resid_decomposition(
                apply_ln=True, return_labels=True, expand_neurons=False
            )
            attribution = result[:, 0, -1] @ logit_dir
            plot_logit_diff(attribution, labels)

        with headtab:
            result, labels = cache.stack_head_results(
                apply_ln=True, return_labels=True
            )
            heads = dt.transformer_config.n_heads
            attribution = result[:, 0, -1] @ logit_dir
            k = t.topk(attribution, max(5, dt.transformer_config.n_heads))
            attribution = attribution.reshape(-1, heads)
            fig = px.imshow(
                attribution.detach(),
                color_continuous_midpoint=0,
                color_continuous_scale="RdBu",
                title="Logit Difference From Each Head",
                labels={"x": "Head", "y": "Layer"},
            )
            st.plotly_chart(fig, use_container_width=True)

            st.write(
                f"Top 5 Heads: {', '.join([f'{labels[k.indices[i]]}: {round(k.values[i].item(), 3)}' for i in range(5)])}"
            )

    return


# RTG Scan Utilities
def rtg_scan_configuration_ui(dt):
    cola, colb = st.columns(2)

    if st.session_state.allow_extrapolation:
        min_value = -10
        max_value = 10
    else:
        min_value = -1
        max_value = 1

    with cola:
        rtg_range = st.slider(
            "RTG Range",
            min_value=min_value,
            max_value=max_value,
            value=(-1, 1),
            step=1,
        )
        min_rtg = rtg_range[0]
        max_rtg = rtg_range[1]

    with colb:
        max_len = get_max_len_from_model_type(dt.model_type, dt.n_ctx)
        if "timestep_adjustment" in st.session_state:
            timesteps = (
                st.session_state.timesteps[:, -max_len:]
                + st.session_state.timestep_adjustment
            )

        if st.checkbox("add timestep noise"):
            # we want to add random integers in the range of a slider to the the timestep, the min/max on slider should be the max timesteps
            if timesteps.max().item() > 0:
                timestep_noise = st.slider(
                    "Timestep Noise",
                    min_value=1.0,
                    max_value=timesteps.max().item(),
                    value=1.0,
                    step=1.0,
                )
                timesteps = timesteps + t.randint(
                    low=int(-1 * timestep_noise),
                    high=int(timestep_noise),
                    size=timesteps.shape,
                    device=timesteps.device,
                )
            else:
                st.info(
                    "Timestep noise only works when we have more than one timestep."
                )
    return min_rtg, max_rtg, max_len, timesteps


def prepare_rtg_scan_tokens(dt, min_rtg, max_rtg, max_len, timesteps):
    batch_size = RTG_SCAN_BATCH_SIZE
    obs = st.session_state.obs[:, -max_len:].repeat(batch_size, 1, 1, 1, 1)
    actions = st.session_state.a[:, -max_len:].repeat(batch_size, 1, 1)
    rtg = st.session_state.rtg[:, -max_len:].repeat(batch_size, 1, 1)
    timesteps = (
        st.session_state.timesteps[:, -max_len:].repeat(batch_size, 1, 1)
        + st.session_state.timestep_adjustment
    )
    rtg = (
        t.linspace(min_rtg, max_rtg, batch_size)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .repeat(1, obs.shape[1], 1)
    )

    # duplicate truncation code
    obs = obs[:, -max_len:] if obs.shape[1] > max_len else obs
    if actions is not None:
        actions = (
            actions[:, -(obs.shape[1] - 1) :]
            if (actions.shape[1] > 1 and max_len > 1)
            else None
        )
    timesteps = (
        timesteps[:, -max_len:] if timesteps.shape[1] > max_len else timesteps
    )
    rtg = rtg[:, -max_len:] if rtg.shape[1] > max_len else rtg

    if dt.time_embedding_type == "linear":
        timesteps = timesteps.to(t.float32)
    else:
        timesteps = timesteps.to(t.long)

    # print out shape of each
    tokens = dt.to_tokens(obs, actions, rtg, timesteps)
    return rtg, tokens


def plot_logit_scan(rtg, action_preds):
    preds_over_rtg = {
        "RTG": rtg[:, 0, 0].detach().cpu().numpy(),
        "Left": action_preds[:, 0, 0].detach().cpu().numpy(),
        "Right": action_preds[:, 0, 1].detach().cpu().numpy(),
        "Forward": action_preds[:, 0, 2].detach().cpu().numpy(),
    }

    if action_preds.shape[-1] == 7:
        preds_over_rtg["Pickup"] = action_preds[:, 0, 3].detach().cpu().numpy()
        preds_over_rtg["Drop"] = action_preds[:, 0, 4].detach().cpu().numpy()
        preds_over_rtg["Toggle"] = action_preds[:, 0, 5].detach().cpu().numpy()
        preds_over_rtg["Done"] = action_preds[:, 0, 6].detach().cpu().numpy()

    df = pd.DataFrame(preds_over_rtg)

    # draw a line graph with left,right forward over RTG
    if action_preds.shape[-1] == 7:
        fig = px.line(
            df,
            x="RTG",
            y=[
                "Left",
                "Right",
                "Forward",
                "Pickup",
                "Drop",
                "Toggle",
                "Done",
            ],
            title="Action Prediction vs RTG",
        )
    else:
        fig = px.line(
            df,
            x="RTG",
            y=["Left", "Right", "Forward"],
            title="Action Prediction vs RTG",
        )

    fig.update_layout(
        xaxis_title="RTG",
        yaxis_title="Action Prediction",
        legend_title="",
    )
    # add vertical dotted lines at RTG = -1, RTG = 0, RTG = 1
    fig.add_vline(x=-1, line_dash="dot", line_width=1, line_color="white")
    fig.add_vline(x=0, line_dash="dot", line_width=1, line_color="white")
    fig.add_vline(x=1, line_dash="dot", line_width=1, line_color="white")

    return fig


def get_decomp_scan(rtg, cache, logit_dir, decomp_level):
    if decomp_level == "Reduced":
        results, labels = cache.decompose_resid(
            apply_ln=True, return_labels=True
        )
    elif decomp_level == "Full":
        results, labels = cache.get_full_resid_decomposition(
            apply_ln=True,
            return_labels=True,
            expand_neurons=False,  # if you don't set this, you'll crash your browser.
        )

    st.write(results.shape)
    attribution = results[:, :, -1, :] @ logit_dir

    df = pd.DataFrame(attribution.T.detach().cpu().numpy(), columns=labels)
    df.index = rtg[:, -1].squeeze(1).detach().cpu().numpy()

    return df


def plot_decomp_scan_line(df):
    fig = px.line(
        df,
        labels={"index": "RTG", "value": "Logit Difference"},
        title="Residual Stream Contributions in Directional Analysis",
    )

    fig.add_vline(x=-1, line_dash="dot", line_width=1, line_color="white")
    fig.add_vline(x=0, line_dash="dot", line_width=1, line_color="white")
    fig.add_vline(x=1, line_dash="dot", line_width=1, line_color="white")

    # # add a little more margin to the top
    fig.update_layout(margin=dict(t=50))

    return fig


def plot_decomp_scan_corr(df, cluster=False):
    if not cluster:
        fig2 = px.imshow(
            df.corr(),
            color_continuous_midpoint=0,
            title="Correlation between RTG and Residual Stream Components",
            color_continuous_scale="RdBu",
        )

    else:
        fig2 = plot_dendrogram_heatmap(df.corr())

    return fig2


def decomp_configuration_ui():
    st.write("Please note that the full decomposition is slow to compute")
    cola, colb = st.columns(2)
    with cola:
        decomp_level = st.selectbox("Decomposition Level", ["Reduced", "Full"])
    with colb:
        cluster = st.checkbox("Cluster", value=False)
    return decomp_level, cluster


def show_rtg_scan(dt, logit_dir):
    with st.expander("Scan Reward-to-Go and Show Residual Contributions"):
        min_rtg, max_rtg, max_len, timesteps = rtg_scan_configuration_ui(dt)
        rtg, tokens = prepare_rtg_scan_tokens(
            dt, min_rtg, max_rtg, max_len, timesteps
        )
        x, cache = dt.transformer.run_with_cache(
            tokens, remove_batch_dim=False
        )
        _, action_preds, _ = dt.get_logits(
            x,
            batch_size=RTG_SCAN_BATCH_SIZE,
            seq_length=max_len,
            no_actions=False,
        )

        logit_tab, decomp_tab, attn_tab = st.tabs(
            ["Logit Scan", "Decomposition", "Attention Scan"]
        )

        with logit_tab:
            fig = plot_logit_scan(rtg, action_preds)
            st.plotly_chart(fig, use_container_width=True)

        with decomp_tab:
            decomp_level, cluster = decomp_configuration_ui()
            df = get_decomp_scan(rtg, cache, logit_dir, decomp_level)
            fig = plot_decomp_scan_line(df)
            st.plotly_chart(fig, use_container_width=True)
            fig2 = plot_decomp_scan_corr(df, cluster)
            st.plotly_chart(fig2, use_container_width=True)
            if cluster:
                st.write("I know this is a bit janky, will fix later.")

        with attn_tab:
            columns = st.columns(2)
            with columns[0]:
                attention_pattern = cache["attn_scores", 0, "attn"]
                layer = st.selectbox(
                    "Layer", list(range(dt.transformer_config.n_layers))
                )
            with columns[1]:
                head = st.selectbox(
                    "Head", list(range(attention_pattern.shape[1]))
                )

            fig = px.line(
                x=t.linspace(min_rtg, max_rtg, RTG_SCAN_BATCH_SIZE),
                y=attention_pattern[:, head, 1, 0],
                title=f"Attention State to RTG for Layer {layer} Head {head}",
                labels={"x": "RTG", "y": "Attention"},
            )
            st.plotly_chart(fig, use_container_width=True)


# Observation View
def render_observation_view(dt, tokens, logit_dir):
    last_obs = st.session_state.obs[0][-1]

    last_obs_reshaped = rearrange(last_obs, "h w c -> c h w")

    height, width, n_channels = dt.environment_config.observation_space[
        "image"
    ].shape

    weights = dt.state_embedding.weight.detach().cpu()

    weights_reshaped = rearrange(
        weights, "d (c h w) -> c d h w", c=n_channels, h=height, w=width
    )

    embeddings = einsum(
        "c d h w, c h w -> c d",
        weights_reshaped,
        last_obs_reshaped.to(t.float32),
    )

    weight_projections = einsum(
        "d, c d h w -> c h w", logit_dir, weights_reshaped
    )

    activation_projection = weight_projections * last_obs_reshaped

    timesteps = st.session_state.timesteps[0][-1]
    if dt.time_embedding_type == "linear":
        timesteps = timesteps.to(t.float32)
    else:
        timesteps = timesteps.to(t.long)

    time_embedding = dt.time_embedding(timesteps)

    with st.expander("Show observation view"):
        st.subheader("Observation View")
        if n_channels == 3:

            def format_func(x):
                return three_channel_schema[x]

        else:
            format_func = twenty_idx_format_func

        selected_channels = st.multiselect(
            "Select Observation Channels",
            options=list(range(n_channels)),
            format_func=format_func,
            key="channels obs",
            default=[0, 1, 2],
        )
        n_selected_channels = len(selected_channels)

        check_columns = st.columns(4)
        with check_columns[0]:
            contributions_check = st.checkbox("Show contributions", value=True)
        with check_columns[1]:
            input_channel_check = st.checkbox(
                "Show input channels", value=True
            )
        with check_columns[2]:
            weight_proj_check = st.checkbox(
                "Show channel weight proj onto logit dir", value=True
            )
        with check_columns[3]:
            activ_proj_check = st.checkbox(
                "Show channel activation proj onto logit dir", value=True
            )

        if contributions_check:
            contributions = {
                format_func(i): (embeddings[i] @ logit_dir).item()
                for i in selected_channels
            }

            if dt.time_embedding_type == "linear":
                time_contribution = (time_embedding @ logit_dir).item()
            else:
                time_contribution = (time_embedding[0] @ logit_dir).item()

            token_contribution = (tokens[0][-1] @ logit_dir).item()

            contributions = {
                **contributions,
                "time": time_contribution,
                "token": token_contribution,
            }

            fig = px.bar(
                contributions.items(),
                x=0,
                y=1,
                labels={"0": "Channel", "1": "Contribution"},
                text=1,
            )

            # add the value to the bar
            fig.update_traces(texttemplate="%{text:.3f}", textposition="auto")
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")
            fig.update_yaxes(range=[-8, 8])
            st.plotly_chart(fig, use_container_width=True)

        if input_channel_check:
            columns = st.columns(n_selected_channels)
            for i, channel in enumerate(selected_channels):
                with columns[i]:
                    st.write(format_func(channel))
                    fancy_imshow(last_obs_reshaped[channel].detach().numpy().T)

                    if n_channels == 3:
                        if i == 0:
                            st.write(IDX_TO_OBJECT)
                        elif i == 1:
                            st.write(IDX_TO_COLOR)
                        else:
                            st.write(IDX_TO_STATE)

        if weight_proj_check:
            columns = st.columns(n_selected_channels)
            for i, channel in enumerate(selected_channels):
                with columns[i]:
                    st.write(format_func(channel))
                    fancy_imshow(
                        weight_projections[channel].detach().numpy().T
                    )
                    fancy_histogram(
                        weight_projections[channel].detach().numpy().flatten()
                    )

        if activ_proj_check:
            columns = st.columns(n_selected_channels)
            for i, channel in enumerate(selected_channels):
                with columns[i]:
                    st.write(format_func(channel))
                    fancy_imshow(
                        activation_projection[channel].detach().numpy().T
                    )
                    fancy_histogram(
                        activation_projection[channel]
                        .detach()
                        .numpy()
                        .flatten()
                    )


def project_weights_onto_dir(weights, dir):
    return t.einsum(
        "d, d h w -> h w", dir, weights.reshape(128, 7, 7)
    ).detach()
