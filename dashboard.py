import streamlit as st
import numpy as np
import requests
import plotly.graph_objects as go

st.title('Heat Equation Gaussian Process Interpolation API')
URL = "http://127.0.0.1:8000/"

T = 20
container = st.container()
container.write(r"""This project aims to interpolate heat simulations according to a set of snapshots. 
                Snapshots are approximate solutions of the heat equation (PDE) below and computed 
                simulations via the finite difference method (Cranck-Nicolson/Implicit Euler Scheme)
                for different parameters. Theoritically $u(x,t)$ can be expressed as 
                $u(x, t)=\sum_{k=0}^{+\infty}a_k(t)\varphi_k(x)$ in a infinite space. This result can be 
                approximate in a smaller subspace. So these were determined by a Proper Order Decomposition (POD).
                And finally coefficients were interpolated for experiments with other parameters by a Gaussian 
                Process regression via a Radial Basis Function kernel (RBF or squarred exponnential kernel).""")


with st.container():
    col1, col2 = st.columns(2)
    with col1:
        col1.container().write(r"""
                #### Design of Experiment (DoE) snapshots:
                * $\nu \in \{0.01, 0.03, 0.06, 0.07\}$
                * $I \in \{1, 5, 20, 50\}$
                * $T \in [|0, 19|]$""")
        st.subheader("Parameters")
        nu = col1.slider(
            r"$\nu$ ((space unit)$^2\cdot$ (time unit)$^{-1}$): thermal diffusivity",
            min_value=0.01,
            max_value=0.07,
            step=0.001
        )
        I = col1.slider(
            r"$I$ (no unit): heat source coefficient",
            min_value=1,
            max_value=50,
        )
        t = col1.slider(
            r"$t$ (time unit): time stamp",
            min_value=0.,
            max_value=19.,
            step=0.1
        )
        
    with col2:
        col2.write("#### Heat Equation:")
        col2.latex(r"\frac{\partial u}{\partial t}(x,y,t)-\nu\Delta u(x, y, t) = I\cdot f(x,y)")
        col2.container().write(r"""
                    where:
                   * $u$: temperature evaluate at position($x, y$) at time $t$
                   * $\Delta u$: laplacian of $u$
                   * $\nu$: thermal diffusivity
                   * $f(x,y)$: heat source at ($x, y$)
                   * $I$ : heat source coefficient
                   """)
        x = requests.post(
		    f"""{URL}predict""", 
		    json={
                'nu':nu,
                'I':I,
                't':t,
            }
	    )
        container = st.container(border=True)
        container.write(":red[Be aware: the results may be inaccurate especially the ones far from DoE points!]")

    mean =x.json()["mean"]
    pred_temperature_lower95 =x.json()["pred_temperature_lower95"]
    pred_temperature_upper95 =x.json()["pred_temperature_upper95"]

    xv, yv = np.meshgrid(np.linspace(0, 1, 102), np.linspace(0, 1, 102), indexing='ij')
    Z_lower = np.reshape(pred_temperature_lower95, (102, 102))
    Z_upper = np.reshape(pred_temperature_upper95, (102, 102))
    Z_mean  = np.reshape(mean, (102, 102))


    step = 4  # ou 3 ou 4

    xv_opt = xv[::step, ::step]
    yv_opt = yv[::step, ::step]
    Z_lower_opt = Z_lower[::step, ::step]
    Z_upper_opt = Z_upper[::step, ::step]
    Z_mean_opt  = Z_mean[::step, ::step]


    fig = go.Figure()
    # ---- 95% lower bound ----
    fig.add_trace(go.Surface(
        x=xv_opt,
        y=yv_opt,
        z=Z_lower_opt,
        colorscale=[[0, "blue"], [1, "blue"]],
        showscale=False,
        showlegend=True,
        opacity=0.2,
        contours=dict(
            x=dict(show=True, color="blue", width=1, start=0, end=1, size=0.1),
            y=dict(show=True, color="blue", width=1, start=0, end=1, size=0.1),
            z=dict(show=False)
        ),
        name="95% confidence interval lower bound"
    ))

    # ---- 95% upper bound ----
    fig.add_trace(go.Surface(
        x=xv_opt,
        y=yv_opt,
        z=Z_upper_opt,
        colorscale=[[0, "red"], [1, "red"]],
        showscale=False,
        opacity=0.2,
        showlegend=True,
        contours=dict(
            x=dict(show=True, color="red", width=1, start=0, end=1, size=0.1),
            y=dict(show=True, color="red", width=1, start=0, end=1, size=0.1),
            z=dict(show=False)
        ),
        name="95% confidence interval upper bound"
    ))

    # ---- Surface moyenne ----
    fig.add_trace(go.Surface(
        x=xv_opt,
        y=yv_opt,
        z=Z_mean_opt,
        colorscale=[[0, "green"], [1, "green"]],
        opacity=0.15,
        showscale=False,
        showlegend=True,
        name="mean prediction"
    ))

    # ---- Labels ----
    fig.update_layout(
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="temperature (°C)"
        ),
        legend=dict(
    orientation="h",
    yanchor="top",
    y=-0.15,
    xanchor="center",
    x=0.5 
),
margin=dict(b=120)
    )
st.plotly_chart(fig)
