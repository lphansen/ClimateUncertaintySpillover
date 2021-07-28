import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import plotly.offline as pyo
pyo.init_notebook_mode()
import plotly.io as pio
pio.templates.default = "none"

θ_list = pd.read_csv('data/model144.csv', header=None).to_numpy()[:, 0] / 1000.
γ_3 = np.linspace(0, 1./3, 20)
ξ_r_list = [100_000, 5., 1., 0.3]

def plot2():
    fig = go.Figure()
    θ_list = pd.read_csv('data/model144.csv', header=None).to_numpy()[:, 0] / 1000.
    trace_base = go.Histogram(
        x=θ_list * 1000,
        histnorm='probability density',
        marker=dict( color="red",line=dict(color='grey', width=1)),
        showlegend=False,
        xbins=dict(size=0.15),
        name='baseline',
        legendgroup=1,
        opacity=0.5,
    )

    fig.add_trace(trace_base)
    # fig.add_trace(trace_worstcase, 1, 1)

    fig.update_layout(
        width=600,              
        height=500,
        plot_bgcolor='white',
        title="Figure 2: Histograms of climate sensitivity parameters."
    )

    fig.update_yaxes(
        showline=True,
        showgrid=True,
        linewidth=1,
        linecolor="black",
        title="Density", 
        range=[0, 1.5], 
    )

    fig.update_xaxes(
        showline=True,
        showgrid=True,
        linewidth=1,
        linecolor='black',
        range=[0.8, 3],
        title=go.layout.xaxis.Title(text="Climate Sensitivity",
                                     font=dict(size=13)),
    )
    return fig

def plot4():
    fig = go.Figure()
    y_arr = np.linspace(0., 3., 1000)
    y_jumps = np.arange(1.5, 2 + 0.05, 0.1)

    γ1 = 0.00017675
    γ2 = 2 * 0.0022
    y_underline = 1.5
    y_overline = 2
    γ3_list = np.linspace(0, 1. / 3, 20)


    def Γs(y_jump, y_overline):
        damages = np.zeros((len(γ3_list), len(y_arr)))
        for i in range(len(γ3_list)):
            γ3_i = γ3_list[i]
            damages[i] = (γ1*y_arr + γ2/2*y_arr**2)*(y_arr<y_jump) \
            +( γ1*(y_arr- y_jump+ y_overline) + γ2/2*(y_arr-y_jump+ y_overline)**2\
              + γ3_i/2*(y_arr-y_jump)**2)*(y_arr >= y_jump)
        return damages

    for y_jump in y_jumps:
        damages = Γs(y_jump, y_overline)
        damage_upper = np.max(np.exp(-damages), axis=0)
        damage_lower = np.min(np.exp(-damages), axis=0)
        mean_damage = np.mean(np.exp(-damages), axis=0)

        fig.add_trace(
            go.Scatter(
                x=y_arr,
                y=damage_lower,
                visible=False,
                showlegend=False,
                line=dict(color="rgba(214,39,40, 0.5)"),
            ))
        fig.add_trace(
            go.Scatter(x=y_arr,
                       y=damage_upper,
                       fill='tonexty',
                       fillcolor="rgba(214,39,40, 0.3)",
                       visible=False,
                       showlegend=False,
                       line=dict(color="rgba(214,39,40, 0.5)")))
        fig.add_trace(
            go.Scatter(
                x=y_arr,
                y=mean_damage,
                visible=False,
                showlegend=False,
                line=dict(color='black'),
            ))
        fig.add_trace(
            go.Scatter(
                x=y_jump * np.ones(100),
                y=np.linspace(0.65, 1.01, 100),
                line=dict(color='black', dash="dash"),
                visible=False,
                showlegend=False,
            ))

    for i in range(4):
        fig.data[i].visible = True

    fig.update_layout(
        height=500,
        width=1000,
    )

    steps = []
    for i in range(len(y_jumps)):
        # Hide all traces
        label = r' ỹ = {:.2f}'.format(y_jumps[i])
        step = dict(method='update',
                    args=[
                        {
                            'visible': [False] * len(fig.data)
                        },
                    ],
                    label=label)
        # Enable the two traces we want to see
        for j in range(4):
            step['args'][0]["visible"][4 * i + j] = True
        # Add step to step list
        steps.append(step)

    sliders = [
        dict(
            active=0,
            currentvalue={"prefix": 'Jump threshold '},
            steps=steps,
            pad={"t": 70},
        )
    ]

    fig.update_layout(sliders=sliders, font=dict(size=13), plot_bgcolor="white")
    fig.update_xaxes(linecolor='black',
                     range=[0, 3],
                     title_text="Temperature anomaly",
                     title_font={"size": 13})
    fig.update_yaxes(range=[0.65, 1 + 0.01],
                     showline=True,
                     linecolor="black",
                     title_text="Proportional reduction in economic output",
                     title_font={"size": 13})
    # fig.write_html('fig1.html')
    return fig

def plot3():
    def J(y_arr, y_underline=1.5):
        r1 = 1.5
        r2 = 2.5
        return r1*(np.exp(r2/2*(y_arr - y_underline)**2) - 1) * (y_arr >= y_underline)

    fig = go.Figure(layout=dict(width=800, height=500, plot_bgcolor="white"))
    y_arr = np.linspace(0., 3., 1000)
    fig.add_trace(go.Scatter(x=y_arr, y=J(y_arr), line=dict(color="darkblue")))
    fig.update_xaxes(linecolor='black',
                     range=[1, 2.],
                     title_text="Temperature anomaly (ᵒC)",
                     title_font={"size":13}  
                    )
    fig.update_yaxes(
                    showline=True,
                    linecolor="black",
                    range=[-0.01, 1],
                    title_text=r"$\mathcal{J}(y)$",
                    title_font={"size":13} 
                     )
    
    return fig


def plot5(pre_jump_res):
    fig = make_subplots(rows=1, cols=1)
    trace_base = go.Histogram(
        x=θ_list * 1000,
        histnorm='probability density',
        marker=dict(color="#d62728", line=dict(color='grey', width=1)),
        showlegend=True,
        xbins=dict(size=0.15),
        name='baseline',
        legendgroup=1,
        opacity=0.5,
        hovertemplate="%{y:.2f}",
    )
    fig.add_trace(trace_base, 1, 1)

    for ξ_r in ξ_r_list:
        name = 'distorted, \n ξᵣ = {:.1f}'.format(ξ_r)
        if ξ_r == 100_000:
            name = "baseline"
        trace_worstcase = go.Histogram(
            histfunc="sum",
            x=θ_list * 1000,
            y=pre_jump_res[ξ_r]["simulation_res"]["πct"][- 1],
            histnorm='probability density',
            marker=dict(color="#1f77b4", line=dict(color='grey', width=1)),
            showlegend=False,
            visible=False,
            xbins=dict(size=0.15),
            name=name,
            legendgroup=1,
            opacity=0.5,
            hovertemplate="%{y:.2f}"
        )
        fig.add_trace(trace_worstcase, 1, 1)

    fig.data[3]['visible']=True
    fig.data[3]['showlegend']=True


    buttons = []
    for i in range(len(ξ_r_list)):
        # Hide all traces
        label = r'ξᵣ = {:.1f}'.format(ξ_r_list[i])
        if i == 0:
            label="baseline"
        button = dict(method='update',
                    args=[
                        {
                            'visible': [False] * (1 + len(ξ_r_list)),
                            'showlegend': [False] * (1 + len(ξ_r_list)),
                        },
                    ],
                    label=label)
        # Enable the two traces we want to see
        button['args'][0]["visible"][0] = True
        button['args'][0]["visible"][i + 1] = True
        button['args'][0]["showlegend"][0] = True
        button['args'][0]["showlegend"][i + 1] = True
        # Add step to step list
        buttons.append(button)

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                active=2,
                x=1.25,
                y=0.8,
                buttons=buttons,
                showactive=True
            )
        ])

    fig.update_layout(
        width=700,              
        height=500,
        plot_bgcolor='white',
        barmode="overlay",
        title="Figure 5: Histograms of climate sensitivity parameters.",
    )

    fig.update_yaxes(
        showline=True,
        showgrid=True,
        linewidth=1,
        linecolor="black",
        title="Density", 
        range=[0, 1.5], 
        col=1, 
        row=1
    )

    fig.update_xaxes(
        showline=True,
        showgrid=True,
        linewidth=1,
        linecolor='black',
        range=[0.8, 3],
        title=go.layout.xaxis.Title(text="Climate Sensitivity",
                                     font=dict(size=13)),
        row=1,
        col=1
    )
    return fig

def plot6(pre_jump_res):
    fig = make_subplots(rows=1, cols=3)
    hovertemplate = "%{y:.2f}"
    trace_base = go.Histogram(
        x=γ_3,
        histnorm='probability',
        marker=dict(color="#d62728", line=dict(color='grey', width=1)),
        showlegend=False,
        xbins=dict(start=0, end=1. / 3, size=1 / 20 / 3),
        name='baseline',
        legendgroup=1,
        opacity=0.5,
        hovertemplate=hovertemplate
    )

    trace_5 = go.Histogram(
        histfunc="sum",
        x=γ_3,
        y=pre_jump_res[5]["simulation_res"]["πdt"][-1],
        histnorm='probability',
        marker=dict(color="#1f77b4", line=dict(color='grey', width=1)),
        showlegend=False,
        xbins=dict(start=0, end=1. / 3, size=1 / 20 / 3),
        name='distorted',
        legendgroup=1,
        opacity=0.5,
        hovertemplate=hovertemplate
    )

    trace_1 = go.Histogram(
        histfunc="sum",
        x=γ_3,
        y=pre_jump_res[1]["simulation_res"]["πdt"][-1],
        histnorm='probability',
        marker=dict(color="#1f77b4", line=dict(color='grey', width=1)),
        showlegend=False,
        xbins=dict(start=0, end=1. / 3, size=1 / 20 / 3),
        name='distorted',
        legendgroup=1,
        opacity=0.5,
        hovertemplate=hovertemplate

    )

    trace_03 = go.Histogram(
        histfunc="sum",
        x=γ_3,
        y=pre_jump_res[0.3]["simulation_res"]["πdt"][-1],
        histnorm='probability',
        marker=dict(color="#1f77b4", line=dict(color='grey', width=1)),
        showlegend=False,
        xbins=dict(start=0, end=1. / 3, size=1 / 20 / 3),
        name='distorted',
        legendgroup=1,
        opacity=0.5,
        hovertemplate=hovertemplate    
    )

    for col in [1, 2, 3]:
        fig.add_trace(trace_base, 1, col)
    fig.add_trace(trace_5, 1, 1)
    fig.add_trace(trace_1, 1, 2)
    fig.add_trace(trace_03, 1, 3)

    fig.update_layout(
        title=r"""Figure 6: Distorted probabilities of damage functions <br>
        left ξᵣ = 5; center: ξᵣ = 1; right: ξᵣ = 0.3""",
        barmode="overlay", 
        plot_bgcolor="white",
        width=900,
        height=400,
        margin=dict(l=30, r=0)
    )
    fig.update_yaxes(range=[0., 0.3],
                     showline=True,
                     showgrid=True,
                     linecolor="black",
                     linewidth=1,
                    )
    fig.update_xaxes(
                     showline=True,
                     showgrid=True,
                     linecolor="black",
                     linewidth=1,
                     title = go.layout.xaxis.Title(text=r"$\gamma_3$", font=dict(size=13)),
                    )
    return fig


def plot7(pre_jump_res):
    fig = make_subplots(1, 2)
    dt = 1 / 4
    ξ_r_list = [100_000, 5., 1., 0.3]

    for ξ_r in ξ_r_list:
        yt = pre_jump_res[ξ_r]["simulation_res"]["yt"]
        T_jump = (np.abs(yt - 1.5).argmin()) * dt
        T_stop = (len(yt) - 1) * dt

        probt = np.insert(pre_jump_res[ξ_r]["simulation_res"]["probt"][:-1],
                          0,
                          0,
                          axis=0)
        Years = np.arange(0, T_stop + dt, dt)
        prob_jump = 1 - np.exp(-np.cumsum(probt))

        fig.add_trace(go.Scatter(x=Years,
                                 y=prob_jump,
                                 name="jump probability",
                                 showlegend=False,
                                 visible=False
                                ),
                      col=1,
                      row=1)
        fig.add_trace(go.Scatter(x=Years,
                                 y=yt,
                                 name="temperature anomaly",
                                 showlegend=False,
                                 visible=False,
                                 line=dict(color="#1f77b4"),
                                ),
                      col=2,
                      row=1)
        fig.add_trace(go.Scatter(x=np.arange(0, T_jump + 1),
                                 y=1.5 * np.ones(int(T_jump) + 1),
                                 showlegend=False,
                                 visible=False,
                                 line=dict(dash="dot", color="black")),
                      col=2,
                      row=1)
        fig.add_trace(go.Scatter(x=T_jump * np.ones(149),
                                 y=np.arange(0, 1.5, 0.01),
                                 showlegend=False,
                                 visible=False,
                                 line=dict(dash="dot", color="black")),
                      col=2,
                      row=1)

        fig.add_trace(go.Scatter(x=np.arange(0, T_stop + 1),
                                 y=2 * np.ones(int(T_stop) + 1),
                                 showlegend=False,
                                 visible=False,
                                 line=dict(dash="dot", color="black")),
                      col=2,
                      row=1)
        fig.add_trace(go.Scatter(x=T_stop * np.ones(199),
                                 y=np.arange(0, 2., 0.01),
                                 showlegend=False,
                                 visible=False,
                                 line=dict(dash="dot", color="black")),
                      col=2,
                      row=1)

    for i in range(6):
        fig.data[i + 6*2]["visible"] = True
    buttons = []
    for i in range(len(ξ_r_list)):
        # Hide all traces
        label = r'ξᵣ = {:.1f}'.format(ξ_r_list[i])
        if i == 0:
            label="baseline"
        button = dict(method='update',
                    args=[
                        {
                            'visible': [False] * (6* len(ξ_r_list)),
                            'showlegend': [False] * (6* len(ξ_r_list)),
                        },
                    ],
                    label=label)
        # Enable the two traces we want to see
        for j in range(6):
            button['args'][0]["visible"][6*i + j] = True
        # Add step to step list
        buttons.append(button)

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=2,
                x=0.65,
                y=1.2,
                buttons=buttons,
                pad={"r": 10,
 "t": 10, "b":10},
                showactive=True
            )
        ])


    fig.update_xaxes(showgrid=True, showline=True, title="Years", range=[0, 120])
    fig.update_yaxes(showgrid=True,
                     showline=True,
                     range=[0, 1.1],
                     col=1,
                     row=1,
                     tickformat=".1f")
    fig.update_yaxes(showgrid=True, showline=True, range=[1., 2.1], col=2, row=1)
    fig.update_yaxes(tickvals=[1, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0],
                     tickformat=".1f",
                     col=2,
                     row=1)
    # fig.update_layout(height=400, width=1280)
    fig.update_layout(
        title=r"""Figure 7:  left: jump probability; right: temperature anomaly""",
        barmode="overlay",
        plot_bgcolor="white",
        width=900,
        height=400,
        margin=dict(l=30, r=0))

    
    return fig


color = ["#d62728", "darkgreen", "darkorange", "navy"]
def plot89(pre_jump_res, y_grid_short, y_underline):
    fig = go.Figure()
    loc_11 = np.abs(y_grid_short - 1.1).argmin()
    loc_end = np.abs(y_grid_short - y_underline).argmin()

    for i, ξ_r_i in enumerate([0.3, 1, 5, 100_000]):
        if ξ_r_i == 100_000:
            name = "baseline"
        else:
            name = r"$\xi_r = {}$".format(ξ_r_i)
        fig.add_trace(
            go.Scatter(x=y_grid_short[loc_11 :loc_end + 1],
                       y=pre_jump_res[ξ_r_i]["model_res"]["e_tilde"][loc_11 :loc_end + 1],
                       name=name,
                       line=dict(color=color[i]),
                      legendgroup=name,
                      hovertemplate="%{y:.2f}"
                      ),
        )



    fig.update_yaxes(range=[0, 7], showline=True, title="Emission")
    fig.update_xaxes(showline=True, title="Temperature anomaly")
    fig.update_layout(width=800, height=500, legend=dict(traceorder="reversed"))
    return fig

def plot1011(pre_jump_res, pre_jump175_res, y_grid_short, y_underline, y_underline_higher, args_scc):
    def logSCC(y_grid, e_tilde, args=()):
        α, η, i_over_k, K0, γ_1, γ_2 = args
        C0 = (α - i_over_k) * K0
        return np.log(1000) + np.log(C0)  - (y_grid*γ_1 + γ_2/2*y_grid**2) \
            -np.log(e_tilde) + np.log(η) - np.log(1- η)


    fig = make_subplots(2,1)
    loc_11 = np.abs(y_grid_short - 1.1).argmin()
    loc_15 = np.abs(y_grid_short - y_underline).argmin()
    loc_175 = np.abs(y_grid_short - y_underline_higher).argmin()
    for i, ξ_r_i in enumerate([0.3, 1, 5, 100_000]):
        if ξ_r_i == 100_000:
            name = "baseline"
        else:
            name = r"$\xi_r = {}$".format(ξ_r_i)
        e_tilde = pre_jump_res[ξ_r_i]["model_res"]["e_tilde"][loc_11 :loc_15 + 1]
        log_SCC = logSCC(y_grid_short[loc_11 :loc_15 + 1], e_tilde, args_scc)
        fig.add_trace(go.Scatter(x=y_grid_short[loc_11 :loc_15 + 1], y=log_SCC,
                                 name=name, 
                                line=dict(color=color[i], width=2)), col=1, row=1)

    for i, ξ_r_i in enumerate([0.3, 1, 5, 100_000]):
        if ξ_r_i == 100_000:
            name = "baseline"
        else:
            name = r"$\xi_r = {}$".format(ξ_redr_i)
        e_tilde = pre_jump175_res[ξ_r_i]["model_res"]["e_tilde"][loc_11 :loc_175 + 1]
        log_SCC = logSCC(y_grid_short[loc_11 :loc_175 + 1], e_tilde, args_scc)
        fig.add_trace(go.Scatter(x=y_grid_short[loc_11 :loc_175 + 1], y=log_SCC,
                                 name=name, 
                                 showlegend=False,
                                line=dict(color=color[i], width=2)), col=1, row=2)
    fig.update_xaxes(showline=True, showgrid=True, linecolor="black")
    fig.update_yaxes(showline=True, range=[4.4, 5.7], title="Emissions")
    fig.update_layout(width=800, height=1000)
    return fig

def plot13(ratios, y_grid_long, y_underline):
    fig = go.Figure(layout=dict(width=800, height=500, plot_bgcolor="white"))
    loc_11 = np.abs(y_grid_long - 1.1).argmin()
    loc_15 = np.abs(y_grid_long - y_underline).argmin()
    colors = ["#d62728", "darkorange", "darkgreen", "navy"]
    labels = [
        "total uncertainty", "damage uncertainty", "temperature uncertainty",
        "carbon uncertainty"
    ]

    for ratio, label, color in zip(ratios, labels, colors):
        fig.add_trace(
            go.Scatter(x=y_grid_long[loc_11:loc_15 + 1],
                       y=ratio,
                       name=label,
                       line=dict(color=color)))

    fig.update_yaxes(showline=True,
                     linecolor="black",
                     range=[0, 40],
                     title="Log difference (scaled by 100)")
    fig.update_xaxes(showline=True, linecolor="black", title="Temperature anomaly")
    fig.update_layout(
        title=
        r"""Figure 13: uncertainty decomposition for the logarithm of the marginal <br>value of emissions (scaled by 100)"""
    )
    return fig


def plot14(
    tech_prob_first, distorted_tech_prob_first_7p5, distorted_tech_prob_first_5,distorted_tech_prob_first_2p5,
    tech_prob_second, distorted_tech_prob_second_7p5, distorted_tech_prob_second_5, distorted_tech_prob_second_2p5,
    dmg_prob, distorted_dmg_prob_7p5, distorted_dmg_prob_5, distorted_dmg_prob_2p5
):
    fig = make_subplots(1, 3)
    # 1
    colors = ["#d62728", "darkorange", "darkgreen", "navy"]
    fig.add_trace(go.Scatter(x=np.arange(0, 40),
                             y=tech_prob_first[:41],
                             name='baseline',
                             line=dict(width=2.5, color=colors[0])),
                  col=1,
                  row=1)
    fig.add_trace(go.Scatter(
        x=np.arange(0, 40),
        y=distorted_tech_prob_first_7p5[:41],
        name=r'$\xi_r = 7.5$',
        line=dict(width=2.5, color=colors[1]),
    ),
                  row=1,
                  col=1)

    fig.add_trace(go.Scatter(x=np.arange(0, 40),
                             y=distorted_tech_prob_first_5[:41],
                             name=r'$\xi_r = 5$',
                             line=dict(width=2.5, color=colors[2]),),
                  row=1,
                  col=1)

    fig.add_trace(go.Scatter(
        x=np.arange(0, 40),
        y=distorted_tech_prob_first_2p5[:41],
        name=r'$\xi_r = 2.5$',
        line=dict(width=2.5, color=colors[3]),   
    ),
                  row=1,
                  col=1)

    # 2

    fig.add_trace(go.Scatter(
        x=np.arange(0, 40),
        y=tech_prob_second[:41],
        name='baseline',
        showlegend=False,
        line=dict(width=2.5, color=colors[0]),

    ),
                  row=1,
                  col=2)
    fig.add_trace(go.Scatter(
        x=np.arange(0, 40),
        y=distorted_tech_prob_second_7p5[:41],
        name=r'$\xi_r = 7.5$',
          showlegend=False,
        line=dict(width=2.5, color=colors[1]),
    ),
                  row=1,
                  col=2)
    fig.add_trace(go.Scatter(
        x=np.arange(0, 40),
        y=distorted_tech_prob_second_5[:41],
        name=r'$\xi_r = 5$',
          showlegend=False,
        line=dict(width=2.5, color=colors[2]),
    ),
                  row=1,
                  col=2)
    fig.add_trace(go.Scatter(
        x=np.arange(0, 40),
        y=distorted_tech_prob_second_2p5[:41],
        name=r'$\xi_r = 2.5$',
          showlegend=False,
        line=dict(width=2.5, color=colors[3]),
    ),
                  row=1,
                  col=2)

    # 3
    fig.add_trace(go.Scatter(x=np.arange(0, 40), y=dmg_prob[:41], name='baseline',
                             showlegend=False,
        line=dict(width=2.5, color=colors[0]),
                            ),
                  row=1,
                  col=3)
    fig.add_trace(go.Scatter(x=np.arange(0, 40),
                             y=distorted_dmg_prob_7p5[:41],
                             name=r'$\xi_r = 7.5$',
                            showlegend=False,
        line=dict(width=2.5, color=colors[1]),),
                  row=1,
                  col=3)
    fig.add_trace(go.Scatter(
        x=np.arange(0, 40),
        y=distorted_dmg_prob_5[:41],
        name=r'$\xi_r = 5$',
        showlegend=False,
        line=dict(width=2.5, color=colors[2]),
    ),
                  row=1,
                  col=3)
    fig.add_trace(go.Scatter(
        x=np.arange(0, 40),
        y=distorted_dmg_prob_2p5[:41],
        name=r'$\xi_r = 2.5$',
        showlegend=False,
        line=dict(width=2.5, color=colors[3]),
    ),
                  row=1,
                  col=3)


    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=0.73
        ),
        title="""Figure 14: Distorted probability of the Poisson events for technology changes and damages<br>
        under different penalty configurations.""",
        plot_bgcolor="white",
    )

    fig.update_xaxes(showline=True, linecolor="black", title="Years")
    fig.update_yaxes(showline=True, linecolor="black", range=[0, 1])
    return fig

def plot15(θ_list, γ_3, distorted_damage_probs, πct):
    fig = make_subplots(rows=1, cols=2)
    trace_damage=go.Histogram(
        x=γ_3,
        histnorm='probability',
        marker=dict(color="#d62728", line=dict(color='grey', width=1)),
        showlegend=False,
        xbins=dict(start=0, end=1. / 3, size=1 / 20 / 3),
        name='baseline',
        legendgroup=1,
        opacity=0.5,
    )

    trace_5 = go.Histogram(
        histfunc="sum",
        x=γ_3,
        y=distorted_damage_probs[:, 40],
        histnorm='probability',
        marker=dict(color="#1f77b4", line=dict(color='grey', width=1)),
        showlegend=False,
        xbins=dict(start=0, end=1. / 3, size=1 / 20 / 3),
        name='distorted',
        legendgroup=1,
        opacity=0.5,
    )
    
    fig.add_trace(trace_damage, 1, 1)
    fig.add_trace(trace_5, 1, 1)
    
    
    trace_base = go.Histogram(
        x=θ_list * 1000,
        histnorm='probability density',
        marker=dict(color="#d62728", line=dict(color='grey', width=1)),
        showlegend=False,
        xbins=dict(size=0.15),
        name='baseline',
        legendgroup=1,
        opacity=0.5,
    )

    trace_worstcase = go.Histogram(
        histfunc="sum",
        x=θ_list * 1000,
        y=πct[:, 40],
        histnorm='probability density',
        marker=dict(color="#1f77b4", line=dict(color='grey', width=1)),
        showlegend=False,
        xbins=dict(size=0.15),
        name='1000 year',
        legendgroup=1,
        opacity=0.5,
    )

    fig.add_trace(trace_base, 1, 2)
    fig.add_trace(trace_worstcase, 1, 2)

    fig.update_layout(
#         width=600,              
#         height=500,
        plot_bgcolor='white',
        barmode="overlay",
        title="Figure 15: Distorted probabilities of damage functions and climate models. "
    )

    fig.update_yaxes(
        showline=True,
        showgrid=True,
        linewidth=1,
        linecolor="black",
        title="Probability", 
        range=[0, 0.3], 
        col=1, 
        row=1
    )

    fig.update_xaxes(
        showline=True,
        showgrid=True,
        linewidth=1,
        linecolor='black',
#         range=[0.8, 3],
        title=go.layout.xaxis.Title(text=r"$γ_3$",
                                     font=dict(size=13)),
        row=1,
        col=1
    )
    fig.update_yaxes(
        showline=True,
        showgrid=True,
        linewidth=1,
        linecolor="black",
        title="Density", 
        range=[0, 1.5], 
        col=2, 
        row=1
    )

    fig.update_xaxes(
        showline=True,
        showgrid=True,
        linewidth=1,
        linecolor='black',
        range=[0.8, 3],
        title=go.layout.xaxis.Title(text="Climate Sensitivity",
                                     font=dict(size=13)),
        row=1,
        col=2
    )
    return fig