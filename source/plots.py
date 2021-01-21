"""
Styling function of plots
"""
import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import requests
# color setting
colors = plotly.colors.qualitative.Set1
color = [colors[0], colors[1], 'forestgreen']

def fig1(data, legend_list = [], xylabel=(), title=''):
    """
    Styling for figure 3
    """
    fig = make_subplots(rows = 1, cols = 1,)
    name_list = legend_list
    xlabel, ylabel = xylabel
    for i, value in enumerate(data.values()):
        fig.add_trace(go.Scatter(x = np.arange(0,200+1), y = value,
                                 mode = "lines",
                                 showlegend = True, name =name_list[i] , legendgroup = str(i), 
                                 line = dict(width = 3, color = color[i], ) ), 
                      row = 1, col = 1)


    fig.update_yaxes(
        showline=True, linewidth=2, linecolor='black', 
        title=go.layout.yaxis.Title(text=ylabel, font=dict(size=20)), 
                    )

    fig.update_xaxes(
        showline=True, linewidth=2, linecolor='black', 
    #                  range= [0,100], 
        title=go.layout.xaxis.Title(text=xlabel, font=dict(size=20))
                    )

   
    fig.update_layout(
        title={
            'text': title,
#             'y':0.9,
#             'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

    fig.update_layout( 
        legend=dict(
            x=0.8,
            y=0.95,
            font=dict(
                size=14,
                color="black"
            ),
        ),
    )
    fig.update_layout(height =500 , width = 750, 
                      template = 'seaborn',margin=dict(l=20, r=20, t=20, b=20), showlegend = True,
    #                   sliders = sliders,
                     plot_bgcolor='rgb(255,255,255)',)

    fig.show()