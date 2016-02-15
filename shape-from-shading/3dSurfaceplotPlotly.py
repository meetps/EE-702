import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np


data = [go.Surface()]

layout = go.Layout(
    title='Shape from Shading',
    autosize=False,
    width=500,
    height=500,
    margin=dict(
        l=65,
        r=50,
        b=65,
        t=90
    )
)
fig = go.Figure(data=data, layout=layout)
plot_url = py.plot(fig, filename='elevations-3d-surface')