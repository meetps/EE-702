import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
import plotly.tools as tls
tls.set_credentials_file(username='meetshah1995efe3', api_key='mxg1wama31')

a = np.load('results/r_50nr_0ns_0lambda_100.npy')

data = [go.Surface(z=a)]

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