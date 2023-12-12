from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px

from Assignments.Assignment_3.matrix import Matrix

app = Dash(__name__)

app.layout = html.Div([
    html.H1(children='Dynamic visualization of Literal Automaton', style={'textAlign': 'center'}),

    dcc.Graph(id='graph-content'),

    html.H4(children='s'),
    dcc.Slider(1, 25, step=1, value=10, tooltip={'always_visible': False}, id="slider-s"),

    html.H4(children='P(L|Y)'),
    dcc.Slider(0, 1, marks=None, value=.1, tooltip={'always_visible': False}, id="slider-ply"),

    html.H4(children='P(Y)'),
    dcc.Slider(0, 1, marks=None, value=.1, tooltip={'always_visible': False}, id="slider-py"),

    html.H4(children='P(¬L|¬Y)'),
    dcc.Slider(0, 1, marks=None, value=.1, tooltip={'always_visible': False}, id="slider-pnlny"),
])


@callback(
    Output('graph-content', 'figure'),
    Input('slider-s', 'value'),
    Input('slider-ply', 'value'),
    Input('slider-py', 'value'),
    Input('slider-pnlny', 'value'),
)
def update_graph(s, PLY, PY, PnLnY):
    matrix = Matrix(s, PLY, PY, PnLnY)

    data = {}

    for i, x in enumerate(matrix.heights):
        data[str(i + 1)] = x

    courses = list(data.keys())
    values = list(data.values())

    fig = px.bar(x=courses, y=values, labels={'x': 'state', 'y': 'probability'},
                 title='Single-literal learning probability with 8 states')
    fig.update_yaxes(range=[0, 1])
    return fig


if __name__ == '__main__':
    app.run(debug=True)
