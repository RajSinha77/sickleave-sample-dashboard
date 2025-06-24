# Imports
import numpy as np
import pandas as pd
from scipy.stats import zscore
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import webbrowser
from threading import Timer

# Siemens corporate colors
SIEMENS_PRIMARY = '#009999'  # Siemens teal
SIEMENS_SECONDARY = '#FFFFFF'  # white
SIEMENS_ACCENT = '#FFCC00'  # Siemens accent yellow (optional accent color)
BACKGROUND_COLOR = '#f7f9fa'

# Data Generation with realistic demographic variations
np.random.seed(42)
num_employees = 80000
shape, scale = 2.0, 6.0
base_sick_days = np.random.gamma(shape, scale, num_employees)

data = pd.DataFrame({
    'base_sick_days': base_sick_days,
    'age_group': np.random.choice(['18-29','30-39','40-49','50-59','60+'], num_employees),
    'gender': np.random.choice(['Male','Female'], num_employees),
    'department': np.random.choice(['Sales','HR','IT','Finance','Operations'], num_employees),
    'location': np.random.choice(['Bavaria','Berlin','Hesse','Saxony','Hamburg','Baden-Württemberg'], num_employees),
    'day_type': np.random.choice(['on Weekday','Near Weekend','Near Holiday'], num_employees)
})

# Demographic multipliers
age_factors = {'18-29':0.9,'30-39':0.95,'40-49':1.0,'50-59':1.1,'60+':1.2}
gender_factors = {'Male':0.95,'Female':1.05}
department_factors = {'Sales':1.05,'HR':0.90,'IT':1.15,'Finance':0.85,'Operations':1.00}
location_factors = {'Bavaria':0.90,'Berlin':1.15,'Hesse':1.00,'Saxony':1.05,'Hamburg':1.10,'Baden-Württemberg':0.95}
day_type_factors = {'on Weekday':0.95,'Near Weekend':1.05,'Near Holiday':1.20}

# Apply factors
data['adjusted_sick_days'] = data.apply(lambda row: row['base_sick_days'] *
    age_factors[row['age_group']] *
    gender_factors[row['gender']] *
    department_factors[row['department']] *
    location_factors[row['location']] *
    day_type_factors[row['day_type']], axis=1)

# Monthly seasonal factors
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
monthly_factors = np.array([1.25,1.15,1.1,0.9,0.8,0.75,0.7,0.75,0.8,0.9,1.1,1.55])
monthly_pattern = monthly_factors * data['adjusted_sick_days'].mean()

# Create Dash app
app = Dash(__name__)
app.title = "Siemens Sick Leave Dashboard"

app.layout = html.Div([
    html.H1("Siemens Sick Leave Analysis Dashboard", style={'textAlign':'center', 'color':SIEMENS_PRIMARY}),
    
    dcc.Tabs([
        dcc.Tab(label='Distribution & Outliers', children=[
            html.Br(),
            dcc.Dropdown(id='outlier-method', options=['IQR', 'Z-score'], value='IQR',
                         style={'width': '300px'}),
            dcc.Graph(id='dist-graph'),
        ], style={'backgroundColor': BACKGROUND_COLOR}),

        dcc.Tab(label='Demographic Analysis', children=[
            html.Br(),
            dcc.Dropdown(id='demo-dim', options=[
                {'label': 'Age & Gender', 'value': 'age_gender'},
                {'label': 'Department', 'value': 'department'},
                {'label': 'Location', 'value': 'location'}
            ], value='age_gender', style={'width': '300px'}),
            dcc.Graph(id='demo-graph'),
        ], style={'backgroundColor': BACKGROUND_COLOR}),

        dcc.Tab(label='Seasonality Analysis', children=[
            html.Br(),
            dcc.Graph(id='seasonality-graph'),
            dcc.Interval(id='interval-component', interval=1*1000, n_intervals=0, max_intervals=1)
        ], style={'backgroundColor': BACKGROUND_COLOR}),

        dcc.Tab(label='Short vs Long-term', children=[
            html.Br(),
            dcc.Graph(id='longshort-graph'),
        ], style={'backgroundColor': BACKGROUND_COLOR}),
    ], colors={"primary": SIEMENS_PRIMARY, "background": BACKGROUND_COLOR, "border": SIEMENS_PRIMARY})
], style={'padding': '20px', 'backgroundColor': BACKGROUND_COLOR})

# Callbacks
@app.callback(Output('dist-graph', 'figure'), Input('outlier-method', 'value'))
def update_dist(method):
    fig = px.histogram(data, x='adjusted_sick_days', nbins=50,
                       title='Adjusted Sick Days Distribution',
                       color_discrete_sequence=[SIEMENS_PRIMARY])
    mean, median = data['adjusted_sick_days'].mean(), data['adjusted_sick_days'].median()
    fig.add_vline(mean, line_color='red', annotation_text='Mean', line_dash='dash')
    fig.add_vline(median, line_color='green', annotation_text='Median')

    if method == 'IQR':
        Q1, Q3 = data['adjusted_sick_days'].quantile([.25, .75])
        IQR = Q3 - Q1
        outliers = data[(data['adjusted_sick_days'] < Q1 - 1.5 * IQR) | (data['adjusted_sick_days'] > Q3 + 1.5 * IQR)]
    else:
        z_scores = zscore(data['adjusted_sick_days'])
        outliers = data[np.abs(z_scores) > 3]

    fig.add_trace(go.Histogram(x=outliers['adjusted_sick_days'],
                               marker_color=SIEMENS_ACCENT, name='Outliers'))
    fig.update_layout(barmode='overlay', plot_bgcolor=SIEMENS_SECONDARY)
    fig.update_traces(opacity=0.75)
    return fig

@app.callback(Output('demo-graph', 'figure'), Input('demo-dim', 'value'))
def update_demo(dim):
    if dim == 'age_gender':
        fig = px.box(data, x='age_group', y='adjusted_sick_days', color='gender',
                     title='Sick Days by Age & Gender',
                     color_discrete_sequence=[SIEMENS_PRIMARY, SIEMENS_ACCENT])
    else:
        avg = data.groupby(dim)['adjusted_sick_days'].mean().reset_index()
        fig = px.bar(avg, x=dim, y='adjusted_sick_days',
                     title=f'Average Sick Days by {dim.capitalize()}',
                     color_discrete_sequence=[SIEMENS_PRIMARY])
    fig.update_layout(plot_bgcolor=SIEMENS_SECONDARY)
    return fig

@app.callback(Output('seasonality-graph', 'figure'), Input('interval-component', 'n_intervals'))
def update_seasonal(n):
    fig = px.line(x=months, y=monthly_pattern, markers=True,
                  title='Seasonal Pattern of Sick Leave',
                  color_discrete_sequence=[SIEMENS_PRIMARY])
    fig.update_layout(plot_bgcolor=SIEMENS_SECONDARY)
    return fig

@app.callback(Output('longshort-graph', 'figure'), Input('longshort-graph', 'id'))
def update_longshort(_):
    short_days = np.random.poisson(6, int(0.75 * num_employees)).sum()
    long_days = np.random.normal(30, 5, int(0.25 * num_employees)).clip(min=5).sum()
    fig = px.pie(values=[short_days, long_days], names=['Short-term', 'Long-term'],
                 title='Short vs Long-term Sick Leaves',
                 color_discrete_sequence=[SIEMENS_PRIMARY, SIEMENS_ACCENT])
    return fig

# Auto-launch browser
def open_browser():
    webbrowser.open_new("http://localhost:8050/")

# Entry point
if __name__ == '__main__':
    Timer(3, open_browser).start()
    try:
        app.run(debug=True, port=8050, use_reloader=False)
    except OSError:
        print("Port 8050 busy, trying 8051")
        Timer(3, lambda: webbrowser.open_new("http://localhost:8051/")).start()
        app.run(debug=True, port=8051, use_reloader=False)
