import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px


# import data
external_stylesheets = ['mystyle.css']
raw_data_2017 = pd.read_csv('IST_North_Tower_2017_Ene_Cons.csv')
raw_data_2018 = pd.read_csv('IST_North_Tower_2018_Ene_Cons.csv')
clean_data = pd.read_csv('Clean_data.csv')
raw_meteo_data = pd.read_csv('IST_meteo_data_2017_2018_2019.csv')
regression_data = pd.read_csv('Regression_Data.csv')
error_values = pd.read_csv('Error_Values.csv')
feature_selection = pd.read_csv('Feature_Selection.csv')
eda = pd.read_csv('EDA.csv')


# figures for graphs
def graph_data(x, y, title, y_name):
    fig = go.Figure(data=go.Scatter(x=x, y=y), layout=go.Layout(paper_bgcolor='rgb(0,0,0,0)', plot_bgcolor='white'))
    fig.update_layout(title={'text': title, 'y': 0.95, 'x': 0.5}, xaxis_title="Time", yaxis_title=y_name)
    fig.update_xaxes(showline=True, linewidth=1, linecolor='#46555f', mirror=True,
                     showgrid=False)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='#46555f', mirror=True,
                     showgrid=True, gridwidth=1, gridcolor='rgb(70,85,95,0.2)')
    return fig


# graphs eda
def graph_eda(y, title):
    fig = px.box(y, labels={'value': ''})
    fig.update_layout(paper_bgcolor='rgb(0,0,0,0)', plot_bgcolor='white',
                      title={'text': title, 'y': 0.95, 'x': 0.5}, xaxis_title=" ")
    fig.update_xaxes(showline=True, linewidth=1, linecolor='#46555f', mirror=True,
                     showgrid=False)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='#46555f', mirror=True,
                     showgrid=True, gridwidth=1, gridcolor='rgb(70,85,95,0.2)')
    return fig


# graphs cluster
def graph_cluster(y, title, y_name):
    fig = px.scatter(clean_data, x="Power_kW", y=y, color="Cluster", labels={'Power_kW': 'Power (kW)', y: y_name})
    fig.update_layout(paper_bgcolor='rgb(0,0,0,0)', plot_bgcolor='white', title={'text': title, 'y': 0.95, 'x': 0.5})
    fig.update_xaxes(showline=True, linewidth=1, linecolor='#46555f', mirror=True,
                     showgrid=False)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='#46555f', mirror=True,
                     showgrid=True, gridwidth=1, gridcolor='rgb(70,85,95,0.2)')
    return fig


# graphs regression -- START
def graph_regression(y, title):
    fig = px.scatter(regression_data, x="y_test", y=y,
                     labels={'y_test': 'Real Data for Energy Consumption', y: 'Predicted Energy Consumption'})
    fig.update_layout(paper_bgcolor='rgb(0,0,0,0)', plot_bgcolor='white', title={'text': title, 'y': 0.95, 'x': 0.5})
    fig.update_xaxes(showline=True, linewidth=1, linecolor='#46555f', mirror=True,
                     showgrid=False)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='#46555f', mirror=True,
                     showgrid=True, gridwidth=1, gridcolor='rgb(70,85,95,0.2)')
    return fig


def draw_histogram(x, y, title, x_name, y_name):
    fig = px.bar(x=x, y=y, labels={'x': x_name, 'y': y_name})
    fig.update_layout(paper_bgcolor='rgb(0,0,0,0)', plot_bgcolor='white',
                      title={'text': title, 'y': 0.95, 'x': 0.5})
    fig.update_xaxes(showline=True, linewidth=1, linecolor='#46555f', mirror=True,
                     showgrid=False)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='#46555f', mirror=True,
                     showgrid=True, gridwidth=1, gridcolor='rgb(70,85,95,0.2)')
    return fig


# -- END


app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server

# create main layout
app.layout = html.Div([
    html.Div([html.Div(html.Img(src=app.get_asset_url('logo3.png'), className='img'), className='subdiv_logo'),
              html.Div(html.Center(html.H3('Forecast of energy consumption in the North Tower building of IST')),
                       className='subdiv_h'),
              html.Div([html.Button('☰', id='button_dropdown', n_clicks=0, className='dropbtn'),
                        html.Div(id='dropdown_menu', className='dropdown-content')], className='subdiv_btn')],
             className='top-bar'),
    dcc.Location(id='url', refresh=False), html.Div(id='page_layout')
])


# creation of the layouts for each of dropdown menu's page -- START
data_layout = html.Div([dcc.Tabs(id='tabs', value='raw-data', children=[
    dcc.Tab(label='Raw Data', value='raw-data', className='color-text'),
    dcc.Tab(label='Clean Data', value='clean-data', className='color-text')]),
                        html.Div(id='tabs-content')])


clean_data_layout = html.Div([
                            html.Center([dcc.RadioItems(id='radio', options=[
                                {'label': 'Energy Consumption', 'value': 'Energy Consumption'},
                                {'label': 'Temperature', 'value': 'Temperature'},
                                {'label': 'Solar Radiation', 'value': 'Solar Radiation'}
                            ],
                                value='Energy Consumption', className='color-text'
                            ),
                                         html.Div([
                                             dcc.Graph(
                                                 id='clean-data-graph',
                                                 figure=graph_data(clean_data.Day, clean_data.Power_kW,
                                                                   "Energy Consumption Clean Data", "Power (kW)")
                                             )
                                         ])])])

raw_data_layout = html.Div([html.Center([dcc.RadioItems(id='radio-raw-type', options=[
                                {'label': 'Energy Consumption 2017', 'value': 'Energy Consumption 2017'},
                                {'label': 'Energy Consumption 2018', 'value': 'Energy Consumption 2018'},
                                {'label': 'Temperature', 'value': 'Temperature'},
                                {'label': 'Solar Radiation', 'value': 'Solar Radiation'}
                            ],
                                value='Energy Consumption 2017', className='color-text'
                            ),
                            html.Div([
                                dcc.Graph(
                                    id='raw-data-graph',
                                    figure=graph_data(raw_data_2017.Date_start, raw_data_2017.Power_kW,
                                                      "Energy Consumption of 2017 Raw Data", "Power (kW)")
                                )
                            ])])])


exploratory_data_analysis_layout = html.Div(html.Center([dcc.RadioItems(id='radio-eda', options=[
                                {'label': 'Power', 'value': 'Power'},
                                {'label': 'Temperature', 'value': 'Temperature'},
                                {'label': 'Solar Radiation', 'value': 'Solar Radiation'}
                            ],
                                value='Power', className='color-text'
                            ),
                            html.Div([
                                dcc.Graph(
                                    id='show-exp-data-analysis',
                                    figure=graph_eda(eda.Power_kW,
                                                     "Visualization of Outliers in the Energy Consumption Data")
                                )
                            ])]))


clustering_layout = html.Div(html.Center([dcc.RadioItems(id='radio-clustering', options=[
                                {'label': 'Temperature', 'value': 'Temperature'},
                                {'label': 'Solar Radiation', 'value': 'Solar Radiation'},
                                {'label': 'Week Day', 'value': 'Week Day'},
                                {'label': 'Hour', 'value': 'Hour'},
                                {'label': 'Holiday', 'value': 'Holiday'},
                            ],
                                value='Temperature', className='color-text'
                            ),
                            html.Div([
                                dcc.Graph(
                                    id='clustering-graph',
                                    figure=graph_cluster("temp_C", "Clustering — Temperature vs Power",
                                                         "Temperature (ºC)")
                                )
                            ])]))


feature_selection_layout = html.Div(html.Center([dcc.RadioItems(id='feature-selection', options=[
                                {'label': 'kBest', 'value': 'kBest'},
                                {'label': 'Recursive Feature Elimination', 'value': 'Recursive Feature Elimination'},
                                {'label': 'Ensemble method', 'value': 'Ensemble method'}
                            ],
                                value='kBest', className='color-text'
                            ),
                            html.Div([
                                dcc.Graph(
                                    id='show-feature-selection',
                                    figure=draw_histogram(feature_selection.features, feature_selection.kBest,
                                                          'kBest scores for each feature', 'Features', 'Scores')
                                )
                            ])]))


regression_models_layout = html.Div([html.Div([html.Div(html.A([html.Center([html.H4('Linear Regression'),
                                                                             html.Img(src=app.get_asset_url('linear.png'),
                                                                                      className='table-img')])],
                                                               href='regression/linear'),
                                                        className='icon-regression'),
                                               html.Div(html.A([html.Center([html.H4('Support Vector Regressor'),
                                                                             html.Img(src=app.get_asset_url('svr.png'),
                                                                                      className='table-img')])],
                                                               href='/regression/svr'),
                                                        className='icon-regression'),
                                               html.Div(html.A([html.Center([html.H4('Decision Tree Regression'),
                                                                             html.Img(src=app.get_asset_url('decision-tree.png'),
                                                                                      className='table-img')])],
                                                               href='/regression/decision_tree'),
                                                        className='icon-regression'),
                                               ], className='table-row'),
                                     html.Div([html.Div(html.A([html.Center([html.H4('Random Forest'),
                                                                             html.Img(src=app.get_asset_url('random-forest.png'),
                                                                                      className='table-img')])],
                                                               href='/regression/random_forest'),
                                                        className='icon-regression'),
                                               html.Div(html.A([html.Center([html.H4('Uniformized Data'),
                                                                             html.Img(src=app.get_asset_url('uniformized-data.png'),
                                                                                      className='table-img')])],
                                                               href='/regression/uniformized_data'),
                                                        className='icon-regression'),
                                               html.Div(html.A([html.Center([html.H4('Gradient Boosting'),
                                                                             html.Img(src=app.get_asset_url('gradient-boosting.png'),
                                                                                      className='table-img')])],
                                                               href='/regression/gradient_boosting'),
                                                        className='icon-regression'),
                                               ], className='table-row'),
                                     html.Div([html.Div(html.A([html.Center([html.H4('Extreme Gradient Boosting'),
                                                                             html.Img(src=app.get_asset_url('extreme-gradient-boosting.png'),
                                                                                      className='table-img')])],
                                                               href='/regression/extreme_gradient_boosting'),
                                                        className='icon-regression'),
                                               html.Div(html.A([html.Center([html.H4('Bootstrapping'),
                                                                             html.Img(src=app.get_asset_url('bootstrapping.png'),
                                                                                      className='table-img')])],
                                                               href='/regression/bootstrapping'),
                                                        className='icon-regression'),
                                               html.Div(html.A([html.Center([html.H4('Neural Networks'),
                                                                             html.Img(src=app.get_asset_url('neural-networks.png'),
                                                                                      className='table-img')])],
                                                               href='/regression/neural_networks'),
                                                        className='icon-regression')],
                                              className='table-row')], className='regression-margin')

compare_models_layout = [html.Center([dcc.RadioItems(id='radio-comp-models', options=[
                                {'label': 'MAE', 'value': 'MAE'},
                                {'label': 'MSE', 'value': 'MSE'},
                                {'label': 'RMSE', 'value': 'RMSE'},
                                {'label': 'cvRMSE', 'value': 'cvRMSE'}
                            ],
                                value='MAE', className='color-text'
                            ),
                            dcc.Graph(
                                    id='compare-models',
                                    figure=draw_histogram(error_values.models, error_values.mae,
                                                          'MAE for each model', 'Models', 'MAE')
                                )
                            ])]

regression_layout = [dcc.Tabs(id='regression', value='models', children=[
    dcc.Tab(label='Model Type', value='models', className='color-text'),
    dcc.Tab(label='Compare the models', value='compare-models', className='color-text')]),
                        html.Div(id='regression-layout')]


about_the_author_layout = html.Table([
                        html.Tbody([html.Tr([html.Td(html.Div(html.Img(src=app.get_asset_url('photo.jpeg'),
                                                                       className='photo'), className='photo-div'),
                                                     rowSpan=3),
                                             html.Td(html.Table([html.Tr(html.Td(html.H3('Inês Andrade Rainho'))),
                                                                          html.Tr(html.Td(html.H4('IST Student Number: 90396'))),
                                                                          html.Tr(html.Td(html.H4('ines.rainho@tecnico.ulisboa.pt')))
                                                                          ])
                                                     )
                                             ])
                                    ])
])
# -- END


# callbacks
# callback for the button with dropdown menu
@app.callback(Output('dropdown_menu', 'children'),
              Input('button_dropdown', 'n_clicks'))
def toggle_dropdown(n_clicks):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'button_dropdown' in changed_id:
        return [html.A('Data', href='/data'),
                html.A('Exploratory Data Analysis', href='/exploratory_data_analysis'),
                html.A('Clustering', href='/clustering'),
                html.A('Feature Selection', href='/feature_selection'),
                html.A('Regression', href='/regression'),
                html.A('About the author', href='/about_the_author')] if n_clicks % 2 != 0 else []
    else:
        return []


@app.callback(
    Output('clean-data-graph', 'figure'),
    [Input('radio', 'value')])
def update_graph_clean(value):
    if value == 'Energy Consumption':
        return graph_data(clean_data.Day, clean_data.Power_kW,
                          "Energy Consumption Clean Data", "Power (kW)")
    elif value == 'Temperature':
        return graph_data(clean_data.Day, clean_data.temp_C,
                          "Temperature Clean Data", "Temperature (ºC)")
    elif value == 'Solar Radiation':
        return graph_data(clean_data.Day, clean_data['solarRad_W/m2'],
                          "Solar Radiation Clean Data", "Solar Radiation (W/m^2)")


@app.callback(
    Output('raw-data-graph', 'figure'),
    [Input('radio-raw-type', 'value')])
def update_graph_raw(value):
    if value == 'Energy Consumption 2017':
        return graph_data(raw_data_2017.Date_start, raw_data_2017.Power_kW,
                          "Energy Consumption of 2017 Raw Data", "Power (kW)")
    elif value == 'Energy Consumption 2018':
        return graph_data(raw_data_2018.Date_start, raw_data_2018.Power_kW,
                          "Energy Consumption of 2018 Raw Data", "Power (kW)")
    elif value == 'Temperature':
        return graph_data(raw_meteo_data['yyyy-mm-dd hh:mm:ss'], raw_meteo_data.temp_C,
                          "Temperature Raw Data", "Temperature (ºC)")
    elif value == 'Solar Radiation':
        return graph_data(raw_meteo_data['yyyy-mm-dd hh:mm:ss'], raw_meteo_data['solarRad_W/m2'],
                          "Solar Radiation Raw Data", "Solar Radiation (W/m^2)")


@app.callback(
    Output('show-exp-data-analysis', 'figure'),
    [Input('radio-eda', 'value')])
def show_eda(value):
    if value == 'Power':
        return graph_eda(eda.Power_kW, "Visualization of Outliers in the Energy Consumption Data")
    elif value == 'Temperature':
        return graph_eda(eda.temp_C, "Visualization of Outliers in the Temperature Data")
    elif value == 'Solar Radiation':
        return graph_eda(eda["solarRad_W/m2"], "Visualization of Outliers in the Solar Radiation Data")


@app.callback(
    Output('clustering-graph', 'figure'),
    [Input('radio-clustering', 'value')])
def update_graph_clustering(value):
    if value == 'Temperature':
        return graph_cluster("temp_C", "Clustering — Temperature (ºC) vs Power (kW)", "Temperature (ºC)")
    elif value == 'Solar Radiation':
        return graph_cluster("solarRad_W/m2", "Clustering — Solar Radiation (W/m^2) vs Power (kW)", "Solar Radiation (W/m^2)")
    elif value == 'Week Day':
        return graph_cluster("Week day", "Clustering — Week day vs Power (kW)", "Week day")
    elif value == 'Hour':
        return graph_cluster("Hour", "Clustering — Hour vs Power (kW)", "Hour")
    elif value == 'Holiday':
        return graph_cluster("Holiday", "Clustering — Holiday vs Power (kW)", "Holiday")


@app.callback(
    Output('show-feature-selection', 'figure'),
    [Input('feature-selection', 'value')])
def update_feature_selection(value):
    if value == 'kBest':
        return draw_histogram(feature_selection.features, feature_selection.kBest,
                              'kBest scores for each feature', 'Features', 'Scores')
    elif value == 'Recursive Feature Elimination':
        return draw_histogram(feature_selection.features, feature_selection.RFE,
                              'RFE ranking for each feature (1:Best, 7:Worst)', 'Features', 'Rankings')
    elif value == 'Ensemble method':
        return draw_histogram(feature_selection.features, feature_selection.EM,
                              'Ensemble method scores for each feature', 'Features', 'Scores')


@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'raw-data':
        return raw_data_layout
    elif tab == 'clean-data':
        return clean_data_layout


@app.callback(Output('regression-layout', 'children'),
              Input('regression', 'value'))
def content_regression(tab):
    if tab == 'models':
        return regression_models_layout
    elif tab == 'compare-models':
        return compare_models_layout


@app.callback(
    Output('compare-models', 'figure'),
    [Input('radio-comp-models', 'value')])
def update_errors_graph(value):
    if value == 'MAE':
        return draw_histogram(error_values.models, error_values.mae, 'MAE for each model', 'Models', 'MAE')
    elif value == 'MSE':
        return draw_histogram(error_values.models, error_values.mse, 'MSE for each model', 'Models', 'MSE')
    elif value == 'RMSE':
        return draw_histogram(error_values.models, error_values.rmse, 'RMSE for each model', 'Models', 'RMSE')
    elif value == 'cvRMSE':
        return draw_histogram(error_values.models, error_values.cvrmse, 'cvRMSE for each model', 'Models', 'cvRMSE')


def model_layout(figure, table):
    return [html.Center([dcc.Graph(figure=figure),
            html.Table([html.Thead(html.Tr([html.Th('MAE'), html.Th('MSE'), html.Th('RMSE'), html.Th('cvRMSE')])),
                        html.Tbody([html.Tr([html.Td(row) for row in error_values.iloc[table, 2:]])
                                    ])
                        ])])]


@app.callback(Output('page_layout', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/exploratory_data_analysis':
        return exploratory_data_analysis_layout
    elif pathname == '/clustering':
        return clustering_layout
    elif pathname == '/feature_selection':
        return feature_selection_layout
    elif pathname == '/regression':
        return regression_layout
    elif pathname == '/about_the_author':
        return about_the_author_layout
    elif pathname == '/regression/linear':
        return model_layout(graph_regression("LR", "Linear Regression Results"), 0)
    elif pathname == '/regression/svr':
        return model_layout(graph_regression("SVR", "Support Vector Regressor Results"), 1)
    elif pathname == '/regression/decision_tree':
        return model_layout(graph_regression("DT", "Decision Tree Results"), 2)
    elif pathname == '/regression/random_forest':
        return model_layout(graph_regression("RF", "Random Forest Results"), 3)
    elif pathname == '/regression/uniformized_data':
        return model_layout(graph_regression("URF", "Uniformized Random Forest Results"), 4)
    elif pathname == '/regression/gradient_boosting':
        return model_layout(graph_regression("GB", "Gradient Boosting Results"), 5)
    elif pathname == '/regression/extreme_gradient_boosting':
        return model_layout(graph_regression("XGB", "Extreme Gradient Boosting Results"), 6)
    elif pathname == '/regression/bootstrapping':
        return model_layout(graph_regression("BT", "Bootstrapping Results"), 7)
    elif pathname == '/regression/neural_networks':
        return model_layout(graph_regression("NN", "Neural Networks Results"), 8)
    else:
        return data_layout


if __name__ == '__main__':
    app.run_server(debug=True)
