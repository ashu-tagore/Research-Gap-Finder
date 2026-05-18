import dash
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

app.layout = dbc.Container("Research Gap Finder — coming soon")

if __name__ == "__main__":
    app.run(debug=True)