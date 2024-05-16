import os
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pyvista as pv
import volman
import preprocessing
from volman import the_index

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.Div([
        dbc.Button("Toggle Options", id="collapse-button", className="mb-3", color="primary"),
        dbc.Collapse(
            dbc.Card(
                dbc.CardBody([
                    html.Div([
                        html.Label('Scroll:', className='mr-2'),
                        dcc.Dropdown(id='scroll-dropdown',
                                     options=[{'label': scroll, 'value': scroll} for scroll in the_index.keys()],
                                     className='d-inline-block w-75')
                    ], className='mb-1'),
                    html.Div([
                        html.Label('Source:', className='mr-2'),
                        dcc.Dropdown(id='source-dropdown', options=[], className='d-inline-block w-75')
                    ], className='mb-1'),
                    html.Div([
                        html.Label('ID Number:', className='mr-2'),
                        dcc.Dropdown(id='idnum-dropdown', options=[], className='d-inline-block w-75')
                    ], className='mb-1'),
                    html.Div([
                        html.Div([html.Label('Chunk Size (Y, X, Z):', className='mr-2'), ], className='mb-1'),
                        dcc.Input(id='chunk-size-y', type='number', value=128, className='d-inline-block w-25 mr-1'),
                        dcc.Input(id='chunk-size-x', type='number', value=128, className='d-inline-block w-25 mr-1'),
                        dcc.Input(id='chunk-size-z', type='number', value=128, className='d-inline-block w-25')
                    ], className='mb-1'),
                    html.Div([
                        html.Div([html.Label('Chunk Offset (Y, X, Z):', className='mr-2'), ], className='mb-1'),
                        dcc.Input(id='chunk-offset-y', type='number', value=1000, className='d-inline-block w-25 mr-1'),
                        dcc.Input(id='chunk-offset-x', type='number', value=1000, className='d-inline-block w-25 mr-1'),
                        dcc.Input(id='chunk-offset-z', type='number', value=1000, className='d-inline-block w-25')
                    ], className='mb-1'),
                    html.Div([
                        html.Label('Output Directory:', className='mr-2'),
                        dcc.Input(id='outdir-input', type='text', value='generated_ply',
                                  className='d-inline-block w-75')
                    ], className='mb-1'),
                    html.Div([
                        html.Label('Ink Labels Path:', className='mr-2'),
                        dcc.Input(id='inklabels-path-input', type='text', value='', className='d-inline-block w-75')
                    ], className='mb-1'),
                    html.Div([
                        dcc.Checklist(
                            id='options-checklist',
                            options=[
                                {'label': 'Preprocess', 'value': 'preprocess'},
                                {'label': 'Superpixel', 'value': 'superpixel'},
                                {'label': 'Postprocess', 'value': 'postprocess'}
                            ],
                            value=['preprocess', 'postprocess'],
                            labelStyle={'display': 'inline-block'}
                        )
                    ], className='mb-1'),
                    html.Div([
                        html.Label('Downsample:', className='mr-1'),
                        dcc.Input(id='downsample-input', type='number', value=1, className='d-inline-block w-25'),
                    ], className='mb-1'),
                    html.Div([
                        html.Label('Isolevel:', className='mr-1'),
                        dcc.Slider(
                            id='isolevel-input',
                            min=0,
                            max=255,
                            step=1,
                            value=128,
                            marks={i: str(i) for i in range(0, 256, 25)},
                            className='d-inline-block w-100'
                        ),
                    ], className='mb-1'),
                    html.Div([
                        html.Label('Colormap:', className='mr-1'),
                        dcc.Input(id='colormap-input', type='text', value='viridis', className='d-inline-block w-50'),
                        html.Label('Quantize:', className='mr-1 ml-2'),
                        dcc.Input(id='quantize-input', type='number', value=8, className='d-inline-block w-25')
                    ], className='mb-2'),
                    html.Div([
                        html.Button('Generate Mesh', id='generate-button', n_clicks=0,
                                    className='btn btn-primary btn-block')
                    ], className='mt-3')
                ])
            ),
            id="collapse",
            is_open=False
        )
    ], style={'position': 'absolute', 'top': '10px', 'left': '10px', 'width': '300px', 'z-index': '1000',
              'background-color': 'rgba(255, 255, 255, 0.8)', 'padding': '10px', 'border-radius': '5px'}),

    # Placeholder for the PyVista plot
    html.Div(id='pv-plot', className='h-100')
], fluid=True, className='h-100')


@app.callback(
    Output('source-dropdown', 'options'),
    Input('scroll-dropdown', 'value')
)
def update_source_options(selected_scroll):
    if selected_scroll:
        return [{'label': source, 'value': source} for source in the_index[selected_scroll].keys()]
    return []


@app.callback(
    Output('idnum-dropdown', 'options'),
    Input('scroll-dropdown', 'value'),
    Input('source-dropdown', 'value')
)
def update_idnum_options(selected_scroll, selected_source):
    if selected_scroll and selected_source:
        return [{'label': idnum, 'value': idnum} for idnum in the_index[selected_scroll][selected_source]]
    return []


@app.callback(
    Output('pv-plot', 'children'),
    Input('generate-button', 'n_clicks'),
    [State('scroll-dropdown', 'value'),
     State('source-dropdown', 'value'),
     State('idnum-dropdown', 'value'),
     State('chunk-size-y', 'value'),
     State('chunk-size-x', 'value'),
     State('chunk-size-z', 'value'),
     State('chunk-offset-y', 'value'),
     State('chunk-offset-x', 'value'),
     State('chunk-offset-z', 'value'),
     State('outdir-input', 'value'),
     State('inklabels-path-input', 'value'),
     State('options-checklist', 'value'),
     State('downsample-input', 'value'),
     State('isolevel-input', 'value'),
     State('colormap-input', 'value'),
     State('quantize-input', 'value')]
)
def update_mesh(n_clicks, scroll, source, idnum, chunk_size_y, chunk_size_x, chunk_size_z,
                chunk_offset_y, chunk_offset_x, chunk_offset_z, outdir, inklabels_path,
                options, downsample, isolevel, colormap, quantize_):
    if n_clicks > 0:
        chunk_size = (chunk_size_y, chunk_size_x, chunk_size_z)
        chunk_offset = (chunk_offset_y, chunk_offset_x, chunk_offset_z)
        preprocess_ = 'preprocess' in options
        superpixel_ = 'superpixel' in options
        postprocess_ = 'postprocess' in options
        inklabels_path = inklabels_path if inklabels_path else None

        chunk = load_chunk(scroll, source, idnum, chunk_size, chunk_offset, outdir,
                           inklabels_path, preprocess_, superpixel_, downsample, isolevel,
                           postprocess_, colormap, quantize_)

        # Use PyVista to create the plot
        plot_div = visualize_voxels(chunk, colormap)
        return plot_div

    return html.Div()


@app.callback(
    Output("collapse", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


def load_chunk(scroll, source, idnum, chunk_size, chunk_offset, outdir,
               inklabels_path, preprocess_, superpixel_, downsample, isolevel,
               postprocess_, colormap, quantize_):
    print("Stacking tiffs")
    vm = volman.VolMan('.')
    yoff, xoff, zoff = chunk_offset
    ysize, xsize, zsize = chunk_size
    chunk = vm.chunk(scroll, source, idnum, yoff, xoff, zoff, ysize, xsize, zsize)

    return chunk


def visualize_voxels(chunk, colormap):
    plotter = pv.Plotter(off_screen=True)

    # Create the grid
    x = np.arange(chunk.shape[0])
    y = np.arange(chunk.shape[1])
    z = np.arange(chunk.shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = chunk.shape

    # Assign the voxel data to the grid
    grid["values"] = chunk.ravel(order="F")

    # Add the volume and show
    plotter.add_volume(grid, cmap=colormap)
    plotter.show(screenshot='screenshot.png')

    return html.Img(src='screenshot.png', style={'width': '100%', 'height': '100%'})


if __name__ == '__main__':
    app.run_server(debug=True)
