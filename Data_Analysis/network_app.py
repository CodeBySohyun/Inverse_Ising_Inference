# Import necessary libraries for data handling, network analysis, and visualisation
from bokeh.io import curdoc
from bokeh.models import (Range1d, Circle, MultiLine, HoverTool, BoxZoomTool,
                          ResetTool, LabelSet, StaticLayoutProvider, GraphRenderer,
                          ColumnDataSource, Slider, Legend, LegendItem,
                          DataTable, TableColumn, Div, LinearColorMapper)
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.transform import transform
from bokeh.palettes import RdBu as palette
from decorators import debug, exception_handler
from network_pdf import calculate_couplings_histogram, create_histogram_plots
from author_credit import add_author_credit
import networkx as nx
import pandas as pd
import numpy as np

# Constants
TITLE = Div(text="""
    <div style="text-align:left;">
        <span style="font-size:16pt;"><b>Currency Network</b></span><br>
        <span style="font-size:10pt;">Data range: 2006-05-17 to 2023-10-09</span>
    </div>
    """)
TOOLTIP_TEMPLATE = """
<div>
    <div><strong>Currency:</strong> @name</div>
    <div><strong>Betweenness:</strong> @betweenness{0.000}</div>
    <div><strong>Weights:</strong></div>
    <div style="margin-left: 10px;">@weights{safe}</div>
</div>
"""
COLORS = {
    'NAFTA': '#FF0000', # '#FF5733'
    'EU': '#0000FF', # '#003399'
    'ASEAN': '#00FF00', # '#00CC66'
    'Others': '#333333'
}
FREE_TRADE_AREAS = {
    'NAFTA': ['CAD', 'MXN'],  # North American Free Trade Agreement
    'EU': ['EUR', 'DKK', 'CZK', 'SEK', 'NOK'],  # European Union
    'ASEAN': ['SGD', 'MYR', 'THB', 'PHP'],  # Association of Southeast Asian Nations
    'Others': ['AUD', 'GBP', 'JPY', 'NZD', 'CHF', 'CNY', 'RUB', 'TRY', 'ZAR']  # Other currencies
}
GRAPH_LAYOUT_RANGE = Range1d(-1.1, 1.1)
DATA_FILE_PATH = 'Results/J_matrix.csv'

def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = [col[:3] for col in df.columns]
    df.index = df.columns
    return df, nx.from_pandas_adjacency(df)

def initialise_graph_components(G):
    positions = nx.spring_layout(G)
    node_indices = list(range(len(G.nodes())))
    node_names = list(G.nodes())
    node_x = [positions[node][0] for node in node_names]
    node_y = [positions[node][1] for node in node_names]
    _, betweenness_values = calculate_betweenness_centrality(G, node_names)

    node_colors = setup_color_mapping(G)
    nodes_cds = ColumnDataSource({
        'index': node_indices, 'name': node_names, 'x': node_x, 'y': node_y,
        'betweenness': betweenness_values, 'weights': ['' for _ in node_names],
        'color': [node_colors[node] for node in node_names]
    })

    edges_cds = ColumnDataSource({
        'start': [node_indices[node_names.index(edge[0])] for edge in G.edges()],
        'end': [node_indices[node_names.index(edge[1])] for edge in G.edges()],
        'weight': [G[u][v]['weight'] for u, v in G.edges()]
    })

    return nodes_cds, edges_cds

def calculate_betweenness_centrality(G, node_names):
    """Calculate betweenness centrality for a given graph and node names."""
    # Create a temporary graph with absolute edge weights (weighted degree centrality)
    G_abs = G.copy()
    for u, v, data in G.edges(data=True):
        G_abs[u][v]['weight'] = abs(data['weight'])

    betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
    betweenness_values = [betweenness_centrality.get(node, 0.0) for node in node_names]
    return betweenness_centrality, betweenness_values

def setup_color_mapping(G):
    node_colors = {node: COLORS['Others'] for node in G.nodes()}
    for area, nodes in FREE_TRADE_AREAS.items():
        for node in nodes:
            if node in node_colors:
                node_colors[node] = COLORS[area]
    return node_colors

def create_plot(nodes_cds, edges_cds):
    plot = figure(
        x_range=GRAPH_LAYOUT_RANGE, y_range=GRAPH_LAYOUT_RANGE,
        width=660, height=660, tools="", x_axis_type=None, y_axis_type=None
    )
    plot.toolbar.logo = None
    graph_renderer = GraphRenderer()
    graph_renderer.node_renderer.data_source = nodes_cds
    graph_renderer.node_renderer.glyph = Circle(size=10, fill_color="color")
    graph_renderer.layout_provider = StaticLayoutProvider(graph_layout=dict(zip(nodes_cds.data['index'], zip(nodes_cds.data['x'], nodes_cds.data['y']))))

    # Define the colors for negative and positive weights
    custom_palette = [palette[11][1], palette[11][-2]]
    weights = edges_cds.data['weight']
    graph_renderer.edge_renderer.data_source = edges_cds
    color_mapper = LinearColorMapper(palette=custom_palette, low=-max(weights), high=max(weights))
    graph_renderer.edge_renderer.glyph = MultiLine(line_color=transform('weight', color_mapper), line_alpha=0.5, line_width=1)

    hover_tool = HoverTool(renderers=[graph_renderer.node_renderer], tooltips=TOOLTIP_TEMPLATE)
    plot.add_tools(hover_tool, BoxZoomTool(), ResetTool())
    labels = LabelSet(x='x', y='y', text='name', source=nodes_cds, text_font_size="10pt", text_color="black")
    plot.add_layout(labels)

    plot.renderers.append(graph_renderer)
    legend = create_legend(plot)
    plot.add_layout(legend, 'below')
    return plot, graph_renderer

def create_legend(plot):
    legend_items = [
        LegendItem(label='NAFTA', renderers=[plot.circle(x=0, y=0, size=0, fill_color=COLORS['NAFTA'])]),
        LegendItem(label='EU', renderers=[plot.circle(x=0, y=0, size=0, fill_color=COLORS['EU'])]),
        LegendItem(label='ASEAN', renderers=[plot.circle(x=0, y=0, size=0, fill_color=COLORS['ASEAN'])]),
        LegendItem(label='Others', renderers=[plot.circle(x=0, y=0, size=0, fill_color=COLORS['Others'])]),
    ]
    legend = Legend(items=legend_items, orientation="horizontal")
    # legend.location = "left"
    legend.label_text_font = "times"
    legend.label_text_font_style = "bold"
    legend.label_text_color = "navy"
    return legend

def create_data_table(source, width=200, height=680):
    columns = [
        TableColumn(field="index_1_based", title="#", width=50),
        TableColumn(field="currency", title="Currency"),
        TableColumn(field="betweenness", title="Betweenness"),
    ]
    data_table = DataTable(source=source, columns=columns, width=width, height=height, index_position=None)
    return column(Div(text="<b>Betweenness Centrality</b>"), data_table)

def create_slider(update_callback, edges_cds, step=0.01, title="Threshold"):
    # Create a mapping function from the weights
    weights = np.array(edges_cds.data['weight'])  # Extract the weight data
    mapping_func = generate_nonlinear_mapping(weights)
    
    # Set the initial value for the slider using the mapping function
    initial_value = mapping_func(0)

    # Create a slider that goes from 0 to 1
    epsilon = 1e-10
    slider = Slider(start=epsilon, end=1-epsilon, value=epsilon, step=0.01, title=title)

    # Create a Div to display the actual value of the threshold
    threshold_value_div = Div(text=f"Absolute Threshold Value: {initial_value}")

    # On slider change, update the threshold based on the non-linear mapping
    def new_update_callback(attr, old, new):
        # Use the mapping function to get the actual threshold value
        mapped_value = mapping_func(new)
        # Update the Div text with the actual threshold value
        threshold_value_div.text = f"Absolute Threshold Value: {mapped_value}"
        # Now call the original update callback with the mapped value
        update_callback(attr, old, mapped_value)
    
    slider.on_change('value', new_update_callback)

    return slider, threshold_value_div

def generate_nonlinear_mapping(weights, num_points=100):
    """
    Generate a non-linear mapping from a linear space (0, 1) to
    correspond to the CDF of the given weights.
    
    Args:
    - weights: array-like, the weights to base the mapping on.
    - num_points: int, the number of points to use for the mapping.
    
    Returns:
    - A function that takes a value in (0, 1) and maps it to the weights' space.
    """
    # Calculate CDF values at specified points
    percentiles = np.linspace(0, 100, num_points)
    # cdf_values = np.percentile(weights, percentiles)

    # Adjust the percentiles and cdf_values to the range of absolute weights
    cdf_values = np.percentile(abs(weights), percentiles)

    # Add the minimum and maximum values to ensure the full range is covered
    # cdf_values = np.concatenate(([min(weights)], cdf_values, [max(weights)]))
    percentiles = np.linspace(0, 100, len(cdf_values))
    
    # Non-linear mapping function
    def mapping_func(slider_val):
        # Map the slider value to a percentile
        percentile_val = slider_val * 100
        # Find the closest percentiles
        idx = np.searchsorted(percentiles, percentile_val, side='right')
        # Make sure the index is within the bounds of the CDF values
        idx = min(max(idx, 0), len(cdf_values) - 1)
        # Interpolate between the two closest percentiles
        weight_val = np.interp(percentile_val, percentiles[idx - 1:idx + 1], cdf_values[idx - 1:idx + 1])
        # Interpolate between the two closest percentiles
        # if idx < len(cdf_values) and percentiles[idx] != percentiles[idx - 1]:
        #     # Linear interpolation for a smoother transition
        #     weight_val = np.interp(percentile_val, percentiles[idx - 1:idx + 1], cdf_values[idx - 1:idx + 1])
        # else:
        #     weight_val = cdf_values[idx - 1] if idx > 0 else cdf_values[0]
        return weight_val
    
    return mapping_func

def update(attr, old, new):
    threshold = new
    # Filter edges based on the threshold and update the edge ColumnDataSource
    # selected_edges = [(start, end, weight) for start, end, weight in zip(original_edges_dataset['start'], original_edges_dataset['end'], original_edges_dataset['weight']) if weight >= threshold]
    
    # Filter edges based on the absolute value of the threshold and update the edge ColumnDataSource
    selected_edges = [(start, end, weight) for start, end, weight in zip(original_edges_dataset['start'], original_edges_dataset['end'], original_edges_dataset['weight']) if abs(weight) >= threshold]
    edges_cds.data = dict(zip(['start', 'end', 'weight'], zip(*selected_edges)))

    # Construct a new graph with the filtered edges
    new_G = nx.Graph()
    new_G.add_nodes_from(nodes_cds.data['name'])  # Nodes are added using currency names
    for start_idx, end_idx, weight in selected_edges:
        new_G.add_edge(nodes_cds.data['name'][start_idx], nodes_cds.data['name'][end_idx], weight=weight)

    # Find the largest connected component
    largest_cc = max(nx.connected_components(new_G), key=len) if new_G.number_of_edges() > 0 else []
    sub_G = new_G.subgraph(largest_cc).copy()

    # Recalculate node positions based on the filtered graph
    new_positions = calculate_positions(sub_G, threshold)

    # Check if new_positions is empty and handle accordingly
    if not new_positions:
        print("Debug: new_positions is empty. Handling accordingly.")
        # Handling empty new_positions
        # For example, setting default positions or skipping updates that rely on new_positions
        return

    # If new_positions is not empty, continue with the normal processing
    # Recalculate node weights based on the filtered graph
    new_weights = update_node_weights(sub_G, largest_cc)

    # Recalculate betweenness centrality
    new_bc, new_bc_values = calculate_betweenness_centrality(sub_G, nodes_cds.data['name'])

    # Create a new graph layout and update nodes_cds.data
    graph_layout = update_graph_layout(largest_cc, new_positions, new_weights, new_bc_values)

    # Update graph_renderer's layout provider with positions
    graph_renderer.layout_provider.graph_layout = graph_layout

    # Sort and filter non-zero betweenness centrality values
    sorted_filtered_bc = sorted(non_zero_betweenness(new_bc).items(), 
                                key=lambda x: x[1], reverse=True)
    
    # Update bc_source with the new data
    bc_source.data.update({
        'currency': [k for k, v in sorted_filtered_bc],
        'betweenness': ['{:.3f}'.format(v) for k, v in sorted_filtered_bc],
        # Now generate the 'index_1_based' list based on the length of the sorted_filtered_bc
        'index_1_based': [i + 1 for i in range(len(sorted_filtered_bc))]
    })

    # Call the function to get the updated histograms based on the current J matrix
    new_positive_source, new_negative_source = calculate_couplings_histogram(J)

    # Update the plot data sources with the new histograms
    positive_plot_data_source.data.update(new_positive_source.data)
    negative_plot_data_source.data.update(new_negative_source.data)

    # Write new_graph to a file
    nx.write_gexf(new_G, 'thresholded_graph.gexf')

def calculate_positions(G, weight_threshold):
    edges_to_keep = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > weight_threshold]
    filtered_graph = G.edge_subgraph(edges_to_keep).copy()
    return nx.spring_layout(filtered_graph, seed=42)

def update_node_weights(G, largest_cc):
    """Update node weights based on the largest connected component in the new graph."""
    node_weights_dict = {node: [] for node in nodes_cds.data['name']}
    for u, v, data in G.edges(data=True):
        if u in largest_cc and v in largest_cc:
            node_weights_dict[u].append((v, data['weight']))
            node_weights_dict[v].append((u, data['weight']))

    new_weights = ['' for _ in nodes_cds.data['name']]
    for node in nodes_cds.data['name']:
        if node in largest_cc:
            weights_list = sorted(node_weights_dict[node], key=lambda x: x[0])
            formatted_weights = '<br>'.join(f"{pair[0]} | {pair[1]:.4f}" for pair in weights_list)
            new_weights[nodes_cds.data['name'].index(node)] = formatted_weights
    return new_weights

def update_graph_layout(largest_cc, new_positions, new_weights, new_bc):
    """Create a new graph layout and update nodes_cds.data based on the largest connected component."""
    name_to_index = {name: index for index, name in enumerate(nodes_cds.data['name'])}
    graph_layout = {name_to_index[node]: new_positions[node] for node in largest_cc}
    new_data = {
        'index': nodes_cds.data['index'],
        'name': nodes_cds.data['name'],
        'x': [graph_layout.get(name_to_index[node], (float('nan'), float('nan')))[0] for node in nodes_cds.data['name']],
        'y': [graph_layout.get(name_to_index[node], (float('nan'), float('nan')))[1] for node in nodes_cds.data['name']],
        'betweenness': new_bc,
        'weights': new_weights,
        'color': nodes_cds.data['color']
    }
    nodes_cds.data = new_data
    
    # Update the ColumnDataSource with new x, y positions for nodes in the largest connected component
    nodes_cds.data['x'] = [graph_layout[name_to_index[node]][0] if node in largest_cc else 'nan' for node in nodes_cds.data['name']]
    nodes_cds.data['y'] = [graph_layout[name_to_index[node]][1] if node in largest_cc else 'nan' for node in nodes_cds.data['name']]
    nodes_cds.data['betweenness'] = new_bc
    return graph_layout

def non_zero_betweenness(betweenness_dict):
    return {k: v for k, v in betweenness_dict.items() if v > 0}

def trigger_initial_update():
    update('value', slider.value, slider.value)

# Load data and create network graph
J, G = load_data(DATA_FILE_PATH)

# Create ColumnDataSource for nodes and edges
nodes_cds, edges_cds = initialise_graph_components(G)

# Create plot, graph renderer and legend
plot, graph_renderer = create_plot(nodes_cds, edges_cds)

# Create histogram plots with specified ranges
positive_fig, negative_fig, positive_plot_data_source, negative_plot_data_source = create_histogram_plots()

# Create DataTable to display betweenness centrality
bc_source = ColumnDataSource({'currency': [], 'betweenness': []})
bc_table = create_data_table(bc_source)

# Setup widgets and layout
slider, threshold_value_div = create_slider(update, edges_cds)

plot_layout = column(TITLE, plot)
histograms_layout = column(positive_fig, negative_fig, sizing_mode="scale_width")
stats_layout = row(bc_table, histograms_layout, sizing_mode="scale_width")
controls_layout = column(slider, threshold_value_div, stats_layout, sizing_mode="scale_width")
main_layout = row(plot_layout, controls_layout, sizing_mode="scale_width")

# add_author_credit(plot, "Sohyun Park", "https://www.linkedin.com/in/sohyun-park-physics/", 10, 10)
curdoc().clear()
curdoc().add_root(main_layout)
curdoc().add_next_tick_callback(trigger_initial_update)

# original_edges_dataset stores the complete, unaltered set of edge data.
# It is used as a static reference for filtering operations in the update function.
# This ensures that the filtering of edges based on the threshold is always performed on the full dataset,
# allowing for dynamic and reversible filtering. When the threshold is adjusted,
# this dataset enables previously excluded edges to reappear if they meet the new threshold criteria.
original_edges_dataset = {
    'start': edges_cds.data['start'],
    'end': edges_cds.data['end'],
    'weight': edges_cds.data['weight']
}
# cd /Users/mizz/Library/CloudStorage/OneDrive-UniversityofBristol/Portfolio/Academic/Final_Year/Data_Analysis/Analysis
# Use the following command: bokeh serve --show network_app.py