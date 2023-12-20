from bokeh.io import curdoc
from bokeh.models import (
    Range1d, Circle, MultiLine, HoverTool, BoxZoomTool,
    ResetTool, LabelSet, StaticLayoutProvider, GraphRenderer,
    ColumnDataSource, Slider, Legend, LegendItem,
    DataTable, TableColumn, Div, LinearColorMapper
)
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.transform import transform
from bokeh.palettes import RdBu as palette
from typing import Dict, List, Tuple
from collections import defaultdict
from network_pdf import calculate_couplings_histogram, create_histogram_plots
from author_credit import add_author_table
import logging
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
    'EEA': '#0000FF', # '#003399'
    'ASEAN': '#00FF00', # '#00CC66'
    'Others': '#333333'
}

FREE_TRADE_AREAS = {
    'NAFTA': ['CAD', 'MXN'],  # North American Free Trade Agreement
    'EEA': ['EUR', 'CZK', 'DKK', 'NOK', 'SEK'],  # European Economic Area
    'ASEAN': ['MYR', 'PHP', 'SGD', 'THB', 'IDR'],  # Association of Southeast Asian Nations
    'Others': ['AUD', 'GBP', 'JPY', 'NZD', 'CHF', 'CNY', 'RUB', 'TRY', 'ZAR', 'HKD', 'HUF', 'ILS', 'INR', 'KRW']  # Other currencies
}

GRAPH_LAYOUT_RANGE = Range1d(-1.1, 1.1)
DATA_FILE_PATH = 'Results/J_matrix.csv'

def load_data(file_path: str) -> tuple:
    """
    Reads a CSV file and converts it into a pandas DataFrame.
    Sets the column names and index of the DataFrame.
    Converts the DataFrame into a networkx graph.

    Args:
    - file_path (str): The path to the CSV file.

    Returns:
    - df (pd.DataFrame): The DataFrame containing the data from the CSV file.
    - G (nx.Graph): The graph representation of the data.
    """
    # Read CSV file into DataFrame
    df = pd.read_csv(file_path)

    # Set column names to first three characters
    df.columns = df.columns.str[:3]

    # Set index to column names
    df.set_index(df.columns, inplace=True)

    # Convert DataFrame to networkx graph
    G = nx.from_pandas_adjacency(df)

    return df, G

def initialise_graph_components(G: nx.Graph) -> Tuple[ColumnDataSource, ColumnDataSource]:
    """
    Initializes the components required for visualising a graph.

    Args:
        G (nx.Graph): The graph object representing the network.

    Returns:
        Tuple[ColumnDataSource, ColumnDataSource]: The ColumnDataSources for the nodes and edges.
    """
    # Calculate the positions of the nodes using the spring layout algorithm
    positions = nx.spring_layout(G)

    # Get the list of nodes and assign indices to them
    nodes = list(G.nodes())
    node_indices = list(range(len(nodes)))

    # Calculate the betweenness centrality values for the nodes
    _, betweenness_values = calculate_betweenness_centrality(G, nodes)

    # Set up the color mapping for the nodes based on the free trade areas
    node_colors = setup_color_mapping(G)

    # Create the ColumnDataSource for the nodes
    x_values = []
    y_values = []
    for node in nodes:
        x_values.append(positions[node][0])
        y_values.append(positions[node][1])
    nodes_cds = ColumnDataSource({
        'index': node_indices,
        'name': nodes,
        'x': x_values,
        'y': y_values,
        'betweenness': betweenness_values,
        'color': [node_colors[node] for node in nodes]
    })

    # Create a dictionary to map node names to their corresponding indices
    node_indices_dict = dict(zip(nodes, node_indices))

    # Create the ColumnDataSource for the edges
    edges_cds = ColumnDataSource({
        'start': [node_indices_dict[edge[0]] for edge in G.edges()],
        'end': [node_indices_dict[edge[1]] for edge in G.edges()],
        'weight': [data['weight'] for _, _, data in G.edges(data=True)]
    })

    return nodes_cds, edges_cds

def calculate_betweenness_centrality(G: nx.Graph, node_names: List[str]) -> Tuple[dict, list]:
    """
    Calculate betweenness centrality (weighted degree centrality) for a given graph and node names.

    Args:
        G (nx.Graph): The graph object representing the network.
        node_names (list): List of node names.

    Returns:
        Tuple[dict, list]: A tuple containing the betweenness centrality dictionary and a list of betweenness values for the given node names.
    """
    # Calculate betweenness centrality using the nx.betweenness_centrality function
    betweenness_centrality = nx.betweenness_centrality(G, weight=lambda u, v, data: abs(data['weight']))
    
    # Extract betweenness values for the given node names
    betweenness_values = [betweenness_centrality.get(node, 0.0) for node in node_names]
    
    return betweenness_centrality, betweenness_values

def setup_color_mapping(G: nx.Graph) -> Dict[str, str]:
    """
    Sets up the color mapping for the nodes based on the free trade areas.

    Args:
        G (nx.Graph): The graph object representing the network.

    Returns:
        dict: A dictionary mapping nodes to their corresponding colors.
    """
    # Map all nodes to the color for 'Others'
    node_colors = {node: COLORS['Others'] for node in G.nodes()}

    # Map nodes in each free trade area to their corresponding color
    node_colors.update({node: COLORS[area] for area, nodes in FREE_TRADE_AREAS.items() for node in nodes})
    
    return node_colors

def create_plot(nodes_cds, edges_cds):
    """
    Create a plot using the provided nodes and edges data sources.

    Parameters:
        nodes_cds (ColumnDataSource): The data source for the nodes.
        edges_cds (ColumnDataSource): The data source for the edges.

    Returns:
        plot (Figure): The created plot.
        graph_renderer (GraphRenderer): The graph renderer used in the plot.
    """
    try:
        plot = figure(
            x_range=GRAPH_LAYOUT_RANGE, y_range=GRAPH_LAYOUT_RANGE,
            width=660, height=660, tools="", x_axis_type=None, y_axis_type=None
        )
        plot.toolbar.logo = None
        
        graph_renderer = GraphRenderer()
        graph_renderer.node_renderer.data_source = nodes_cds
        graph_renderer.node_renderer.glyph = Circle(size=10, fill_color="color")
        graph_renderer.layout_provider = StaticLayoutProvider(graph_layout=dict(zip(nodes_cds.data['index'], zip(nodes_cds.data['x'], nodes_cds.data['y']))))

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
    except Exception as e:
        logging.error(f'Error occurred: {e}')
        # Handle the error or exception here

def create_legend(plot):
    """
    Creates a legend for a Bokeh plot.

    Args:
        plot (Figure): The Bokeh plot object.

    Returns:
        legend (Legend): The created legend object.
    """
    legend_items = [
        LegendItem(label='NAFTA', renderers=[plot.circle(x=0, y=0, size=0, fill_color=COLORS['NAFTA'])]),
        LegendItem(label='EEA', renderers=[plot.circle(x=0, y=0, size=0, fill_color=COLORS['EEA'])]),
        LegendItem(label='ASEAN', renderers=[plot.circle(x=0, y=0, size=0, fill_color=COLORS['ASEAN'])]),
        LegendItem(label='Others', renderers=[plot.circle(x=0, y=0, size=0, fill_color=COLORS['Others'])]),
    ]
    legend = Legend(items=legend_items, orientation="horizontal")
    legend.label_text_font = "times"
    legend.label_text_font_style = "bold"
    legend.label_text_color = "navy"
    return legend

def create_data_table(source, width=200, height=680):
    """
    Creates a data table using the provided data source, with columns for index, currency, and betweenness centrality.

    Args:
        source (ColumnDataSource): The data source containing the data for the table.
        width (int, optional): The width of the table. Defaults to 200.
        height (int, optional): The height of the table. Defaults to 680.

    Returns:
        column: The created data table.
    """
    # Define the columns for the table
    columns = [
        TableColumn(field="index_1_based", title="#", width=50),
        TableColumn(field="currency", title="Currency"),
        TableColumn(field="betweenness", title="Betweenness"),
    ]
    
    # Create the DataTable object with the provided source and columns
    data_table = DataTable(source=source, columns=columns, width=width, height=height, index_position=None)
    
    # Return the table wrapped in a column layout with a title
    return column(Div(text="<b>Betweenness Centrality</b>"), data_table)

def create_slider(update_callback, edges_cds, positive_threshold_line, negative_threshold_line, epsilon=1e-10, step=0.01, title="Threshold"):
    """
    Create a slider and a Div to display the threshold value.

    Parameters:
    update_callback (function): The callback function to be called when the slider value changes.
    edges_cds (ColumnDataSource): The ColumnDataSource containing the edge data.
    positive_threshold_line (GlyphRenderer): The GlyphRenderer for the positive threshold line.
    negative_threshold_line (GlyphRenderer): The GlyphRenderer for the negative threshold line.
    epsilon (float, optional): The epsilon value for the slider. Defaults to 1e-10.
    step (float, optional): The step value for the slider. Defaults to 0.01.
    title (str, optional): The title for the slider. Defaults to "Threshold".

    Returns:
    tuple: A tuple containing the slider and the Div.
    """
    # Extract the weight data
    weights = np.array(edges_cds.data['weight'])
    
    # Generate a non-linear mapping function based on the weights
    mapping_func = generate_nonlinear_mapping(weights)
    
    # Set the initial value for the slider using the mapping function
    initial_value = mapping_func(0)

    # Create a slider with the specified range, step, and title
    slider = Slider(start=epsilon, end=1-epsilon, value=epsilon, step=step, title=title)

    # Create a Div element to display the threshold value
    threshold_value_div = Div(text=f"Absolute Threshold Value: {initial_value}")

    def new_update_callback(attr, old, new):
        """
        Callback function that updates the threshold value and calls the original update callback with the mapped value.

        Parameters:
        attr (str): The attribute that triggered the callback.
        old (float): The old value of the slider.
        new (float): The new value of the slider.
        """
        # Use the mapping function to get the actual threshold value
        mapped_value = mapping_func(new)
        
        # Update the Div text with the actual threshold value
        threshold_value_div.text = f"Absolute Threshold Value: {mapped_value}"
        
        # Call the original update callback with the mapped value
        update_callback(attr, old, mapped_value)

        # Update the location of the threshold lines
        positive_threshold_line.location = mapped_value
        negative_threshold_line.location = mapped_value
    
    # Bind the new callback function to the slider's value attribute
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
    # Adjust the percentiles and cdf_values to the range of absolute weights
    cdf_values = np.percentile(np.abs(weights), np.linspace(0, 100, num_points))

    # Add the minimum and maximum values to ensure the full range is covered
    percentiles = np.linspace(0, 100, len(cdf_values))
    
    def mapping_func(slider_val):
        """
        Map a value in the range (0, 1) to the weights' space.
        
        Args:
        - slider_val: float, a value in the range (0, 1).
        
        Returns:
        - weight_val: float, the mapped value in the weights' space.
        """
        # Map the slider value to a percentile
        percentile_val = slider_val * 100
        # Find the closest percentiles
        idx = np.searchsorted(percentiles, percentile_val, side='right')
        # Make sure the index is within the bounds of the CDF values
        idx = np.clip(idx, 0, len(cdf_values) - 1)
        # Interpolate between the two closest percentiles
        weight_val = np.interp(percentile_val, percentiles[idx - 1:idx + 1], cdf_values[idx - 1:idx + 1])
        return weight_val
    
    return mapping_func

def update(attr, old, new) -> None:
    """
    Update the visualization of a currency network graph based on a threshold value.

    Args:
        attr: The attribute that triggered the update.
        old: The old value of the attribute.
        new: The new value of the attribute.

    Returns:
        None. The function updates the visualization of the currency network graph and writes the new graph to a file.
    """
    threshold = new

    # Filter edges based on the absolute value of the threshold and update the edge ColumnDataSource
    selected_edges = [(start, end, weight) for start, end, weight in zip(original_edges_dataset['start'], original_edges_dataset['end'], original_edges_dataset['weight']) if abs(weight) >= threshold]
    edges_cds.data = dict(zip(['start', 'end', 'weight'], zip(*selected_edges)))

    # Construct a new graph with the filtered edges
    new_G = nx.Graph()
    new_G.add_nodes_from(nodes_cds.data['name'])
    new_G.add_edges_from([(nodes_cds.data['name'][start_idx], nodes_cds.data['name'][end_idx], {'weight': weight}) for start_idx, end_idx, weight in selected_edges])

    # Find the largest connected component
    largest_cc = max(nx.connected_components(new_G), key=len) if new_G.number_of_edges() > 0 else []
    sub_G = new_G.subgraph(largest_cc).copy()

    # Recalculate node positions based on the filtered graph
    new_positions = calculate_positions(sub_G, threshold, nodes_cds.data['name'])

    # Check if new_positions is empty and handle accordingly
    if not new_positions:
        print("Debug: new_positions is empty. Handling accordingly.")
        # Handling empty new_positions
        # For example, setting default positions or skipping updates that rely on new_positions
        return

    # Recalculate node weights based on the filtered graph
    new_weights = update_node_weights(sub_G, largest_cc)

    # Recalculate betweenness centrality
    new_bc, new_bc_values = calculate_betweenness_centrality(sub_G, nodes_cds.data['name'])

    # Create a new graph layout and update nodes_cds.data
    graph_layout = update_graph_layout(largest_cc, new_positions, new_weights, new_bc_values)

    # Update graph_renderer's layout provider with positions
    graph_renderer.layout_provider.graph_layout = graph_layout

    # Sort and filter non-zero betweenness centrality values
    sorted_filtered_bc = sorted(non_zero_betweenness(new_bc).items(), key=lambda x: x[1], reverse=True)

    # Update bc_source with the new data
    bc_source.data.update({
        'currency': [k for k, v in sorted_filtered_bc],
        'betweenness': ['{:.3f}'.format(v) for k, v in sorted_filtered_bc],
        'index_1_based': [i + 1 for i in range(len(sorted_filtered_bc))]
    })

    # Call the function to get the updated histograms based on the current J matrix
    new_positive_source, new_negative_source = calculate_couplings_histogram(J)

    # Update the plot data sources with the new histograms
    positive_plot_data_source.data.update(new_positive_source.data)
    negative_plot_data_source.data.update(new_negative_source.data)

    # Write new_graph to a file
    # nx.write_gexf(new_G, 'thresholded_graph.gexf')

def calculate_positions(G: nx.Graph, weight_threshold: float, all_nodes: List[str]) -> Dict[str, Tuple[float, float]]:
    """
    Calculates the positions of nodes in a graph based on a weight threshold.

    Args:
        G (nx.Graph): The graph object representing the network.
        weight_threshold (float): The threshold value for edge weights.
        all_nodes (list): List of all nodes in the graph.

    Returns:
        positions (dict): A dictionary mapping nodes to their positions in the graph.
    """
    # Filter the edges in the graph based on the weight threshold
    edges_to_keep = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > weight_threshold]

    # Create a subgraph with the filtered edges
    filtered_graph = G.edge_subgraph(edges_to_keep).copy()

    # Use the spring_layout algorithm to calculate the positions of nodes in the subgraph
    positions = nx.spring_layout(filtered_graph, seed=42)

    # Assign default positions to nodes that are not present in the subgraph
    for node in all_nodes:
        if node not in positions:
            positions[node] = (0, 0)  # Default position, can be adjusted as needed

    return positions

def update_node_weights(G: nx.Graph, largest_cc: List) -> List:
    """
    Update node weights based on the largest connected component in the new graph.

    Args:
    - G (nx.Graph): The graph object representing the network.
    - largest_cc (list): The largest connected component in the graph.

    Returns:
    - new_weights (list): A list of updated weights for each node in the graph.
    """
    node_weights_dict = defaultdict(list)
    edge_weights = nx.get_edge_attributes(G, 'weight')
    for u, v in G.edges():
        if u in largest_cc and v in largest_cc:
            node_weights_dict[u].append((v, edge_weights[(u, v)]))
            node_weights_dict[v].append((u, edge_weights[(u, v)]))

    new_weights = [('<br>'.join(f"{pair[0]} | {pair[1]:.4f}" for pair in sorted(node_weights_dict[node], key=lambda x: x[0]))) if node in largest_cc else '' for node in nodes_cds.data['name']]
    return new_weights

def update_graph_layout(largest_cc, new_positions, new_weights, new_bc):
    """
    Create a new graph layout and update nodes_cds.data based on the largest connected component.

    Args:
        largest_cc (list): The list of nodes in the largest connected component.
        new_positions (dict): The new positions of the nodes in the graph layout.
        new_weights (list): The new weights of the nodes.
        new_bc (dict): The new betweenness centrality values of the nodes.

    Returns:
        dict: The dictionary mapping node indices to their positions in the graph layout.
    """
    name_to_index = {name: index for index, name in enumerate(nodes_cds.data['name'])}
    graph_layout = {name_to_index[node]: new_positions[node] for node in largest_cc}
    nodes_cds.data = {
        'index': nodes_cds.data['index'],
        'name': nodes_cds.data['name'],
        'x': [graph_layout.get(name_to_index[node], (float('nan'), float('nan')))[0] if node in largest_cc else 'nan' for node in nodes_cds.data['name']],
        'y': [graph_layout.get(name_to_index[node], (float('nan'), float('nan')))[1] if node in largest_cc else 'nan' for node in nodes_cds.data['name']],
        'betweenness': new_bc,
        'weights': new_weights,
        'color': nodes_cds.data['color']
    }
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
positive_fig, negative_fig, positive_plot_data_source, negative_plot_data_source, positive_threshold_line, negative_threshold_line = create_histogram_plots()

# Create DataTable to display betweenness centrality
bc_source = ColumnDataSource({'currency': [], 'betweenness': []})
bc_table = create_data_table(bc_source)

# Setup widgets and layout
slider, threshold_value_div = create_slider(update, edges_cds, positive_threshold_line, negative_threshold_line)

# Create a Div for name and LinkedIn icon
author_table = add_author_table("Sohyun Park", "https://www.linkedin.com/in/sohyun-park-physics/", "https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png")

# Create layout
plot_layout = column(TITLE, plot)
histograms_layout = column(positive_fig, negative_fig, author_table, sizing_mode="scale_width")
stats_layout = row(bc_table, histograms_layout, sizing_mode="scale_width")
controls_layout = column(slider, threshold_value_div, stats_layout, sizing_mode="scale_width")
main_layout = row(plot_layout, controls_layout, sizing_mode="scale_width")

curdoc().title = "Currency Network"
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