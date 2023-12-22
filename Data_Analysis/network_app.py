from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
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
from collections import defaultdict
from network_pdf import calculate_couplings_histogram, create_histogram_plots
import logging
import networkx as nx
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE_PATH = os.path.join(BASE_DIR, 'Results', 'J_matrix.csv')
author_name = "Sohyun Park"
website_url = "https://www.linkedin.com/in/sohyuniverse"
icon_url = "https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png"

TITLE = Div(text="""
    <div style="text-align:left;">
        <span style="font-size:16pt;"><b>PLM Currency Network</b></span><br>
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

class CurrencyNetworkApp:
    def __init__(self):
        try:
            print("Initialising CurrencyNetworkApp instance...")

            print("Loading data...")
            self.J, self.G = self.load_data(DATA_FILE_PATH)

            print("Initialising graph components...")
            self.nodes_cds, self.edges_cds = self.initialise_graph_components(self.G)

            print("Creating plot...")
            self.plot, self.graph_renderer = self.create_plot(self.nodes_cds, self.edges_cds)

            print("Creating histogram plots...")
            self.positive_fig, self.negative_fig, self.positive_plot_data_source, \
            self.negative_plot_data_source, self.positive_threshold_line, \
            self.negative_threshold_line = create_histogram_plots()

            print("Creating betweenness centrality table...")
            self.bc_source = ColumnDataSource({'currency': [], 'betweenness': []})
            self.bc_table = self.create_data_table(self.bc_source)

            print("Creating slider...")
            self.slider, self.threshold_value_div = self.create_slider(self.update, self.edges_cds, 
                                                                       self.positive_threshold_line, 
                                                                       self.negative_threshold_line)

            self.original_edges_dataset = {
                'start': self.edges_cds.data['start'],
                'end': self.edges_cds.data['end'],
                'weight': self.edges_cds.data['weight']
            }
        except Exception as e:
            print(f"Error in CurrencyNetworkApp initialisation: {e}")

    @staticmethod
    def load_data(file_path):
        df = pd.read_csv(file_path)
        df.columns = df.columns.str[:3]
        df.set_index(df.columns, inplace=True)
        G = nx.from_pandas_adjacency(df)
        return df, G

    def initialise_graph_components(self, G):
        positions = nx.spring_layout(G)
        nodes = list(G.nodes())
        node_indices = list(range(len(nodes)))
        _, betweenness_values = self.calculate_betweenness_centrality(G, nodes)
        node_colors = self.setup_color_mapping(G)
        x_values, y_values = zip(*[positions[node] for node in nodes])
        nodes_cds = ColumnDataSource({
            'index': node_indices,
            'name': nodes,
            'x': x_values,
            'y': y_values,
            'betweenness': betweenness_values,
            'color': [node_colors[node] for node in nodes]
        })
        node_indices_dict = dict(zip(nodes, node_indices))
        edges_cds = ColumnDataSource({
            'start': [node_indices_dict[edge[0]] for edge in G.edges()],
            'end': [node_indices_dict[edge[1]] for edge in G.edges()],
            'weight': [data['weight'] for _, _, data in G.edges(data=True)]
        })
        return nodes_cds, edges_cds

    @staticmethod
    def calculate_betweenness_centrality(G, node_names):
        betweenness_centrality = nx.betweenness_centrality(G, weight=lambda u, v, data: abs(data['weight']))
        betweenness_values = [betweenness_centrality.get(node, 0.0) for node in node_names]
        return betweenness_centrality, betweenness_values
    
    @staticmethod
    def setup_color_mapping(G):
        node_colors = {node: COLORS['Others'] for node in G.nodes()}
        node_colors.update({node: COLORS[area] for area, nodes in FREE_TRADE_AREAS.items() for node in nodes})
        return node_colors

    def create_plot(self, nodes_cds, edges_cds):
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
            legend = self.create_legend(plot)
            plot.add_layout(legend, 'below')
            return plot, graph_renderer
        except Exception as e:
            logging.error(f'Error occurred: {e}')
            # Handle the error or exception here

    @staticmethod
    def create_legend(plot):
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
    
    @staticmethod
    def create_data_table(source, width=200, height=680):
        columns = [
            TableColumn(field="index_1_based", title="#", width=50),
            TableColumn(field="currency", title="Currency"),
            TableColumn(field="betweenness", title="Betweenness"),
        ]
        data_table = DataTable(source=source, columns=columns, width=width, height=height, index_position=None)
        return column(Div(text="<b>Betweenness Centrality</b>"), data_table)

    def create_slider(self, update_callback, edges_cds, positive_threshold_line, negative_threshold_line, epsilon=1e-10, step=0.01, title="Threshold"):
        weights = np.array(edges_cds.data['weight'])
        mapping_func = self.generate_nonlinear_mapping(weights)
        initial_value = mapping_func(0)
        slider = Slider(start=epsilon, end=1-epsilon, value=epsilon, step=step, title=title)
        threshold_value_div = Div(text=f"Absolute Threshold Value: {initial_value}")

        def new_update_callback(attr, old, new):
            mapped_value = mapping_func(new)
            threshold_value_div.text = f"Absolute Threshold Value: {mapped_value}"
            update_callback(attr, old, mapped_value)
            positive_threshold_line.location = mapped_value
            negative_threshold_line.location = mapped_value

        slider.on_change('value', new_update_callback)
        return slider, threshold_value_div

    @staticmethod
    def generate_nonlinear_mapping(weights, num_points=100):
        cdf_values = np.percentile(np.abs(weights), np.linspace(0, 100, num_points))
        percentiles = np.linspace(0, 100, len(cdf_values))

        def mapping_func(slider_val):
            percentile_val = slider_val * 100
            idx = np.searchsorted(percentiles, percentile_val, side='right')
            idx = np.clip(idx, 0, len(cdf_values) - 1)
            weight_val = np.interp(percentile_val, percentiles[idx - 1:idx + 1], cdf_values[idx - 1:idx + 1])
            return weight_val

        return mapping_func

    def update(self, attr, old, new):
        threshold = new

        # Filter edges based on the absolute value of the threshold and update the edge ColumnDataSource
        selected_edges = [(start, end, weight) for start, end, weight in zip(self.original_edges_dataset['start'], self.original_edges_dataset['end'], self.original_edges_dataset['weight']) if abs(weight) >= threshold]
        self.edges_cds.data = dict(zip(['start', 'end', 'weight'], zip(*selected_edges)))

        # Construct a new graph with the filtered edges
        new_G = nx.Graph()
        new_G.add_nodes_from(self.nodes_cds.data['name'])
        new_G.add_edges_from([(self.nodes_cds.data['name'][start_idx], self.nodes_cds.data['name'][end_idx], {'weight': weight}) for start_idx, end_idx, weight in selected_edges])

        # Find the largest connected component
        largest_cc = max(nx.connected_components(new_G), key=len) if new_G.number_of_edges() > 0 else []
        sub_G = new_G.subgraph(largest_cc).copy()

        # Recalculate node positions based on the filtered graph
        new_positions = self.calculate_positions(sub_G, threshold, self.nodes_cds.data['name'])

        # Check if new_positions is empty and handle accordingly
        if not new_positions:
            print("Debug: new_positions is empty. Handling accordingly.")
            # Handling empty new_positions
            # For example, setting default positions or skipping updates that rely on new_positions
            return

        # Recalculate node weights based on the filtered graph
        new_weights = self.update_node_weights(sub_G, largest_cc)

        # Recalculate betweenness centrality
        new_bc, new_bc_values = self.calculate_betweenness_centrality(sub_G, self.nodes_cds.data['name'])

        # Create a new graph layout and update nodes_cds.data
        graph_layout = self.update_graph_layout(largest_cc, new_positions, new_weights, new_bc_values)

        # Update graph_renderer's layout provider with positions
        self.graph_renderer.layout_provider.graph_layout = graph_layout

        # Sort and filter non-zero betweenness centrality values
        sorted_filtered_bc = sorted(self.non_zero_betweenness(new_bc).items(), key=lambda x: x[1], reverse=True)

        # Update bc_source with the new data
        self.bc_source.data.update({
            'currency': [k for k, v in sorted_filtered_bc],
            'betweenness': ['{:.3f}'.format(v) for k, v in sorted_filtered_bc],
            'index_1_based': [i + 1 for i in range(len(sorted_filtered_bc))]
        })

        # Call the function to get the updated histograms based on the current J matrix
        new_positive_source, new_negative_source = calculate_couplings_histogram(self.J)

        # Update the plot data sources with the new histograms
        self.positive_plot_data_source.data.update(new_positive_source.data)
        self.negative_plot_data_source.data.update(new_negative_source.data)

    @staticmethod
    def calculate_positions(G, weight_threshold, all_nodes):
        edges_to_keep = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > weight_threshold]
        filtered_graph = G.edge_subgraph(edges_to_keep).copy()
        positions = nx.spring_layout(filtered_graph, seed=42)
        for node in all_nodes:
            if node not in positions:
                positions[node] = (0, 0)
        return positions

    def update_node_weights(self, G, largest_cc):
        node_weights_dict = defaultdict(list)
        edge_weights = nx.get_edge_attributes(G, 'weight')
        for u, v in G.edges():
            if u in largest_cc and v in largest_cc:
                node_weights_dict[u].append((v, edge_weights[(u, v)]))
                node_weights_dict[v].append((u, edge_weights[(u, v)]))
        new_weights = ['<br>'.join(f"{pair[0]} | {pair[1]:.4f}" for pair in sorted(node_weights_dict[node], key=lambda x: x[0])) if node in largest_cc else '' for node in self.nodes_cds.data['name']]
        return new_weights

    def update_graph_layout(self, largest_cc, new_positions, new_weights, new_bc):
        name_to_index = {name: index for index, name in enumerate(self.nodes_cds.data['name'])}
        graph_layout = {name_to_index[node]: new_positions[node] for node in largest_cc}
        self.nodes_cds.data = {
            'index': self.nodes_cds.data['index'],
            'name': self.nodes_cds.data['name'],
            'x': [graph_layout.get(name_to_index[node], (float('nan'), float('nan')))[0] if node in largest_cc else 'nan' for node in self.nodes_cds.data['name']],
            'y': [graph_layout.get(name_to_index[node], (float('nan'), float('nan')))[1] if node in largest_cc else 'nan' for node in self.nodes_cds.data['name']],
            'betweenness': new_bc,
            'weights': new_weights,
            'color': self.nodes_cds.data['color']
        }
        return graph_layout

    @staticmethod
    def non_zero_betweenness(betweenness_dict):
        return {k: v for k, v in betweenness_dict.items() if v > 0}
    
    def trigger_initial_update(self):
        self.update('value', self.slider.value, self.slider.value)

def modify_doc(doc):
    app = CurrencyNetworkApp()
    doc.title = "PLM Currency Network"
    doc.clear()

    # Create a new instance of the Div component
    print("Adding author table...")
    author_table = Div(text=f"""
        <table style="border: 0; padding: 0;">
            <tr>
                <td style="padding: 0;">
                    <a href="{website_url}" target="_blank">
                        <img src="{icon_url}" style="height: 14pt; width: 14pt; vertical-align: middle;">
                    </a>
                </td>
                <td style="padding: 0; vertical-align: middle;">
                    <span style="font-size: 12pt;"><b>&nbsp;{author_name}</b></span>
                </td>
            </tr>
        </table>
        """)

    # Add the new instance of the Div component to the layout
    print("Setting up layout...")
    plot_layout = column(TITLE, app.plot)
    histograms_layout = column(app.positive_fig, app.negative_fig, author_table, sizing_mode="scale_width")
    stats_layout = row(app.bc_table, histograms_layout, sizing_mode="scale_width")
    controls_layout = column(app.slider, app.threshold_value_div, stats_layout, sizing_mode="scale_width")
    main_layout = row(plot_layout, controls_layout, sizing_mode="scale_width")

    doc.add_root(main_layout)
    doc.add_next_tick_callback(app.trigger_initial_update)

# Check if running with 'bokeh serve' or not
if __name__ != '__main__':
    modify_doc(curdoc())
else:
    # Running directly, set up server
    bokeh_app = Application(FunctionHandler(modify_doc))
    port = int(os.environ.get('PORT', 5006))  # Default to 5006 if $PORT not set
    allowed_origins = [
        'plm-currency-network.com',  # Custom domain
        'currency-network-ffd38c966f8f.herokuapp.com'  # Default Heroku domain
    ]
    server = Server({'/': bokeh_app}, port=port, allow_websocket_origin=allowed_origins)
    server.start()
    server.run_until_shutdown()