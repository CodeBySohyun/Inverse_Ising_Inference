from bokeh.models import ColumnDataSource, Span
from bokeh.plotting import figure
from bokeh.palettes import RdBu as palette
import numpy as np

def bins(parameter):
    # Calculate the ideal number of bins using the Freedman-Diaconis rule
    IQR = np.percentile(parameter, 75) - np.percentile(parameter, 25)
    bin_width = 2 * IQR * len(parameter) ** (-1/3)
    bins = int((max(parameter) - min(parameter)) / bin_width)
    return bins

def calculate_couplings_histogram(J_df):
    # Convert dataframe to numpy array and extract upper triangle values
    J = J_df.values
    upper_triangle_indices = np.triu_indices_from(J, k=1)
    J_values = J[upper_triangle_indices]
    
    # Separate positive and negative couplings
    positive_couplings = J_values[J_values > 0]
    negative_couplings = -J_values[J_values < 0]  # Make negative values positive

    # Compute histograms
    positive_histogram, positive_bin_edges = np.histogram(positive_couplings, bins='auto', density=True)
    negative_histogram, negative_bin_edges = np.histogram(negative_couplings, bins='auto', density=True)
    
    # Calculate the widths and bin centers for the positive histogram
    positive_widths = np.diff(positive_bin_edges)
    positive_bin_centers = positive_bin_edges[:-1] + positive_widths / 2

    # Calculate the widths and bin centers for the negative histogram
    negative_widths = np.diff(negative_bin_edges)
    negative_bin_centers = negative_bin_edges[:-1] + negative_widths / 2

    # Create ColumnDataSources for positive and negative histograms
    positive_source = ColumnDataSource(data={
        'histogram': positive_histogram, 
        'bin_centers': positive_bin_centers, 
        'widths': positive_widths
    })
    negative_source = ColumnDataSource(data={
        'histogram': negative_histogram, 
        'bin_centers': negative_bin_centers, 
        'widths': negative_widths
    })

    return positive_source, negative_source

def create_histogram_plots(J_df):
    # Initialise data sources for the histograms
    positive_source, negative_source = calculate_couplings_histogram(J_df)

    # Create the figure for the positive couplings histogram
    positive_fig = figure(title="Positive Couplings", tools=[],
                          sizing_mode="stretch_width",
                          height=300, max_width=400)
    positive_fig.vbar(x='bin_centers', top='histogram', width='widths', color=palette[11][-2], 
                      source=positive_source, legend_label="Data (PDF)")

    positive_fig.xaxis.axis_label = r"$$J_{ij}$$"
    positive_fig.xaxis.axis_label_text_font_size = '10pt'
    positive_fig.yaxis.axis_label_text_font_size = '10pt'
    positive_fig.title.text_font_size = '10pt'
    positive_fig.toolbar.logo = None
    positive_fig.toolbar_location = None

    # Customising and positioning the legend inside the plot
    positive_fig.legend.location = "top_right"
    positive_fig.legend.label_text_font_size = '8pt'  # Adjusting the font size
    positive_fig.legend.background_fill_alpha = 0.5  # Setting background transparency
    positive_fig.legend.background_fill_color = 'white'  # Setting background color

    # Create the figure for the negative couplings histogram
    negative_fig = figure(title="Negative Couplings", tools=[],
                          sizing_mode="stretch_width",
                          height=300, max_width=400)
    negative_fig.vbar(x='bin_centers', top='histogram', width='widths', color=palette[11][1], 
                      source=negative_source, legend_label="Data (PDF)")
    negative_fig.xaxis.axis_label = r"$$-J_{ij}$$"
    negative_fig.xaxis.axis_label_text_font_size = '10pt'
    negative_fig.yaxis.axis_label_text_font_size = '10pt'
    negative_fig.title.text_font_size = '10pt'
    negative_fig.toolbar.logo = None
    negative_fig.toolbar_location = None

    # Customising and positioning the legend inside the plot
    negative_fig.legend.location = "top_right"
    negative_fig.legend.label_text_font_size = '8pt'  # Adjusting the font size
    negative_fig.legend.background_fill_alpha = 0.5  # Setting background transparency
    negative_fig.legend.background_fill_color = 'white'  # Setting background color

    # Initialise the vertical line glyph at x=0, this location will be updated dynamically
    positive_threshold_line = Span(location=0, dimension='height', line_color='black', line_width=1)
    negative_threshold_line = Span(location=0, dimension='height', line_color='black', line_width=1)

    # Add the vertical line glyph to the figures
    positive_fig.add_layout(positive_threshold_line)
    negative_fig.add_layout(negative_threshold_line)

    return positive_fig, negative_fig, positive_source, negative_source, positive_threshold_line, negative_threshold_line
