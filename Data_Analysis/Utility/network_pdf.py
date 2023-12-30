from bokeh.models import ColumnDataSource, Span, Legend
from bokeh.plotting import figure
from bokeh.palettes import RdBu as palette
import numpy as np

# Let's define a function that takes in the J matrix and returns the data sources
# for positive and negative couplings histograms. This function will be prepared
# for integration with the existing Bokeh app.

def calculate_couplings_histogram(J_df):
    """
    This function calculates the histograms for the positive and negative couplings
    from a symmetric J matrix, considering a log scale for the bins.
    
    Parameters:
    - J_df: A pandas DataFrame representing the J matrix.
    
    Returns:
    - A tuple of ColumnDataSources for positive and negative histograms.
    """
    # Convert dataframe to numpy array and extract upper triangle values
    J = J_df.values
    upper_triangle_indices = np.triu_indices_from(J, k=1)
    J_values = J[upper_triangle_indices]
    
    # Separate positive and negative couplings
    positive_couplings = J_values[J_values > 0]
    negative_couplings = -J_values[J_values < 0]  # Make negative values positive

    # Calculate logarithmic bins
    positive_bins = np.logspace(np.log10(positive_couplings.min()), np.log10(positive_couplings.max()), num=len(positive_couplings))
    negative_bins = np.logspace(np.log10(negative_couplings.min()), np.log10(negative_couplings.max()), num=len(negative_couplings))

    # Compute histograms
    positive_histogram, _ = np.histogram(positive_couplings, bins=positive_bins, density=True)
    negative_histogram, _ = np.histogram(negative_couplings, bins=negative_bins, density=True)
    
    # Prepare data for plotting
    positive_bin_edges = positive_bins[:-1]
    negative_bin_edges = negative_bins[:-1]
    
    # Create ColumnDataSources for positive and negative histograms
    positive_source = ColumnDataSource(data=dict(couplings=positive_couplings, bins=positive_bins, histogram=positive_histogram, bin_edges=positive_bin_edges))
    negative_source = ColumnDataSource(data=dict(couplings=negative_couplings, bins=negative_bins, histogram=negative_histogram, bin_edges=negative_bin_edges))
    
    return positive_source, negative_source

# Now we have a function that can be integrated into the Bokeh app script.
# The function returns ColumnDataSources which can be directly used to source a Bokeh plot.
# This can be inserted into the script, and the plotting part can be called within the Bokeh server document lifecycle.

# For integration, this function should be called inside the main app function or the callback that
# handles threshold changes, then the resulting ColumnDataSources should be used to update the plots.

# A simplified integration might look like this within the Bokeh app script:

# When the threshold slider is adjusted:
# positive_source, negative_source = calculate_couplings_histogram(J_df)
# # Update the plot data sources
# positive_plot_data_source.data.update(positive_source.data)
# negative_plot_data_source.data.update(negative_source.data)

# The plots would need to be initially created with these data sources, and the callback
# would update them as described above.

# Let's define another function that creates the histogram plots for the positive and negative couplings.
# This function will initialise the plots and data sources, which can be updated in the 'update' function.

from scipy.optimize import curve_fit

def plot_power_law_pdf(data):
    """
    Plot the probability density function (PDF) for the given data
    and fit a power-law to it.
    """
    # Prepare the data for PDF
    data_hist, bin_edges = np.histogram(data, bins='auto', density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Fitting the power law using curve_fit
    power_law_func = lambda x, a, b: a * x**-b
    params, _ = curve_fit(power_law_func, bin_centers, data_hist, maxfev=5000)

    # Generating values for the fitted line
    x = np.linspace(min(data), max(data), 1000)
    y_fit = power_law_func(x, *params)

    return x, y_fit

def create_histogram_plots(J_df):
    """
    This function creates histogram plots for the positive and negative couplings
    and initialises the ColumnDataSources for them.

    Returns:
    - A tuple of (positive_fig, negative_fig, positive_source, negative_source)
      where positive_fig and negative_fig are the Bokeh figure objects for the histograms,
      and positive_source and negative_source are the ColumnDataSources for the plots.
    """
    # Initialise data sources for the histograms
    positive_source, negative_source = calculate_couplings_histogram(J_df)

    # Generate a fitted line for positive couplings
    fitted_positive_x, fitted_positive_y = plot_power_law_pdf(positive_source.data['couplings'])

    # Generate a fitted line for negative couplings
    fitted_negative_x, fitted_negative_y = plot_power_law_pdf(negative_source.data['couplings'])

    # Create the figure for the positive couplings histogram
    positive_fig = figure(title="Positive Couplings", tools=[],
                          x_axis_type="log", y_axis_type="log", sizing_mode="stretch_width",
                          height=300, max_width=400)
    positive_fig.y_range.end = max(positive_source.data['histogram']) * 1.5
    positive_fig.circle(x='bin_edges', y='histogram', size=5, color=palette[11][-2], source=positive_source, legend_label="Data (PDF)")
    positive_fig.line(fitted_positive_x, fitted_positive_y, line_width=2, color=palette[11][-1], legend_label="Fitted Power Law")
    positive_fig.xaxis.axis_label = r"$$J_{ij}$$"
    # positive_fig.yaxis.axis_label = "pdf"
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
                          x_axis_type="log", y_axis_type="log", sizing_mode="stretch_width",
                          height=300, max_width=400)
    negative_fig.y_range.end = max(negative_source.data['histogram']) * 1.5
    negative_fig.circle(x='bin_edges', y='histogram', size=5, color=palette[11][1], source=negative_source, legend_label="Data (PDF)")
    negative_fig.line(fitted_negative_x, fitted_negative_y, line_width=2, color=palette[11][0], legend_label="Fitted Power Law")
    negative_fig.xaxis.axis_label = r"$$-J_{ij}$$"
    # negative_fig.yaxis.axis_label = "pdf"
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

# This function can be called in the main part of the Bokeh app script to create the plots and data sources.
# The returned figures can be added to the layout, and the data sources can be updated in the 'update' callback.
