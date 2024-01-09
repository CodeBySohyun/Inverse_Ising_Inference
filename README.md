# Inverse Ising Inference on Foreign Exchange Markets

## Overview
This repository hosts my final year pioneering Physics project from the University of Bristol. It focuses on applying inverse Ising inference to unravel the complex correlations between currency pairs in the foreign exchange markets. My approach, inspired by neuroscience and condensed matter physics, adapts techniques typically used in spin system analyses to the realm of financial data.

## Project Description
My primary objective is to identify and correct for the spill-over effect in currency correlations, distinguishing real interactions from superficial ones. I explore how economic announcements and global events, like the COVID-19 pandemic or the 2008 financial crisis, influence these correlations. Unlike traditional correlation studies, my project leverages the machine learning techniques-inverse Ising inference proposes a novel perspective on the underlying structure and dynamics of currency relationships.

## Methodology
My methodology involves mapping foreign exchange rates onto a network of spins, where each node represents a currency. By studying the coupling constants between these nodes, I aim to determine the strength and nature of currency interactions. This method allows me to look beyond apparent correlations to the true connections, potentially altered by economic regimes and policies.

I have implemented a pseudolikelihood maximisation technique, an efficient variant of maximum likelihood estimation, suitable for the complex nature of financial networks. This approach enables me to fit the inverse Ising model to my dataset, which spans from the early 2000s, post the introduction of the Euro, to the present, focusing on major freely floating currencies excluding those under heavy governmental control.

## Contents
- **Data_Analysis/**:
  - `ising_data_preprocessing.ipynb`: Data preparation and preprocessing, focusing on daily changes in currency rates relative to the dollar.
  - `ising_optimisation_original.ipynb`: Preliminary model optimisation with comprehensive annotations.
  - `ising_optimisation.ipynb`: Streamlined code for advanced model optimisation techniques.
  - `ising_visualisation.ipynb`: Visual representations of data and model outputs.
  - `ising_statistics.ipynb`: Statistical analysis of currency correlations and interactions.
  - `network_app.py`: Sets up a Bokeh server application for visualising inferred network structures. It integrates features such as dynamic threshold filtering, histogram plotting, and betweenness centrality analysis, enhancing the visual analysis aspect of `ising_visualisation.ipynb`.

- **Data_Analysis/Utility/**:
  - `IsingOptimiser.py`: Defines the IsingOptimiser class for optimising an Ising model on financial data. This script includes methods for data subset division, optimisation, and result visualisation. It plays a crucial role in `ising_optimisation.ipynb` by providing the necessary tools for model fitting and analysis.
  - `BoltzmannMachine.py`: Establishes the BoltzmannMachine class for training a Boltzmann machine. This script is vital for extending the capabilities of `ising_optimisation.ipynb`, offering advanced optimisation methods including the hidden variable, detailed result analysis, visualisation, and saving extended results.
  - `network_pdf.py`: Generates plots of histograms from a coupling matrix, including power law fitting. This script is used within `network_app.py`.
  - `decorators.py`: Provides auxiliary functions that augment the functionality of the main analysis scripts

- **Additional Resources**:
  - `Data/Currency_Pairs/`: Historical exchange rate csv files.
  - `Interim_Report.pdf`: Summary of project objectives and initial findings in the first half.
  - `References/`: Literature on theoretical and methodological approaches.

## Execution Guide
Follow this sequence for processing and analysing the data:
1. Data Preparation: Start with `ising_data_preprocessing.ipynb`.
2. Model Fitting: Proceed to `ising_optimisation.ipynb`.
3. Analysis: Utilise `ising_visualisation.ipynb` and `ising_statistics.ipynb` for in-depth statistical analysis.
4. Network Examination: [PLM Currency Network](https://currency-network-ffd38c966f8f.autoidleapp.com)

For detailed execution instructions, please refer to the individual files.

## Acknowledgments
Special thanks to Dr. Thomas Machon and Dr. Francesco Turci for their guidance, and to the University of Bristol for their support.
