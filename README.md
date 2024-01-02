# Inverse Ising Inference on Foreign Exchange Markets

## Overview
This repository hosts my Physics final year project from the University of Bristol, focusing on applying inverse Ising inference to analyse correlations in foreign exchange markets. The core idea revolves around understanding the true underlying relationships between different currency pairs, considering both direct and indirect interactions influenced by global economic factors.

## Project Description
My project aims to transcend traditional statistical methods of measuring correlation coefficients by employing a physics-inspired technique—inverse Ising inference. I draw parallels from condensed matter physics, viewing currencies as a network of spins, each influencing its neighbour. This model helps me understand not just direct correlations but also indirect influences, thus revealing the true structure of currency relationships.

I apply machine learning techniques to fit a spin model to financial data, examining currencies' intricate dynamics. My focus lies in differentiating genuine connections from apparent correlations caused by intermediary influences. This approach, novel in the realm of financial analysis, seeks to uncover the underlying structure of currency correlations, potentially reshaped by major global events or economic shifts.

## Methodology
My methodology maps currencies onto a network of spins, with each currency pair connection represented by coupling constants. I investigate these relationships over time, hypothesising that they undergo phase transitions, influenced by economic events like the 2008 financial crash or the COVID-19 pandemic.

The study utilises a pseudo-likelihood maximisation approach, a variant of Maximum Likelihood Estimation, suitable for large datasets and intricate networks like foreign exchange markets. My analysis spans from the early 2000s, post the introduction of the Euro, to present times, focusing on freely floating currencies and excluding those under heavy governmental control.

## Contents
- **Data_Analysis/**:
  - `ising_data_preprocessing.ipynb`: Data preparation and preprocessing.
  - `ising_optimisation_original.ipynb`: Preliminary work for model optimisation with detailed comments.
  - `ising_optimisation.ipynb`: Streamlined code for advanced model optimisation techniques.
  - `ising_visualisation.ipynb`: Visual representations of data and model outputs.
  - `ising_statistics.ipynb`: In-depth statistical analysis.
  - `network_app.py`: Sets up a Bokeh server application for visualising a currency network using a graph layout. It integrates features such as dynamic threshold filtering, histogram plotting, and betweenness centrality analysis, enhancing the visual analysis aspect of `ising_visualisation.ipynb`.

- **Data_Analysis/Utility/**:
  - `IsingOptimiser.py`: Defines the IsingOptimiser class for optimising an Ising model on financial data. This script includes methods for data subset division, optimisation, and result visualisation. It plays a crucial role in `ising_optimisation.ipynb` by providing the necessary tools for model fitting and analysis.
  - `BoltzmannMachine.py`: Establishes the BoltzmannMachine class for training a Boltzmann machine. This script is vital for extending the capabilities of `ising_optimisation.ipynb`, offering advanced optimisation methods including the hidden variable, detailed result analysis, visualisation, and saving extended results.
  - `network_pdf.py`: Generates plots of histograms from a coupling matrix, including power law fitting. This script is used within `network_app.py`.
  - `decorators.py`: Provides auxiliary functions that augment the functionality of the main analysis scripts

- **Additional Resources**:
  - `Data/Currency_Pairs/`: Historical exchange rate csv files.
  - `Interim_Report.pdf`: Summary of project objectives and findings in the first half.
  - `References/`: Literature on theoretical and methodological approaches.

## Execution Guide
Follow this sequence for processing and analysing the data:
1. Data Preparation: Start with `IsingDataPreprocessing.ipynb`.
2. Model Fitting: Proceed to `IsingOptimisation.ipynb`.
3. Analysis: Utilise `IsingVisualisation.ipynb` and `IsingStatistics.ipynb` for in-depth statistical analysis.
4. Network Examination: [PLM Currency Network](https://currency-network-ffd38c966f8f.autoidleapp.com)

For detailed execution instructions, please refer to the individual files.

## Acknowledgments
Special thanks to Dr. Thomas Machon and Dr. Francesco Turci for their guidance, and to the University of Bristol for their support.
