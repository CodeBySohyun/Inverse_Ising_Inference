# Inverse Ising Inference on Foreign Exchange Markets

## Overview
This repository hosts my pioneering final-year physics project from the University of Bristol. Inspired by computational neuroscience and condensed matter physics, I focus on applying inverse Ising inference to uncover the complex correlation structure of foreign exchange markets.

## Abstract
Financial datasets are difficult to estimate due to their inherent heavy-tailed distribution. A solution to this problem is inverse Ising inference, also known as pairwise maximum entropy modelling. The correlation structure of foreign exchange markets was reconstructed using this machine learning technique. Additionally, data-driven methods including pseudolikelihood maximisation and the Metropolis algorithm were employed to explore two decades of time series, to identify transformative shifts corresponding to known geopolitical and financial events, and infer criticality in dynamic market structure. This may suggest a role for self-correcting feedback mechanisms in maintaining a market critical state.

## Contents
- **Data_Analysis/**:
  - `ising_data_preprocessing.ipynb`: Data preparation and preprocessing.
  - `ising_optimisation.ipynb`: Ising model parameters optimisation.
  - `ising_visualisation.ipynb`: Visual representations of inferred parameters.
  - `ising_statistics.ipynb`: Extensive statistical analysis.
  - `network_app.py`: Sets up a Bokeh server application for visualising the inferred currency network. It integrates features such as dynamic threshold filtering, histogram plotting, and betweenness centrality analysis.

- **Data_Analysis/Utility/**:
  - `IsingOptimiser.py`: Defines the Ising Optimiser class for optimising an Ising model on financial data. This script includes methods for optimisation and result visualisation. It plays a crucial role in `ising_optimisation.ipynb` by providing the necessary tools.
  - `BoltzmannMachine.py`: Establishes the Boltzmann Machine class for training a Boltzmann machine, which is the Ising model with a hidden variable.
  - `network_pdf.py`: Generates plots of histograms from an inferred connectivity matrix. This script is used within `network_app.py`.
  - `decorators.py`: Provides auxiliary functions that augment the functionality of the main analysis scripts.

- **Additional Resources**:
  - `Data/Currency_Pairs/`: Historical exchange rate CSV files.
  - `Interim_Report.pdf`: Summary of project objectives and initial findings in the first half.
  - **`Statistical Mechancis of Foreign Exchange Markets.pdf`: Final report.**
  - `References/`: Literature on theoretical and methodological approaches, [Litmaps](https://app.litmaps.com/shared/4ff00cbb-1c76-432a-af00-e0c6af755c6f).

## Execution Guide
Follow this sequence for processing and analysing the data:
1. Data Preparation: Start with `ising_data_preprocessing.ipynb`.
2. Optimisation: Proceed to `ising_optimisation.ipynb`.
3. Analysis: Utilise `ising_visualisation.ipynb` and `ising_statistics.ipynb` for in-depth statistical analysis.
4. Network Examination: [PLM Currency Network](https://currency-network-ffd38c966f8f.autoidleapp.com).

For detailed execution instructions, please refer to the individual files.

## Acknowledgments
Special thanks to Dr Thomas Machon and Dr Francesco Turci for their guidance, and to the University of Bristol for their support.
