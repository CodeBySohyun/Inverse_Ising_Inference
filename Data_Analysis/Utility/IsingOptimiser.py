import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.lines as mlines
import warnings
import time

class IsingOptimiser:
    def __init__(self, data_path):
        """
        Initialiser for the IsingOptimiser class.

        Args:
        data_path (str): Path to the CSV file containing currency data.
        """
        self.df = pd.read_csv(data_path)
        self.symbols = self.df.columns.tolist()[1:]
        self.X = self.df.drop(columns=["Date"]).to_numpy()

    def train(self, max_attempts=1000, display_results=None):
        """
        This method attempts to optimise the model multiple times (up to 'max_attempts') 
        If the optimisation is successful, the results are stored for averaging.
        After reaching a predetermined number of successful optimisations,
        it averages the results to obtain the final optimised parameters.

        Args:
            max_iterations (int): The maximum number of optimisation attempts.
            display_results (str): If the input is 'scatter', it displays the scatter plots of optimised results
        """

        # Initialise lists to store the results of successful optimisations
        J_results, h_results = [], []

        d = self.X.shape[1]  # Dimensionality of the model
        success_count = 0
        max_successes = max_attempts / 10 # Maximum number of successful optimisations required

        for attempt in range(max_attempts):
            # Initialise J and h
            J_upper = np.triu(np.random.uniform(-1, 1, size=(d, d)), 1)
            J = J_upper + J_upper.T
            h = np.random.uniform(-1, 1, size=d)

            # Start timing just before calling _optimise_model
            start_time = time.time()
            # Perform the optimisation
            J_optimised, h_optimised, success = self._optimise_model(self.X, J, h)
            # End timing right after the function call
            end_time = time.time()

            # Store and count successful optimisations
            if success:
                J_results.append(J_optimised)
                h_results.append(h_optimised)
                success_count += 1
                print(f"Attempt {attempt + 1} of optimisation successful, execution time: {end_time - start_time:.3f} seconds")

                # Stop if the required number of successes is achieved
                if success_count >= max_successes:
                    break
            else:
                print(f"Attempt {attempt + 1} of optimisation unsuccessful, restarting...")

        # Check and analyse the results if there were successful optimisations
        if J_results:
            # Calculate average results and sample standard deviation
            self.J_optimised = np.mean(J_results, axis=0)
            self.h_optimised = np.mean(h_results, axis=0)
            self.J_std = np.std(J_results, axis=0, ddof=1)
            self.h_std = np.std(h_results, axis=0, ddof=1)

            # Print statistical analysis results on J and h results
            # print("Statistical Analysis of J_optimised and h_optimised:")
            # print(f"Mean of h_optimised: {self.J_optimised}")
            # print(f"Standard Deviation of J_optimised: {self.J_std}")
            # print(f"Mean of h_optimised: {self.h_optimised}")
            # print(f"Standard Deviation of h_optimised: {self.h_std}")

            print(f"Optimisation completed with averaged results after {success_count} successes.\n")

            if display_results == 'scatter':
                self.scatter_plots()
            elif display_results == 'vertical':
                self.vertical_plots()
        else:
            print("Optimisation was unsuccessful after maximum attempts.\n")

    def _optimise_model(self, X, J, h):
        """
        Optimise the Ising model using the L-BFGS-B algorithm.
        """
        d = J.shape[0]  # Number of spins (currencies)

        # Flatten the J matrix and h vector for the optimisation
        x0 = np.hstack([J[np.triu_indices(d, k=1)], h])

        def objective_function(x):
            J, h = self._reconstruct_parameters(x, d)
            # Modified to unpack the overflow flag
            log_likelihood, grad_J, grad_h, overflow_occurred = self._log_pseudolikelihood_and_gradients(J, h, X)
            
            if overflow_occurred:
                # If overflow occurred, return a large positive value to indicate a bad optimisation step
                return 0, np.zeros_like(x)
            return log_likelihood, np.hstack([grad_J[np.triu_indices(d, k=1)], grad_h])

        # Use 'L-BFGS-B' for optimisation with Jacobian    
        res = minimize(objective_function, x0, method='L-BFGS-B', jac=True)

        # Consider the optimisation unsuccessful if overflow occurred (indicated by `fun` being 0)
        if res.fun == 0:
            return None, None, False

        # Reconstruct the optimised J matrix and h vector from the optimisation result
        J_optimised, h_optimised = self._reconstruct_parameters(res.x, d)

        return J_optimised, h_optimised, res.success

    @staticmethod
    def _reconstruct_parameters(flattened_array, d):
        """
        Helper function to reconstruct the symmetric J matrix and h vector from a flattened array.
        """
        num_J_elements = d * (d - 1) // 2
        J_upper_tri = flattened_array[:num_J_elements]
        h = flattened_array[num_J_elements:]

        # Construct the symmetric J matrix from the upper triangular part
        J = np.zeros((d, d))
        J[np.triu_indices(d, k=1)] = J_upper_tri
        J += J.T  # Symmetrise the J matrix

        return J, h

    @staticmethod
    def _log_pseudolikelihood_and_gradients(J, h, X):
        """
        Calculate the log-pseudo-likelihood for the Ising model and its gradients.
        Catch overflow warnings and handle them gracefully.
        """
        d = X.shape[1]  # Number of dimensions

        try:
            # Temporarily treat overflow warnings as errors
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                
                # Compute S, broadcasting h
                S = X @ J + h
                S_exp = np.exp(2 * X * S)

                # Update log-likelihood
                log_likelihood = np.sum(- np.log(1 + 1/S_exp))

                grad_term = 2 * X / (1 + S_exp)
                grad_h = np.sum(grad_term, axis=0)

                # Compute the gradient for J, excluding the diagonal
                # Create a mask to zero out diagonal contributions in the grad_J calculation
                mask = np.ones_like(J) - np.eye(d)
                grad_J = (grad_term.T @ X + X.T @ grad_term) * mask

                overflow_occurred = False  # No overflow occurred

        except Warning as e:
            print(f'Warning encountered during optimisation: {e}')
            # Return zeros and indicate that an overflow occurred
            return 0, np.zeros_like(J), np.zeros_like(h), True

        # Return the negative likelihood and gradients for minimisation, and the overflow flag
        return -log_likelihood, -grad_J, -grad_h, overflow_occurred

    def scatter_plots(self):
        d = len(self.symbols)
        reversed_symbols = [col[:3] if len(col) == 6 else col for col in self.symbols[::-1]]  # Reverse the order of symbols
        J_means = self.J_optimised[np.triu_indices(d, k=1)]
        J_std = self.J_std[np.triu_indices(d, k=1)]

        # Plotting scatter plots for optimised parameters
        plt.figure(figsize=(12, 6))

        # Scatter plot for J
        plt.subplot(1, 2, 1)
        plt.errorbar(J_means, range(len(J_means))[::-1], xerr=J_std, fmt='o', ecolor='r', capsize=5, label='J elements', markersize=5)
        plt.title(r"Coupling Constants (\mathbf{\mathit{J}}) Between Currencies", fontsize=16)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.ylabel(r"Upper Triangular Element Index in \mathbf{\mathit{J}}", fontsize=14)
        # plt.xlabel("Mean Values and Standard Deviations", fontsize=14)
        plt.grid(True)

        # Scatter plot for h
        plt.subplot(1, 2, 2)
        plt.errorbar(self.h_optimised[::-1], reversed_symbols, xerr=self.h_std, fmt='o', ecolor='r', capsize=5, label='h elements')
        plt.title(r"External Fields (\mathbf{\mathit{h}}) for Each Currencies", fontsize=16)
        plt.yticks(range(d), reversed_symbols, fontsize=12)
        plt.xticks(fontsize=12)
        # plt.xlabel("Mean Values and Standard Deviations", fontsize=14)
        plt.grid(True)

        plt.tight_layout()

        # Custom legend
        mean_value_legend = mlines.Line2D([], [], marker='o', linestyle='None', markersize=10, label='Mean values')
        std_dev_legend = mlines.Line2D([], [], color='r', marker='|', linestyle='None', markersize=10, mew=2, label='Standard deviations')

        # Assemble legend rows
        legend_row = [mean_value_legend, std_dev_legend]

        # Create common legend
        plt.figlegend(handles=legend_row, loc='lower center', bbox_to_anchor=(0.5, -0.08), fancybox=True, shadow=True, ncol=2, fontsize='xx-large')

        plt.savefig("Images/optimised_parameters.svg", format="svg", bbox_inches='tight')

        # plt.suptitle("Scatter Plots of Optimised Parameter Values Across Successful Optimisations", fontsize=20, y=1.05)
        plt.show()

    def vertical_plots(self):
        d = len(self.symbols)
        reversed_symbols = [col[:3] if len(col) == 6 else col for col in self.symbols[::-1]]  # Reverse the order of symbols
        J_means = self.J_optimised[np.triu_indices(d, k=1)]
        J_std = self.J_std[np.triu_indices(d, k=1)]

        # Create a figure and multiple axes
        _, axs = plt.subplots(2, 1, figsize=(8, 14))  # 2 rows, 1 column

        # Scatter plot for J
        axs[0].errorbar(J_means, range(len(J_means))[::-1], xerr=J_std, fmt='o', ecolor='r', capsize=5, label='J elements', markersize=5)
        axs[0].set_title(r'Couplings ($\mathit{J}_{ij}$) Between Currencies', fontsize=20, pad=10)
        axs[0].set_ylabel(r'Upper Triangular Element Index in $\mathbf{\mathbf{J}}$', fontsize=16)
        axs[0].tick_params(axis='both', which='major', labelsize=14)
        axs[0].grid(True)

        # Scatter plot for h
        axs[1].errorbar(self.h_optimised[::-1], reversed_symbols, xerr=self.h_std, fmt='o', ecolor='r', capsize=5, label='h elements')
        axs[1].set_title(r'External Fields ($\mathit{h}_{i}$) for Each Currencies', fontsize=20, pad=10)
        axs[1].set_ylabel("Currency", fontsize=16)
        axs[1].set_yticks(range(d))
        axs[1].set_yticklabels(reversed_symbols)
        axs[1].tick_params(axis='both', which='major', labelsize=14)
        axs[1].grid(True)

        # Add legend
        mean_value_legend = mlines.Line2D([], [], marker='o', linestyle='None', markersize=10, label='Mean values')
        std_dev_legend = mlines.Line2D([], [], color='r', marker='|', linestyle='None', markersize=10, mew=2, label='Standard deviations')
        axs[1].legend(handles=[mean_value_legend, std_dev_legend], loc='lower center', bbox_to_anchor=(0.5, -0.175), fancybox=True, shadow=True, ncol=2, fontsize='xx-large')

        plt.tight_layout()

        plt.savefig("Images/optimised_parameters.svg", format="svg", bbox_inches='tight')
        plt.show()

    def save_results(self, J_file_path, h_file_path, J_std_file_path, h_std_file_path):
        """
        Save the optimised J matrix and h vector to CSV files.
        """
        J_df = pd.DataFrame(self.J_optimised, columns=self.symbols, index=self.symbols)
        h_df = pd.DataFrame({'Symbol': self.symbols, 'h': self.h_optimised})

        J_df.to_csv(J_file_path, index=False)
        h_df.to_csv(h_file_path, index=False, header=False)

        print(f"The optimised J matrix has been saved to '{J_file_path}'.")
        print(f"The optimised h vector has been saved to '{h_file_path}'.")

        J_std_df = pd.DataFrame(self.J_std, columns=self.symbols, index=self.symbols)
        h_std_df = pd.DataFrame({'Symbol': self.symbols, 'h': self.h_std})

        J_std_df.to_csv(J_std_file_path, index=False)
        h_std_df.to_csv(h_std_file_path, index=False, header=False)

        print(f"The standard deviation of J matrix has been saved to '{J_std_file_path}'.")
        print(f"The standard deviation of h vector has been saved to '{h_std_file_path}'.")