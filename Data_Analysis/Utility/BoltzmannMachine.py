import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
import time

class BoltzmannMachine:
    def __init__(self, data_matrix_df, J_matrix_df, h_vector_df):
        self.data_matrix_df = data_matrix_df
        self.symbols = data_matrix_df.columns.tolist()[1:]
        self.X = data_matrix_df.drop(columns=["Date"]).to_numpy()
        self.J = J_matrix_df.values
        self.h = h_vector_df[1].values
        self.Z = np.random.choice([-1, 1], size=(self.X.shape[0],))

    def train(self, max_attempts=1000, display_results=None):
        """
        This method attempts to optimise the model multiple times (up to 'max_attempts') 
        If the optimisation is successful, the results are stored for averaging.
        After reaching a predetermined number of successful optimisations,
        it averages the results to obtain the final optimised parameters.
        
        Args:
            max_attempts (int): The maximum number of optimisation attempts.
            display_results (str): If the input is 'scatter', it displays the scatter plots of optimised results
        """

        # Initialise lists to store the results of successful optimisations
        J_results, h_results, w_results, b_results = [], [], [], []

        N, d = self.X.shape  # Dimensionality of the model
        epsilon = 1e-6  # small constant to avoid very small values

        success_count = 0
        max_successes = max_attempts / 10  # Maximum number of successful optimisations required

        for attempt in range(max_attempts):
            # Initialise weights using Xavier and He initialisation methods
            # J_upper = np.triu(np.random.normal(0, np.sqrt(2 / N) + epsilon, size=(d, d)), 1)
            # J = J_upper + J_upper.T
            # h = np.random.normal(0, np.sqrt(2 / N) + epsilon, size=(d,))
            w = np.random.normal(0, np.sqrt(2 / N) + epsilon, size=(d,))  # Xavier initialisation
            b = np.random.normal(0, np.sqrt(2 / N) + epsilon)  # He initialisation

            # Start timing just before calling _optimise_model
            start_time = time.time()
            # Perform the optimisation
            J_optimised, h_optimised, w_optimised, b_optimised, success = self._optimise_model(
                self.X, self.J, self.h, w, b)
            # End timing right after the function call
            end_time = time.time()

            # Store and count successful optimisations
            if success:
                J_results.append(J_optimised)
                h_results.append(h_optimised)
                w_results.append(w_optimised)
                b_results.append(b_optimised)
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
            self.w_optimised = np.mean(w_results, axis=0)
            self.b_optimised = np.mean(b_results)

            self.J_std = np.std(J_results, axis=0, ddof=1)
            self.h_std = np.std(h_results, axis=0, ddof=1)
            self.w_std = np.std(w_results, axis=0, ddof=1)
            self.b_std = np.std(b_results, ddof=1)
            self.b = b_results
            self.Z = self._predict_Z()  # Predict the hidden variable of Z

            # Print statistical analysis results on w and b results
            # print("Statistical Analysis of w_optimised and b_optimised:")
            # print(f"Mean of w_optimised: {self.w_optimised}")
            # print(f"Standard Deviation of w_optimised: {self.w_std}")
            # print(f"Mean of b_optimised: {self.b_optimised}")
            # print(f"Standard Deviation of b_optimised: {self.b_std}")

            print(f"Optimisation completed with averaged results after {success_count} successes.\n")

            if display_results == 'scatter':
                self.scatter_plots()
        else:
            print("Optimisation was unsuccessful after maximum attempts.\n")

    def _optimise_model(self, X, J, h, w, b):
        """
        Optimise the Ising model using the L-BFGS-B algorithm.
        """
        d = J.shape[0]  # Number of spins (currencies)

        # Flatten parameters for the optimisation
        x0 = np.hstack([J[np.triu_indices(d, k=1)], h, w, np.array([b])])

        def objective_function(x):
            J, h, w, b = self._reconstruct_parameters(x, d)
            log_likelihood, grad_J, grad_h, grad_w, grad_b, overflow_occurred = self._log_PL_and_gradients(J, h, w, b, X)

            if overflow_occurred:
                # If overflow occurred, return a large positive value to indicate a bad optimisation step
                return np.inf, np.zeros_like(x)
            return log_likelihood, np.hstack([grad_J[np.triu_indices(d, k=1)], grad_h, grad_w, np.array([grad_b])])

        # Use 'L-BFGS-B' for optimisation with Jacobian
        res = minimize(objective_function, x0, method='L-BFGS-B', jac=True)

        # Consider the optimisation unsuccessful if overflow occurred (indicated by `fun` being 0)
        if res.fun == 0:
            return None, None, None, None, False

        # Reconstruct the optimised parameters from the optimisation result
        J_optimised, h_optimised, w_optimised, b_optimised = self._reconstruct_parameters(res.x, d)

        return J_optimised, h_optimised, w_optimised, b_optimised, res.success

    @staticmethod
    def _reconstruct_parameters(flattened_array, d):
        num_J_elements = d * (d - 1) // 2
        J_upper_tri = flattened_array[:num_J_elements]
        h = flattened_array[num_J_elements:num_J_elements + d]
        w = flattened_array[num_J_elements + d:-1]
        b = flattened_array[-1]  # Last element is scalar b

        # Construct the symmetric J matrix from the upper triangular part
        J = np.zeros((d, d))
        J[np.triu_indices(d, k=1)] = J_upper_tri
        J += J.T  # Symmetrise the J matrix

        return J, h, w, b

    @staticmethod
    def _log_PL_and_gradients(J, h, w, b, X):
        d = X.shape[1]  # Number of dimensions

        try:
            # Temporarily treat overflow warnings as errors
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                
                S = X @ J + h  # Broadcasting h
                S_wb = X @ w + b  # for all samples
                S_wb_not_i = X @ w.reshape(-1, 1) + b - X * w  # Broadcasting w

                # Calculate the components used in the log likelihood and gradients
                exp_S = np.exp(S)
                cosh_pos = np.cosh(S_wb_not_i + w)
                cosh_neg = np.cosh(S_wb_not_i - w)
                sinh_pos = np.sinh(S_wb_not_i + w)
                sinh_neg = np.sinh(S_wb_not_i - w)
                denominator = exp_S * cosh_pos + cosh_neg / exp_S

                log_numerator = np.log(np.cosh(S_wb.reshape(-1, 1)))  # Reshape to 2D
                log_denominator = np.log(denominator)  # Sum over dimensions for each sample

                # Combine the terms and sum over all samples to get the scalar log-likelihood
                log_likelihood = np.sum(X * S + log_numerator - log_denominator)

                # Gradients for h, w, and b
                numerator_h = -cosh_neg / exp_S + exp_S * cosh_pos
                grad_h = np.sum(X - numerator_h / denominator, axis=0)

                numerator_w = -sinh_neg / exp_S + exp_S * sinh_pos
                grad_w = np.sum(X.T * np.tanh(S_wb.T) - (numerator_w / denominator).T, axis=1)

                numerator_b = sinh_neg / exp_S + exp_S * sinh_pos
                grad_b = np.sum(np.tanh(S_wb.reshape(-1, 1)) - (numerator_b / denominator))

                # Gradient for J, excluding the diagonal
                mask = np.ones_like(J) - np.eye(d)
                numerator_J = -cosh_neg / exp_S + exp_S * cosh_pos
                grad_term_J = X - numerator_J / denominator
                grad_J = (grad_term_J.T @ X + X.T @ grad_term_J) * mask

                overflow_occurred = False  # No overflow occurred

        except Warning as e:
            print(f'Warning encountered during optimisation: {e}')
            # Return zeros and indicate that an overflow occurred
            return 0, np.zeros_like(J), np.zeros_like(h), np.zeros_like(w), np.zeros_like(b), True

        # Return the negative likelihood and gradients for minimisation
        return -log_likelihood, -grad_J, -grad_h, -grad_w, -grad_b, overflow_occurred

    def _predict_Z(self):
        """
        Predict Z using the observed data X and optimised parameters
        """
        n = self.X.shape[0]
        Z_predicted = np.zeros(n)
        for i in range(n):
            # Calculate the effective field for each sample
            h_eff = np.dot(self.w_optimised, self.X[i, :]) + self.b_optimised
            # Predict the state of Z based on the sign of the effective field
            Z_predicted[i] = 1 if h_eff > 0 else -1
        return Z_predicted.astype(int)

    def scatter_plots(self):
        d = len(self.symbols)
        reversed_symbols = [col[:3] if len(col) == 6 else col for col in self.symbols[::-1]]  # Reverse the order of symbols
        J_means = self.J_optimised[np.triu_indices(d, k=1)]
        J_std = self.J_std[np.triu_indices(d, k=1)]

        # Plotting scatter plots for optimised parameters
        plt.figure(figsize=(12, 12))

        # Scatter plot for J
        plt.subplot(2, 2, 1)
        plt.errorbar(J_means, range(len(J_means))[::-1], xerr=J_std, fmt='o', ecolor='r', capsize=5, label='J elements', markersize=5)
        plt.title("Coupling Constants (J) Between Observed Currencies")
        plt.ylabel("Element Index in J")
        plt.xlabel("Mean Values and Standard Deviations")
        plt.grid(True)
        plt.legend()

        # Scatter plot for h
        plt.subplot(2, 2, 2)
        plt.errorbar(self.h_optimised[::-1], reversed_symbols, xerr=self.h_std, fmt='o', ecolor='r', capsize=5, label='h elements')
        plt.title("External Fields (h) for Observed Currencies")
        plt.yticks(range(d), reversed_symbols)
        plt.xlabel("Mean Values and Standard Deviations")
        plt.grid(True)
        plt.legend()

        # Scatter plot for w
        plt.subplot(2, 2, 3)
        plt.errorbar(self.w_optimised[::-1], reversed_symbols, xerr=self.w_std, fmt='o', ecolor='r', capsize=5, label='w elements')
        plt.title("Coupling Constants (w) Between Observed and Hidden Currencies")
        plt.yticks(range(d), reversed_symbols)
        plt.xlabel("Mean Values and Standard Deviations")
        plt.grid(True)
        plt.legend()

        # Scatter plot for b
        plt.subplot(2, 2, 4)
        plt.errorbar(self.b, range(len(self.b))[::-1], xerr=self.b_std, fmt='o', ecolor='b', capsize=5, label='Individual b values')
        plt.axvline(x=self.b_optimised, color='r', linestyle='-', label='Mean of b')
        plt.title("External Field (b) for the Hidden Currency")
        plt.ylabel("Index of Successful Optimised Attempts")
        plt.xlabel("Values in Each Run and Standard Deviation")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.suptitle("Scatter Plots of Optimised Parameter Values Across Successful Optimisations", fontsize=16, y=1.02)
        plt.show()

    def save_results(self, J_path, h_path, J_extended_path, h_extended_path, data_matrix_extended_path):
        """
        Save optimised and extended results of the Boltzmann machine.
        """
        # Save optimised results (J matrix and h vector only)
        self._save_optimised_results(J_path, h_path)

        # Save extended results with Z (Extended J, h, w, b and data matrix)
        self._save_extended_results(J_extended_path, h_extended_path, data_matrix_extended_path)

    def _save_optimised_results(self, J_path, h_path):
        """
        Save the optimised J matrix and h vector.
        """
        pd.DataFrame(self.J_optimised, columns=self.symbols, index=self.symbols).to_csv(J_path, index=False)
        print(f"The optimised J matrix has been saved to '{J_path}'.")

        pd.DataFrame({'h': self.h_optimised}, index=self.symbols).to_csv(h_path, index=True, header=False)
        print(f"The optimised h vector has been saved to '{h_path}'.")

    def _save_extended_results(self, J_extended_path, h_extended_path, data_matrix_extended_path):
        """
        Save the extended J matrix, h vector, and data matrix including Z.
        """
        d = self.X.shape[1] + 1
        # Extended J matrix
        J_extended = np.zeros((d, d))
        J_extended[0, 1:] = J_extended[1:, 0] = self.w_optimised
        J_extended[1:, 1:] = self.J_optimised
        symbols_extended = ['USD'] + self.symbols
        pd.DataFrame(J_extended, columns=symbols_extended, index=symbols_extended).to_csv(J_extended_path, index=False)
        print(f"The extended optimised J matrix has been saved to '{J_extended_path}'.")

        # Extended h vector
        h_values_with_b = [self.b_optimised] + list(self.h_optimised)
        h_df_extended = pd.DataFrame({'h': h_values_with_b}, index=['USD'] + self.symbols)
        h_df_extended.to_csv(h_extended_path, index=True, header=False)
        print(f"The extended optimised h vector has been saved to '{h_extended_path}'.")

        # Extended data matrix with Z
        data_matrix_df_extended = self.data_matrix_df.copy()
        data_matrix_df_extended.insert(1, 'USD', self.Z)
        data_matrix_df_extended.to_csv(data_matrix_extended_path, index=False)
        print(f"Extended data matrix saved to '{data_matrix_extended_path}'.")