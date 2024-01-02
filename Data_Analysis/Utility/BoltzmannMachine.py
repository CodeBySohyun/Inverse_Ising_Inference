import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from Utility.decorators import timer

class BoltzmannMachine:
    def __init__(self, data_matrix_df, J_matrix_df, h_vector_df):
        self.data_matrix_df = data_matrix_df
        self.symbols = data_matrix_df.columns.tolist()[1:]
        self.X = data_matrix_df.drop(columns=["Date"]).to_numpy()
        self.J = J_matrix_df.values
        self.h = h_vector_df[1].values
        self.Z = np.random.choice([-1, 1], size=(self.X.shape[0],))

    def train(self, max_attempts=1000):
        """
        This method attempts to optimise the model multiple times (up to 'max_attempts') 
        If the optimisation is successful, the results are stored for averaging.
        After reaching a predetermined number of successful optimisations,
        it averages the results to obtain the final optimised parameters.
        
        Args:
            max_attempts (int): The maximum number of optimisation attempts.
        """

        # Initialise lists to store the results of successful optimisations
        J_results, h_results, w_results, b_results = [], [], [], []

        d = self.X.shape[1]  # Dimensionality of the model
        epsilon = 1e-6  # small constant to avoid very small values

        success_count = 0
        max_successes = max_attempts / 10  # Maximum number of successful optimisations required

        for attempt in range(max_attempts):
            print(f"Attempt {attempt + 1} of optimisation")

            # Initialise weights using Xavier and He initialisation methods
            w = np.random.normal(0, np.sqrt(2 / d) + epsilon, size=(d,))  # Xavier initialisation
            b = np.random.normal(0, np.sqrt(2 / d) + epsilon)  # He initialisation

            # Perform the optimisation
            J_optimised, h_optimised, w_optimised, b_optimised, success = self._optimise_model(
                self.X, self.J, self.h, w, b, 'full')

            # Store and count successful optimisations
            if success:
                J_results.append(J_optimised)
                h_results.append(h_optimised)
                w_results.append(w_optimised)
                b_results.append(b_optimised)
                success_count += 1
                print("Optimisation successful.\n")

                # Stop if the required number of successes is achieved
                if success_count >= max_successes:
                    break
            else:
                print("Optimisation unsuccessful, restarting...\n")

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
            print("Statistical Analysis of w_optimised and b_optimised:")
            print(f"Mean of w_optimised: {self.w_optimised}")
            print(f"Standard Deviation of w_optimised: {self.w_std}")
            print(f"Mean of b_optimised: {self.b_optimised}")
            print(f"Standard Deviation of b_optimised: {self.b_std}")

            print(f"Optimisation completed with averaged results after {success_count} successes.\n")

            self.scatter_plots()
        else:
            print("Optimisation was unsuccessful after maximum attempts.\n")

    @timer
    def _optimise_model(self, X_subset, J_subset, h_subset, w_subset, b_subset, subset_index):
        """
        Optimise a subset of the Ising model using 'L-BFGS-B'.

        Args:
        subset_index (int): Index of the subset being optimised.
        """
        d = J_subset.shape[0]  # Number of spins (currencies)

        # Flatten the J matrix and h vector for the optimisation
        x0 = np.hstack([J_subset[np.triu_indices(d, k=1)], h_subset, w_subset, np.array([b_subset])])

        def objective_function(x):
            J, h, w, b = self._reconstruct_parameters(x, d)
            log_likelihood, grad_J, grad_h, grad_w, grad_b = self._log_PL_and_gradients(J, h, w, b, X_subset)
            grad = np.hstack([grad_J[np.triu_indices(d, k=1)], grad_h, grad_w, np.array([grad_b])])

            return log_likelihood, grad

        # Use 'L-BFGS-B' for optimisation with Jacobian
        res = minimize(objective_function, x0, method='L-BFGS-B', jac=True)

        # Reconstruct the optimised J matrix and h vector from the optimisation result
        J_optimised, h_optimised, w_optimised, b_optimised = self._reconstruct_parameters(res.x, d)

        return J_optimised, h_optimised, w_optimised, b_optimised, res.success

    def optimise_all_subsets(self, max_attempts):
        self._divide_into_subsets()
        optimised = False
        for attempt in range(max_attempts):
            print(f"Attempt {attempt + 1} of optimisation")
            # Weight initialisation in the training of neural networks
            d = self.X.shape[1]
            self.w = np.random.normal(0, np.sqrt(2 / d), size=(d,))  # Xavier initialisation
            self.b = np.random.normal(0, np.sqrt(2 / d))  # He initialisation

            J_initial = {key: self.J[np.ix_(indices, indices)] for key, indices in self.subsets_indices.items()}
            h_initial = {key: self.h[indices] for key, indices in self.subsets_indices.items()}
            w_initial = {key: self.w[indices] for key, indices in self.subsets_indices.items()}
            optimised_results = {}
            for index, (key, data_subset) in enumerate(self.data_subsets.items(), 1):
                J_opt, h_opt, w_opt, b_opt, success = self._optimise_model(data_subset, J_initial[key], h_initial[key],
                                                                           w_initial[key], self.b, index)
                if not success:
                    break
                optimised_results[key] = (J_opt, h_opt, w_opt, b_opt, success)
            else:
                # If all subsets are successfully optimised, proceed to next step

                # Extract a portion of self.J for the combined matrix
                size_1_2 = optimised_results['subset_1'][0].shape[0] + optimised_results['subset_2'][0].shape[0]
                J_initial_1_2 = self.J[:size_1_2, :size_1_2]
                J_initial_3_4 = self.J[size_1_2:, size_1_2:]

                # Combine subsets 1 and 2, and subsets 3 and 4
                J_combined_1_2, h_combined_1_2, w_combined_1_2, b_averaged_1_2 = (
                    self._combine_J_matrices(J_initial_1_2, optimised_results['subset_1'][0], optimised_results['subset_2'][0]),
                    np.hstack((optimised_results['subset_1'][1], optimised_results['subset_2'][1])),
                    np.hstack((optimised_results['subset_1'][2], optimised_results['subset_2'][2])),
                    np.mean((optimised_results['subset_1'][3], optimised_results['subset_2'][3]), axis=0))

                J_combined_3_4, h_combined_3_4, w_combined_3_4, b_averaged_3_4 = (
                    self._combine_J_matrices(J_initial_3_4, optimised_results['subset_3'][0], optimised_results['subset_4'][0]),
                    np.hstack((optimised_results['subset_3'][1], optimised_results['subset_4'][1])),
                    np.hstack((optimised_results['subset_3'][2], optimised_results['subset_4'][2])),
                    np.mean((optimised_results['subset_3'][3], optimised_results['subset_4'][3]), axis=0))

                # Optimise the combined larger subsets
                J_opt_1_2, h_opt_1_2, w_opt_1_2, b_opt_1_2, success_1_2 = self._optimise_model(
                    self.X[:, self.subsets_indices['subset_1'] + self.subsets_indices['subset_2']],
                    J_combined_1_2, h_combined_1_2, w_combined_1_2, b_averaged_1_2, '1_2')
                if not success_1_2:
                    continue

                J_opt_3_4, h_opt_3_4, w_opt_3_4, b_opt_3_4, success_3_4 = self._optimise_model(
                    self.X[:, self.subsets_indices['subset_3'] + self.subsets_indices['subset_4']],
                    J_combined_3_4, h_combined_3_4, w_combined_3_4, b_averaged_3_4, '3_4')
                if not success_3_4:
                    continue

                # Final optimisation with the entire dataset
                J_combined_full = self._combine_J_matrices(self.J, J_opt_1_2, J_opt_3_4)
                h_combined_full = np.hstack((h_opt_1_2, h_opt_3_4))
                w_combined_full = np.hstack((w_opt_1_2, w_opt_3_4))
                b_averaged_full = np.mean((b_opt_1_2, b_opt_3_4))

                self.J_optimised, self.h_optimised, self.w_optimised, self.b_optimised, success_full = self._optimise_model(
                    self.X, J_combined_full, h_combined_full, w_combined_full, b_averaged_full, 'full')

                # Check success for final optimisation
                if not success_full:
                    continue

                print("Final optimisation completed.\n")
                return success_full
        print("Optimisation was unsuccessful after maximum attempts.\n")
        return optimised

    def _divide_into_subsets(self, num_subsets=4):
        """
        Divide the data set of observed currencies into approximately equal subsets.

        Args:
        num_subsets (int): The number of subsets to divide the data into.
        """
        num_currencies = len(self.symbols)
        subset_size = num_currencies // num_subsets
        extra = num_currencies % num_subsets

        self.subsets_indices = {}
        start = 0
        for i in range(num_subsets):
            end = start + subset_size + (1 if i < extra else 0)
            self.subsets_indices[f'subset_{i + 1}'] = list(range(start, end))
            start = end

        self.data_subsets = {key: self.X[:, indices] for key, indices in self.subsets_indices.items()}

        # Print information about how the currencies have been divided into subsets
        for key, indices in self.subsets_indices.items():
            print(f"{key}: Currencies {indices[0] + 1} to {indices[-1] + 1}")

    @staticmethod
    def _log_PL_and_gradients(J, h, w, b, X):
        n, d = X.shape  # Number of samples and dimensions

        S_ij = X @ J + h  # Broadcasting h
        full_S_ik = X @ w + b  # for all samples
        S_ik = X @ w.reshape(-1, 1) + b - X * w  # Broadcasting w

        # Calculate the components used in the log likelihood and gradients
        exp_pos_Sij = np.exp(S_ij)
        exp_neg_Sij = 1 / exp_pos_Sij
        cosh_pos = np.cosh(S_ik + w)
        cosh_neg = np.cosh(S_ik - w)
        sinh_pos = np.sinh(S_ik + w)
        sinh_neg = np.sinh(S_ik - w)
        denominator = exp_pos_Sij * cosh_pos + exp_neg_Sij * cosh_neg

        x_times_Sij = X * S_ij  # Sum over dimensions for each sample
        log_numerator = np.log(np.cosh(full_S_ik.reshape(-1, 1)))  # Reshape to 2D
        log_denominator = np.log(denominator)  # Sum over dimensions for each sample

        # Combine the terms and sum over all samples to get the scalar log-likelihood
        log_likelihood = np.sum(x_times_Sij + log_numerator - log_denominator)

        # Gradients for h, w, and b
        numerator_h = -exp_neg_Sij * cosh_neg + exp_pos_Sij * cosh_pos
        grad_h = np.sum(X - numerator_h / denominator, axis=0)

        numerator_w = -exp_neg_Sij * sinh_neg + exp_pos_Sij * sinh_pos
        grad_w = np.sum(X.T * np.tanh(full_S_ik.T) - (numerator_w / denominator).T, axis=1)

        numerator_b = exp_neg_Sij * sinh_neg + exp_pos_Sij * sinh_pos
        grad_b = np.sum(np.tanh(full_S_ik.reshape(-1, 1)) - (numerator_b / denominator))

        # Gradient for J, excluding the diagonal
        mask = np.ones_like(J) - np.eye(d)
        numerator_J = -exp_neg_Sij * cosh_neg + exp_pos_Sij * cosh_pos
        grad_J = ((X - numerator_J / denominator).T @ X) * mask
        grad_J = grad_J + grad_J.T  # Symmetrise

        # Return the negative likelihood and gradients for minimisation
        return -log_likelihood / n, -grad_J / n, -grad_h / n, -grad_w / n, -grad_b / n

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
        J = J + J.T  # Symmetrise the J matrix

        return J, h, w, b

    def _combine_J_matrices(self, J_initial, *matrices):
        """
        Combine smaller J matrices into a larger J matrix using portions of self.J.
        
        Args:
        J_initial (np.ndarray): Initial J matrix to use as a template.
        *matrices: Variable number of matrices to combine into J_initial.
        """
        J_combined = np.copy(J_initial)  # Create a copy of J_initial to modify

        current_index = 0
        for m in matrices:
            m_size = m.shape[0]
            J_combined[current_index:current_index + m_size, current_index:current_index + m_size] = m
            current_index += m_size
            
        return J_combined

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
        # Extended J matrix
        J_extended = np.zeros((27, 27))
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