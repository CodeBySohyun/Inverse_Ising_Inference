import numpy as np
import pandas as pd
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
        self._divide_into_subsets()

    def _divide_into_subsets(self):
        """
        Divide the data set of observed currencies into approximately equal subsets.
        """
        num_currencies = len(self.symbols)
        subset_size = num_currencies // 4
        extra = num_currencies % 4

        self.subsets_indices = {}
        start = 0
        for i in range(4):
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

        S_ij = X @ J - X * np.diag(J) + h  # Broadcasting h and subtracting k=j terms
        full_S_ik = X @ w + b  # for all samples
        S_ik = X @ w.reshape(-1, 1) + b - X * w  # Broadcasting w

        # Calculate the components used in the log likelihood and gradients
        exp_pos_Sij = np.exp(S_ij)
        exp_neg_Sij = 1 / exp_pos_Sij
        cosh_pos = np.cosh(w + S_ik)
        cosh_neg = np.cosh(w - S_ik)
        sinh_pos = np.sinh(w + S_ik)
        sinh_neg = np.sinh(w - S_ik)
        denominator = exp_pos_Sij * cosh_pos + exp_neg_Sij * cosh_neg

        x_times_Sij = X * S_ij  # Sum over dimensions for each sample
        log_numerator = np.log(np.cosh(full_S_ik.reshape(-1, 1)))  # Reshape to 2D
        log_denominator = np.log(denominator)  # Sum over dimensions for each sample

        # Combine the terms and sum over all samples to get the scalar log-likelihood
        log_likelihood = (x_times_Sij + log_numerator - log_denominator).sum()

        # Gradients
        numerator_h = -exp_neg_Sij * cosh_neg + exp_pos_Sij * cosh_pos
        grad_h = (X - numerator_h / denominator).sum(axis=0)

        numerator_w = -exp_neg_Sij * sinh_neg + exp_pos_Sij * sinh_pos
        grad_w = (X.T * (np.tanh(full_S_ik.T) - (numerator_w / denominator).T)).sum(axis=1)

        numerator_b = exp_neg_Sij * sinh_neg + exp_pos_Sij * sinh_pos
        grad_b = (np.tanh(full_S_ik.reshape(-1, 1)) - (numerator_b / denominator)).sum(axis=0).sum()

        # Gradient for J, excluding the diagonal
        mask = np.ones_like(J) - np.eye(d)
        numerator_J = -exp_neg_Sij * cosh_neg + exp_pos_Sij * cosh_pos
        grad_J = ((X - numerator_J / denominator).T @ X) * mask
        grad_J = grad_J + grad_J.T  # Symmetrise

        # Return the negative likelihood and gradients for minimisation
        return -log_likelihood, -grad_J, -grad_h, -grad_w, -grad_b

    @staticmethod
    def _log_pseudolikelihood(J, h, w, b, X):
        S_ij = X @ J - X * np.diag(J) + h  # Broadcasting h and subtracting k=j terms
        full_S_ik = X @ w + b  # for all samples
        S_ik = X @ w.reshape(-1, 1) + b - X * w  # Broadcasting w

        # Calculate the components used in the log likelihood and gradients
        exp_pos_Sij = np.exp(S_ij)
        exp_neg_Sij = 1 / exp_pos_Sij
        cosh_pos = np.cosh(w + S_ik)
        cosh_neg = np.cosh(w - S_ik)
        denominator = exp_pos_Sij * cosh_pos + exp_neg_Sij * cosh_neg

        x_times_Sij = X * S_ij  # Sum over dimensions for each sample
        log_numerator = np.log(np.cosh(full_S_ik.reshape(-1, 1)))  # Reshape to 2D
        log_denominator = np.log(denominator)  # Sum over dimensions for each sample

        # Combine the terms and sum over all samples to get the scalar log-likelihood
        log_likelihood = (x_times_Sij + log_numerator - log_denominator).sum()

        # Return the negative likelihood and gradients for minimisation
        return -log_likelihood

    @timer
    def _optimise_model(self, X_subset, J_subset, h_subset, w_subset, b_subset, subset_index, method='Powell'):
        """
        Optimise a subset of the Ising model using the specified optimisation method.
        Useful derivative methods include Powell, COBYLA and trust-constr (jac='2-point')

        Args:
        subset_index (int): Index of the subset being optimised.
        method (str): The optimisation method to use. Default is 'Powell'.
        """
        d = J_subset.shape[0]  # Number of spins (currencies)

        # Flatten the J matrix and h vector for the optimisation
        x0 = np.concatenate([J_subset[np.triu_indices(d, k=1)], h_subset, w_subset, np.array([b_subset])])

        if method == 'Powell':
            def objective_function(x):
                """
                'Powell' is a direction-set method, which means it performs one-dimensional minimisation along multiple directions in sequence.
                Often more efficient than Nelder-Mead, especially for higher-dimensional problems.
                More suitable for problems where the objective function varies smoothly.
                Like Nelder-Mead, it doesn't require derivatives but is generally more robust in higher dimensions.
                """
                J, h, w, b = self._reconstruct_parameters(x, d)
                log_likelihood = self._log_pseudolikelihood(J, h, w, b, X_subset)

                return log_likelihood

            # Use a derivative-free optimisation method such as 'Powell'
            res = minimize(objective_function, x0, method='Powell')

        elif method == 'L-BFGS-B':
            def objective_function_2(x):
                J, h, w, b = self._reconstruct_parameters(x, d)
                log_likelihood, grad_J, grad_h, grad_w, grad_b = self._log_PL_and_gradients(J, h, w, b, X_subset)
                grad = np.concatenate([grad_J[np.triu_indices(d, k=1)], grad_h, grad_w, np.array([grad_b])])

                return log_likelihood, grad

            # Use 'L-BFGS-B' for optimisation with Jacobian
            res = minimize(objective_function_2, x0, method='L-BFGS-B', jac=True)

        # Reconstruct the optimised J matrix and h vector from the optimisation result
        J_optimised, h_optimised, w_optimised, b_optimised = self._reconstruct_parameters(res.x, d)

        # Print the optimisation results for the subset
        if res.success:
            print(f"Optimised J matrix for subset {subset_index}: \n{J_optimised.round(3)}")
            print(f"Optimised h vector for subset {subset_index}: \n{h_optimised.round(3)}")
            print(f"Optimised w vector for subset {subset_index}: \n{w_optimised.round(3)}")
            print(f"Optimised b scalar for subset {subset_index}: {round(b_optimised, 3)}")
            print(f"Optimisation successful for subset {subset_index}: {res.success}")

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
        J = J + J.T  # Symmetrise the J matrix

        return J, h, w, b

    @staticmethod
    def _combine_J_matrices(*matrices):
        """
        Combine smaller J matrices into a larger J matrix.
        """
        size = sum(m.shape[0] for m in matrices)
        J_combined = np.zeros((size, size))

        current_index = 0
        for m in matrices:
            m_size = m.shape[0]
            J_combined[current_index:current_index + m_size, current_index:current_index + m_size] = m
            current_index += m_size

        return J_combined

    @staticmethod
    def _combine_h_vectors(*vectors):
        """
        Combine smaller h vectors into a larger h vector.
        """
        return np.concatenate(vectors)

    @staticmethod
    def _combine_w_vectors(*vectors):
        """
        Combine smaller w vectors into a larger w vector.
        """
        return np.concatenate(vectors)

    @staticmethod
    def _average_b_values(*values):
        """
        Average b scalar values.
        """
        return np.mean(values)

    def optimise_all_subsets(self, max_attempts):
        for attempt in range(max_attempts):
            print(f"Attempt {attempt + 1} of subset optimisation")

            # Weight initialisation in the training of neural networks
            d = self.X.shape[1]
            self.w = np.random.normal(0, np.sqrt(2 / d), size=(d,))  # Xavier initialisation
            self.b = np.random.normal(0, np.sqrt(2 / d))  # He initialisation

            print("Optimising initial subsets...\n")
            J_initial = {key: self.J[np.ix_(indices, indices)] for key, indices in self.subsets_indices.items()}
            h_initial = {key: self.h[indices] for key, indices in self.subsets_indices.items()}
            w_initial = {key: self.w[indices] for key, indices in self.subsets_indices.items()}
            optimised_results = {}
            for index, (key, data_subset) in enumerate(self.data_subsets.items(), 1):
                J_opt, h_opt, w_opt, b_opt, success = self._optimise_model(data_subset, J_initial[key], h_initial[key],
                                                                           w_initial[key], self.b, index)
                if not success:
                    print(f"Subset {key} failed to optimise. Restarting optimisation.")
                    break
                optimised_results[key] = (J_opt, h_opt, w_opt, b_opt, success)
                print(f"Subset {key} optimised.\n")
            else:
                # If all subsets are successfully optimised, proceed to next step
                print("Combining into larger subsets for further optimisation...\n")
                # Combine subsets 1 and 2, and subsets 3 and 4
                J_combined_1_2, h_combined_1_2, w_combined_1_2, b_averaged_1_2 = (
                    self._combine_J_matrices(optimised_results['subset_1'][0], optimised_results['subset_2'][0]),
                    self._combine_h_vectors(optimised_results['subset_1'][1], optimised_results['subset_2'][1]),
                    self._combine_w_vectors(optimised_results['subset_1'][2], optimised_results['subset_2'][2]),
                    self._average_b_values(optimised_results['subset_1'][3], optimised_results['subset_2'][3]))

                J_combined_3_4, h_combined_3_4, w_combined_3_4, b_averaged_3_4 = (
                    self._combine_J_matrices(optimised_results['subset_3'][0], optimised_results['subset_4'][0]),
                    self._combine_h_vectors(optimised_results['subset_3'][1], optimised_results['subset_4'][1]),
                    self._combine_w_vectors(optimised_results['subset_3'][2], optimised_results['subset_4'][2]),
                    self._average_b_values(optimised_results['subset_3'][3], optimised_results['subset_4'][3]))

                # Optimise the combined larger subsets
                J_opt_1_2, h_opt_1_2, w_opt_1_2, b_opt_1_2, success_1_2 = self._optimise_model(
                    self.X[:, self.subsets_indices['subset_1'] + self.subsets_indices['subset_2']],
                    J_combined_1_2, h_combined_1_2, w_combined_1_2, b_averaged_1_2, '1_2')
                # Check success for combined subsets
                if not success_1_2:
                    print("Combined subset 1_2 optimisation failed. Restarting optimisation.")
                    continue

                J_opt_3_4, h_opt_3_4, w_opt_3_4, b_opt_3_4, success_3_4 = self._optimise_model(
                    self.X[:, self.subsets_indices['subset_3'] + self.subsets_indices['subset_4']],
                    J_combined_3_4, h_combined_3_4, w_combined_3_4, b_averaged_3_4, '3_4')
                # Check success for combined subsets
                if not success_3_4:
                    print("Combined subset 3_4 optimisation failed. Restarting optimisation.")
                    continue
                print("Larger subsets optimised.\n")

                # Final optimisation with the entire dataset
                print("Performing final optimisation with the entire dataset...\n")
                J_combined_full = self._combine_J_matrices(J_opt_1_2, J_opt_3_4)
                h_combined_full = self._combine_h_vectors(h_opt_1_2, h_opt_3_4)
                w_combined_full = self._combine_w_vectors(w_opt_1_2, w_opt_3_4)
                b_averaged_full = self._average_b_values(b_opt_1_2, b_opt_3_4)

                self.J_optimised, self.h_optimised, self.w_optimised, self.b_optimised, success_full = self._optimise_model(
                    self.X, J_combined_full, h_combined_full, w_combined_full, b_averaged_full, 'full')

                # Check success for final optimisation
                if not success_full:
                    print("Final optimisation failed. Restarting optimisation.")
                    continue

                print("Final optimisation completed.\n")
                return success_full

        print("Optimisation was unsuccessful after maximum attempts.")
        return None, None, None, None

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
        return Z_predicted

    def train(self, max_iterations=10, convergence_threshold=0.01):
        """
        Optimisation and Prediction Process

        1. 'convergence_threshold' is a parameter representing the fraction of elements in Z that must change for the algorithm to be considered not yet converged.
        2. 'change_threshold' translates this fraction into an actual number of elements.
        3. After each iteration, the method checks how many elements in Z have changed ('num_changed') compared to the previous iteration.
        4. If 'num_changed' is less than or equal to 'change_threshold', the method considers the algorithm to have converged and stops the iterations.
        """
        previous_Z = np.copy(self.Z)
        num_elements = len(self.Z)
        change_threshold = int(convergence_threshold * num_elements)

        success = self.optimise_all_subsets(max_iterations)
        if success:
            for iteration in range(max_iterations):
                print(f"Iteration: {iteration + 1}")

                self.Z = self._predict_Z()

                # Check for convergence
                num_changed = np.sum(self.Z != previous_Z)
                print(f"Number of Z elements changed in iteration {iteration + 1}: {num_changed}")

                if num_changed <= change_threshold:
                    print(f"Convergence reached in iteration {iteration}")
                    break

                previous_Z = np.copy(self.Z)
        else:
            print("Restart training")

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

        h_df = pd.DataFrame({'h': self.h_optimised}, index=self.symbols)
        h_df.to_csv(h_path, index=True, header=False)
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
