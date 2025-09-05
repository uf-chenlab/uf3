from typing import List, Dict, Collection, Tuple
import os
import warnings
import numpy as np
import pandas as pd
import ndsplines
from uf3.representation import bspline, process
from uf3.data import io
from uf3.data import composition
from uf3.util import json_io
from uf3.util import parallel
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from concurrent.futures import as_completed
from tqdm import tqdm
import gc
import psutil, os
from sklearn.utils.extmath import randomized_svd
import threading
import multiprocessing
from queue import Queue
import time
import joblib
from joblib import Parallel, delayed
def log_total_memory(label=""):
    parent = psutil.Process(os.getpid())
    mem = parent.memory_info().rss
    for child in parent.children(recursive=True):
        try:
            mem += child.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    print(f"[MEMORY] {label} | Total RSS: {mem / 1024 ** 2:.2f} MB")
def print_mem():
    process = psutil.Process(os.getpid())
    print(f"RSS Memory: {process.memory_info().rss / 1e6:.2f} MB")

class VarianceRecorder:
    """Convenience class for computing online variance and mean"""
    def __init__(self, mean=0, std=0, n=0):
        self.mean = mean
        self.std = std
        self.n = int(n)
    def update_manual(self, mean, std, n):
        if self.n == 0:
            self.mean = mean
            self.std = std
            self.n = n
            return self.mean, self.std, self.n
        else:
            batch_std = std
            batch_mean = mean
            m = float(self.n)
            n = n
            std = (m / (m + n) * self.std**2
                   + n / (m + n) * batch_std**2
                   + m * n / (m + n)**2 * (self.mean - batch_mean)**2)
            self.std = np.sqrt(std)
            self.mean = m / (m + n) * self.mean + n / (m + n) * batch_mean
            self.n += n
            return self.mean, self.std, self.n
    def update(self, batch: Collection) -> Tuple:
        """
        Args:
            batch (list or np.ndarray): n-dimensional data. For speed purposes,
                dimensions are not checked for compatibility so caution
                is advised when working with multidimensional data.f
                Statistics are computed along the first axis.

        Returns:
            (current mean, current standard deviation, current entry count)
        """
        if self.n == 0:
            self.mean = np.mean(batch, axis=0)
            self.std = np.std(batch, axis=0)
            self.n = len(batch)
            return self.mean, self.std, self.n
        else:
            batch_std = np.std(batch, axis=0)
            batch_mean = np.mean(batch, axis=0)
            m = float(self.n)
            n = len(batch)
            std = (m / (m + n) * self.std**2
                   + n / (m + n) * batch_std**2
                   + m * n / (m + n)**2 * (self.mean - batch_mean)**2)
            self.std = np.sqrt(std)
            self.mean = m / (m + n) * self.mean + n / (m + n) * batch_mean
            self.n += n
            return self.mean, self.std, self.n

    def update_with_components(self, df, keys=None):
        """Wrapper for dataframe with multiple columns of interest"""
        if keys is None:
            keys = ["fx", "fy", "fz"]
        batch = []
        for j, *components in df[keys].itertuples():
            if any([component is np.nan for component in components]):
                continue
            if np.ndim(components) > 1:  # if components are not scalars
                components = list(np.concatenate(components))
            batch.extend(components)
        self.update(batch)
        return self.mean, self.std, self.n


class BasicLinearModel:
    """
    Base class for linear regression.
    """
    def __init__(self,
                 regularizer: np.ndarray = None):
        """
        Args:
            regularizer (np.ndarray): regularization matrix.
        """
        self.coefficients = None
        self.regularizer = regularizer

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            ridge_penalty: float = 1e-8,
            ):
        """
        Direct solution to linear least squares with LU decomposition.

        Args:
            x (np.ndarray): input matrix of shape (n_samples, n_features).
            y (np.ndarray): output vector of length n_samples.
            ridge_penalty (float): magnitude of ridge penalty. Ignored
                if self.regularizer is set at initialization.
        """
        gram, ordinate = moore_penrose_components(x, y)
        if self.regularizer is None:
            regularizer = np.eye(len(gram)) * ridge_penalty
        else:
            regularizer = self.regularizer
        regularizer = np.dot(regularizer.T, regularizer)
        coefficients = lu_factorization(gram + regularizer, ordinate)
        self.coefficients = coefficients

    def predict(self, x: np.ndarray):
        """
        Predict using fit coefficients.

        Args:
            x (np.ndarray): input matrix of shape (n_samples, n_features).

        Returns:
            predictions (np.ndarray): vector of predictions.
        """
        predictions = np.dot(x, self.coefficients)
        return predictions

    def score(self, x, y, weights=None, normalize=True):
        """
        Evaluate score (negative error metric).

        Args:
            x (np.ndarray): input matrix of shape (n_samples, n_features).
            y (np.ndarray): output vector of length n_samples.
            weights (np.ndarray): sample weights (optional).
            normalize (bool): whether to normalize by the std of y.

        Returns:
            score (float): negative weighted root-mean-square-error.
        """
        n_features = len(x[0])
        if weights is not None:
            w_matrix = np.eye(n_features) * np.sqrt(weights)
            x = np.dot(w_matrix, x)
            y = np.dot(w_matrix, y)
        predictions = self.predict(x)
        score = -rmse_metric(y, predictions)
        if normalize:
            score /= np.std(y)
        return score


class WeightedLinearModel(BasicLinearModel):
    """
    Handler class for regularized linear least squares using energies and
    forces and basis set provided by bspline.BsplineBasis.
    """
    def __init__(self,
                 bspline_config,
                 regularizer=None,
                 data_coverage=None,
                 **params):
        super().__init__(regularizer)
        self.bspline_config = bspline_config
        n_basis = np.sum(self.bspline_config.get_feature_partition_sizes())
        if data_coverage is not None:
            if len(data_coverage) == n_basis:
                self.data_coverage = data_coverage
            else:
                raise ValueError(
                    f"Incorrect data_coverage shape: "
                    f"{len(data_coverage)} != {n_basis}"
                )
        else:
            self.data_coverage = np.zeros(n_basis, dtype=bool)

        if self.regularizer is None:
            # initialize regularizer matrix if unspecified.
            self.set_params(**params)
        self._acc_gram_e = None
        self._acc_gram_f = None
        self._acc_ord_e = None
        self._acc_ord_f = None
        self._acc_e_variance = VarianceRecorder()
        self._acc_f_variance = VarianceRecorder()

    def set_params(self, **params):
        """Set parameters from keyword arguments. Initializes
            regularizer with default parameters if unspecified."""
        if "bspline_config" in params:
            self.bspline_config = params["bspline_config"]
        if "regularizer" in params:
            self.regularizer = params["regularizer"]
        elif self.regularizer is None:
            reg_params = {k: v for k, v in params.items()
                          if isinstance(v, (int, float, np.floating))}
            reg = self.bspline_config.get_regularization_matrix(**reg_params)
            self.regularizer = reg

    @staticmethod
    def from_config(config):
        return WeightedLinearModel.from_dict(config)

    @staticmethod
    def from_dict(config, bspline_config=None):
        if bspline_config is None:
            bspline_config = bspline.BSplineBasis.from_dict(config)
        else:
            bspline_config = bspline_config
        regularizer = config.get("regularizer", None)
        data_coverage = config.get("data_coverage", None)
        model = WeightedLinearModel(bspline_config,
                                    regularizer=regularizer,
                                    data_coverage=data_coverage)
        model.load(solution=config, bspline_config=bspline_config)
        return model

    @staticmethod
    def from_json(filename, bspline_config=None):
        """Load model (coefficients and knots map) from json file."""
        dump = json_io.load_interaction_map(filename)
        return WeightedLinearModel.from_dict(dump, bspline_config=bspline_config)
    

    def as_dict(self):
        solution = arrange_coefficients(self.coefficients, self.bspline_config)
        for trio in self.bspline_config.interactions_map.get(3, []):
            solution[trio] = self.bspline_config.decompress_3B(solution[trio],
                                                               trio)
        knots_map = self.bspline_config.knots_map
        dump = dict(coefficients=solution,
                    knots=knots_map,
                    data_coverage=self.data_coverage,
                    **self.bspline_config.as_dict())
        return dump

    @property
    def n_feats(self):
        return self.bspline_config.n_feats

    @property
    def frozen_c(self):
        return self.bspline_config.frozen_c

    @property
    def col_idx(self):
        return self.bspline_config.col_idx

    @property
    def mask(self):
        return get_freezing_mask(self.n_feats, self.col_idx)

    def __repr__(self):
        if self.coefficients is None:
            fit = "False"
        else:
            fit = "True"
        summary = ["WeightedLinearModel:",
                   f"    Fit: {fit}",
                   self.bspline_config.__repr__()
                   ]
        return "\n".join(summary)

    def __str__(self):
        return self.__repr__()

    def fit_with_gram(self, gram: np.ndarray, ordinate: np.ndarray):
        """
        Intermediate function for direct solution using gram matrix
        and ordinate (Moore-penrose inverse).

        Args:
            gram (np.ndarray): gram matrix (x^T x)
            ordinate (np.ndarray: ordinate (x^T y)
        """
        data_coverage = (np.sum(gram, axis=0) != 0)
        data_coverage = revert_frozen_coefficients(data_coverage,
                                                   self.n_feats,
                                                   self.mask,
                                                   self.frozen_c,
                                                   self.col_idx)
        self.data_coverage = np.logical_or(self.data_coverage, data_coverage)

        regularizer = freeze_regularizer(self.regularizer, self.mask)
        regularizer = np.dot(regularizer.T, regularizer)
        coefficients = lu_factorization(gram + regularizer, ordinate)
        coefficients = revert_frozen_coefficients(coefficients,
                                                  self.n_feats,
                                                  self.mask,
                                                  self.frozen_c,
                                                  self.col_idx)
        self.coefficients = coefficients

    def fit(self,
            x_e: np.ndarray,
            y_e: np.ndarray,
            x_f: np.ndarray = None,
            y_f: np.ndarray = None,
            weight: float = 0.5,
            batch_size=500,
            ):
        """
        Direct solution from input-output pairs corresponding to
        energies and forces, with option to weigh their respective
        contributions.

        Args:
            x_e (np.ndarray): input matrix of shape (n_samples, n_features).
            y_e (np.ndarray): output vector of length n_samples.
            x_f (np.ndarray): input matrix corresponding to forces.
            y_f (np.ndarray): output vector corresponding to forces.
            weight (float): parameter balancing contribution from energies
                vs. forces. Higher values favor energies; defaults to 0.5.
            batch_size: maximum batch size for gram matrix construction.
        """
        x_e, y_e = freeze_columns(x_e,
                                  y_e,
                                  self.mask,
                                  self.frozen_c,
                                  self.col_idx)
        gram_e, ord_e = batched_moore_penrose(x_e, y_e, batch_size=batch_size)
        if x_f is not None:
            energy_weight, force_weight = calc_E_F_weights(len(y_e),
                                                           len(y_f),
                                                           np.std(y_e),
                                                           np.std(y_f))
            x_f, y_f = freeze_columns(x_f,
                                      y_f,
                                      self.mask,
                                      self.frozen_c,
                                      self.col_idx)
            gram_f, ord_f = batched_moore_penrose(x_f,
                                                  y_f,
                                                  batch_size=batch_size)
            gram, ordinate = self.combine_weighted_gram(gram_e, gram_f, ord_e,
                                                        ord_f, energy_weight,
                                                        force_weight, weight)
        else:
            gram = gram_e
            ordinate = ord_e
        self.fit_with_gram(gram, ordinate)

    def combine_weighted_gram(self,
                              gram_e: np.ndarray,
                              gram_f: np.ndarray,
                              ord_e: np.ndarray,
                              ord_f: np.ndarray,
                              energy_weight: float,
                              force_weight: float,
                              weight: float):
        """
        Apply weighting to gram matrices and ordinates for energy and
        force contributions to the fit.

        Args:
            gram_e (np.ndarray): gram matrix (x^T x) for energies.
            gram_f (np.ndarray): gram matrix (x^T x) for forces.
            ord_e (np.ndarray): ordinate (x^T y) for energies.
            ord_f (np.ndarray): ordinate (x^T y) for forces.
            energy_weight: 1 / (# energies * sqrt(Var(energies)))
            force_weight: 1 / (# forces * sqrt(Var(forces)))
            weight (float): parameter balancing contribution from energies
                vs. forces. Higher values favor energies; defaults to 0.5.

        Returns:
            gram (np.ndarray): gram matrix (x^T x) for fitting.
            ordinate (np.ndarray): ordinate (x^T y) for fitting.
        """
        gram = ((weight * energy_weight**2 * gram_e)
                + ((1 - weight) * force_weight**2 * gram_f))
        ordinate = ((weight * energy_weight**2 * ord_e)
                    + ((1 - weight) * force_weight**2 * ord_f))
        return gram, ordinate
    
    @staticmethod
    def process_table_chunked(j,
                            table_names,
                            filenames_list,
                            subset,
                            batch_size,
                            sample_weights,
                            energy_key,
                            model_instance,
                            chunk_size=25000):
        table_name = table_names[j]
        filename = filenames_list[j]

        gram_e, gram_f, ord_e, ord_f = model_instance.initialize_gram_ordinate()
        local_e_variance = VarianceRecorder()
        local_f_variance = VarianceRecorder()

        if not os.path.isfile(filename):
            return None

        def chunk_table_reader(filename, table_name, chunk_size):
            try:
                with pd.HDFStore(filename, mode='r') as store:
                    start = 0
                    while True:
                        df_chunk = store.select(table_name, start=start, stop=start + chunk_size)
                        if df_chunk.empty:
                            break
                        yield df_chunk
                        start += chunk_size
            except Exception as err:
                print(f"Error reading chunks from {filename}/{table_name}: {err}")

        for df_chunk in chunk_table_reader(filename, table_name, chunk_size):
            try:
                keys = df_chunk.index.unique(level=0).intersection(subset)
                if len(keys) == 0:
                    continue
                df_chunk = df_chunk.loc[keys]
                g_e, g_f, o_e, o_f = model_instance.gram_from_df(
                    df_chunk, keys, e_variance=local_e_variance,
                    f_variance=local_f_variance, sample_weights=sample_weights,
                    energy_key=energy_key, batch_size=batch_size
                )
                gram_e += g_e
                gram_f += g_f
                ord_e += o_e
                ord_f += o_f
            except Exception as chunk_err:
                print(f"Error processing chunk from {filename}/{table_name}: {chunk_err}")
            finally:
                del df_chunk, g_e, g_f, o_e, o_f
                gc.collect()

        return gram_e, gram_f, ord_e, ord_f, local_e_variance, local_f_variance

    @staticmethod
    def process_table(j, table_names, filename, subset, batch_size, sample_weights, energy_key, model_instance):
        """
        Helper function to process a single table in parallel.
        """
        table_name = table_names[j]
        df = process.load_feature_db(filename, table_name, subset)
        print_mem()

        if df is None:
            return None

        keys = df.index.unique(level=0)

        local_e_variance = VarianceRecorder()
        local_f_variance = VarianceRecorder()
    
        # Compute gram matrices and ordinates
        gram_e, gram_f, ordinate_e, ordinate_f = model_instance.gram_from_df(
            df, keys,
            e_variance=local_e_variance,
            f_variance=local_f_variance,
            sample_weights=sample_weights,
            energy_key=energy_key,
            batch_size=batch_size
        )
        
         # ensure safe types for threading
        gram_e = np.asarray(gram_e, dtype=np.float64)
        gram_f = np.asarray(gram_f, dtype=np.float64)
        ordinate_e = np.asarray(ordinate_e, dtype=np.float64)
        ordinate_f = np.asarray(ordinate_f, dtype=np.float64)

        return gram_e, gram_f, ordinate_e, ordinate_f, local_e_variance, local_f_variance
    @staticmethod
    def process_keys_stream(df, keys, batch_size, sample_weights, energy_key, model_instance, outlier_config: Dict = None):
        local_e_variance = VarianceRecorder()
        local_f_variance = VarianceRecorder()

        gram_e, gram_f, ordinate_e, ordinate_f = model_instance.streamed_gram_from_df(
            df, keys,
            e_variance=local_e_variance,
            f_variance=local_f_variance,
            sample_weights=sample_weights,
            energy_key=energy_key,
            batch_size=batch_size,
            outlier_config=outlier_config
        )

        return (
            np.asarray(gram_e, dtype=np.float64),
            np.asarray(gram_f, dtype=np.float64),
            np.asarray(ordinate_e, dtype=np.float64),
            np.asarray(ordinate_f, dtype=np.float64),
            local_e_variance,
            local_f_variance
        )
    @staticmethod
    def process_keys(df, keys, batch_size, sample_weights, energy_key, model_instance,outlier_config: Dict = None):
        # local vars for each thread
        local_e_variance = VarianceRecorder()
        local_f_variance = VarianceRecorder()

        gram_e, gram_f, ordinate_e, ordinate_f = model_instance.gram_from_df(
            df, keys,
            e_variance=local_e_variance,
            f_variance=local_f_variance,
            sample_weights=sample_weights,
            energy_key=energy_key,
            batch_size=batch_size,
            outlier_config=outlier_config
        )

        gram_e = np.asarray(gram_e, dtype=np.float64)
        gram_f = np.asarray(gram_f, dtype=np.float64)
        ordinate_e = np.asarray(ordinate_e, dtype=np.float64)
        ordinate_f = np.asarray(ordinate_f, dtype=np.float64)
        return gram_e, gram_f, ordinate_e, ordinate_f, local_e_variance, local_f_variance

    def fit_from_file_parallel(self,
                           filename: str,
                           subset: Collection,
                           weight: float = 0.5,
                           batch_size=500,
                           sample_weights: Dict = None,
                           energy_key="energy",
                           num_cores=1,
                           progress: str = "bar"):
        """
        Parallelized version of fit_from_file with a progress bar.
        """
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)
    
        # Analyze the HDF5 file
        n_tables, _, table_names, _ = io.analyze_hdf_tables(filename)
        gram_e, gram_f, ord_e, ord_f = self.initialize_gram_ordinate()
    
        # Initialize global variance recorders
        e_variance_global = VarianceRecorder()
        f_variance_global = VarianceRecorder()

        # Use ThreadPoolExecutor (or ProcessPoolExecutor) locally on *this* worker
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            # Submit tasks for each table
            futures = []
            for j in range(n_tables):
                future = executor.submit(
                    WeightedLinearModel.process_table,
                    j,
                    table_names,
                    filename,
                    subset,
                    batch_size,
                    sample_weights,
                    energy_key,
                    self
                )
                futures.append(future)

            # Collect results as they complete
            with tqdm(total=n_tables, desc="Processing Tables", unit="table") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    pbar.update()

                    # If table had no matching keys, result is None
                    if result is None:
                        continue

                    (g_e, g_f, o_e, o_f, local_e_variance, local_f_variance) = result

                    # Merge partial Gram/ordinate into the global aggregator
                    gram_e += g_e
                    gram_f += g_f
                    ord_e  += o_e
                    ord_f  += o_f

                    # Merge local variance recorders
                    if local_e_variance.n > 0:
                        e_variance_global.update_manual(
                            local_e_variance.mean,
                            local_e_variance.std,
                            local_e_variance.n
                        )
                    if local_f_variance.n > 0:
                        f_variance_global.update_manual(
                            local_f_variance.mean,
                            local_f_variance.std,
                            local_f_variance.n
                        )

        # Now we have final Gram/ordinate and variance recorders
        energy_weight, force_weight = calc_E_F_weights(
            e_variance_global.n, f_variance_global.n,
            e_variance_global.std, f_variance_global.std
        )

        # Combine Gram matrices (energy and force) and do the final fit
        gram, ordinate = self.combine_weighted_gram(
            gram_e, gram_f, ord_e, ord_f,
            energy_weight, force_weight, weight
        )
        self.fit_with_gram(gram, ordinate)

    def fit_from_file(self,
                      filename: str,
                      subset: Collection,
                      weight: float = 0.5,
                      batch_size=500,
                      sample_weights: Dict = None,
                      energy_key="energy",
                      progress: str = "bar",
                      drop_columns: List[str] = None):
        """
        Accumulate inputs and outputs from batched parsing of HDF5 file
        and compute direct solution via LU decomposition.

        Args:
            filename (str): path to HDF5 file.
            subset (list): list of keys for training.
            weight (float): parameter balancing contribution from energies
                vs. forces. Higher values favor energies; defaults to 0.5.
            batch_size (int): batch size, in rows, for matrix multiplication
                operations in constructing gram matrices.
            sample_weights (dict):
            energy_key (str): column name for energies, default "energy".
            progress (str): style for progress indicators.
            drop_columns (list): list of columns to drop. Used when modifying
                the cutoffs of the feature vectors from HDF5 file. No internal
                checks are performed to see if dropping provided columns produce
                features of the intended cutoffs. Use with Caution.
        """
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)
        n_tables, _, table_names, _ = io.analyze_hdf_tables(filename)
        gram_e, gram_f, ord_e, ord_f = self.initialize_gram_ordinate()
        e_variance = VarianceRecorder()
        f_variance = VarianceRecorder()
        table_iterator = parallel.progress_iter(np.arange(n_tables),
                                                style=progress)
        for j in table_iterator:
            table_name = table_names[j]
            df = process.load_feature_db(filename, table_name)
            keys = df.index.unique(level=0).intersection(subset)
            if len(keys) == 0:
                continue

            if drop_columns != None:
                df.drop(columns=drop_columns,inplace=True)

            intermediates = self.gram_from_df(df,
                                              keys,
                                              e_variance=e_variance,
                                              f_variance=f_variance,
                                              sample_weights=sample_weights,
                                              energy_key=energy_key,
                                              batch_size=batch_size)
            g_e, g_f, o_e, o_f = intermediates
            gram_e += g_e
            gram_f += g_f
            ord_e += o_e
            ord_f += o_f
        energy_weight, force_weight = calc_E_F_weights(e_variance.n,
                                                       f_variance.n,
                                                       e_variance.std,
                                                       f_variance.std)
        gram, ordinate = self.combine_weighted_gram(gram_e,
                                                    gram_f,
                                                    ord_e,
                                                    ord_f,
                                                    energy_weight,
                                                    force_weight,
                                                    weight)
        self.fit_with_gram(gram, ordinate)
        
    def fit_from_files(self,
                      filenames: List,
                      subset: Collection,
                      weight: float = 0.5,
                      batch_size=500,
                      sample_weights: Dict = None,
                      energy_key="energy",
                      progress: str = "bar"):
        """
        Accumulate inputs and outputs from batched parsing of HDF5 file
        and compute direct solution via LU decomposition.
        Args:
            filename (list): List of paths to HDF5 file.
            subset (list): list of keys for training.
            weight (float): parameter balancing contribution from energies
                vs. forces. Higher values favor energies; defaults to 0.5.
            batch_size (int): batch size, in rows, for matrix multiplication
                operations in constructing gram matrices.
            sample_weights (dict):
            energy_key (str): column name for energies, default "energy".
            progress (str): style for progress indicators.
        """
        n_tables = 0
        table_names = []
        filenames_list = []
        for filename in filenames:
            if not os.path.isfile(filename):
                raise FileNotFoundError(filename)
            n_table, _, table_name, _ = io.analyze_hdf_tables(filename)
            n_tables = n_tables + n_table
            table_names = table_names + table_name
            filenames_list = filenames_list + [filename for _ in table_name]
        gram_e, gram_f, ord_e, ord_f = self.initialize_gram_ordinate()
        e_variance = VarianceRecorder()
        f_variance = VarianceRecorder()
        table_iterator = parallel.progress_iter(np.arange(n_tables),
                                                style=progress)
        for j in table_iterator:
            table_name = table_names[j]
            df = process.load_feature_db(filenames_list[j], table_name)
            keys = df.index.unique(level=0).intersection(subset)
            if len(keys) == 0:
                continue
            intermediates = self.gram_from_df(df,
                                              keys,
                                              e_variance=e_variance,
                                              f_variance=f_variance,
                                              sample_weights=sample_weights,
                                              energy_key=energy_key,
                                              batch_size=batch_size)
            g_e, g_f, o_e, o_f = intermediates

            gram_e += g_e
            gram_f += g_f
            ord_e += o_e
            ord_f += o_f
        energy_weight, force_weight = calc_E_F_weights(e_variance.n,
                                                       f_variance.n,
                                                       e_variance.std,
                                                       f_variance.std)
        gram, ordinate = self.combine_weighted_gram(gram_e,
                                                    gram_f,
                                                    ord_e,
                                                    ord_f,
                                                    energy_weight,
                                                    force_weight,
                                                    weight)
        self.fit_with_gram(gram, ordinate)
    
    def fit_from_files_parallel(self, 
                             filenames: List,
                             subset: Collection,
                             weight: float = 0.5,
                             batch_size=25000,
                             sample_weights: Dict = None,
                             energy_key="energy",
                             num_cores=2,
                             progress: str = "bar"):
        gram_e, gram_f, ord_e, ord_f = self.initialize_gram_ordinate()
        e_variance = VarianceRecorder()
        f_variance = VarianceRecorder()

        # Prepares the (filename, table_name) pairs
        table_jobs = []

        for filename in filenames:
            if not os.path.isfile(filename):
                raise FileNotFoundError(filename)
            n_table, _, table_names, _ = io.analyze_hdf_tables(filename)
            for table_name in table_names:
                table_jobs.append((filename, table_name))

        if not table_jobs:
            raise ValueError("No tables found.")

        # Function to load a table
        def load_table(job):
            filename, table_name = job
            try:
                df = process.load_feature_db(filename, table_name, subset)
                if df is not None and len(df.index) > 0:
                    print_mem()
                    return (filename, table_name), df
            except Exception as e:
                print(f"Failed loading {filename}:{table_name}: {e}")
            return None  # Failed load or empty

        # Parallel load all tables
        all_data = {}  # key: (filename, table_name), value: df

        with ThreadPoolExecutor(max_workers=max(1, num_cores)) as executor:
            futures = {executor.submit(load_table, job): job for job in table_jobs}

            if progress == "bar":
                from tqdm import tqdm
                pbar = tqdm(total=len(futures), desc="Loading Data", unit="table")

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    key, df = result
                    all_data[key] = df
                    print_mem()
                if progress == "bar":
                    pbar.update()

            if progress == "bar":
                pbar.close()
        all_df = pd.concat(list(all_data.values()), axis=0, copy=False)
        # Split the loaded data across all cores (not the table names, but the data itself)
        data_splits = [all_df.iloc[idx] for idx in np.array_split(np.arange(len(all_df)), max(1, num_cores))]
        lock = threading.Lock()
        # Function to process a chunk of data (data split across cores)
        def process_data_chunk(data_chunk):
            if data_chunk.empty:
                return

            keys = data_chunk.index.unique(level=0)

            result = WeightedLinearModel.process_keys(
                data_chunk,
                keys,
                batch_size,
                sample_weights,
                energy_key,
                self
            )
            print_mem()
            if result is not None:
                g_e, g_f, o_e, o_f, e_var, f_var = result
                with lock:
                    np.add(gram_e, g_e, out=gram_e)
                    np.add(gram_f, g_f, out=gram_f)
                    np.add(ord_e, o_e, out=ord_e)
                    np.add(ord_f, o_f, out=ord_f)
                    e_variance.update_manual(e_var.mean, e_var.std, e_var.n)
                    f_variance.update_manual(f_var.mean, f_var.std, f_var.n)

        # Run the parallel processing across the cores
        with ThreadPoolExecutor(max_workers=max(1, num_cores)) as executor:
            futures = {executor.submit(process_data_chunk, data_split): i for i, data_split in enumerate(data_splits)}

            with tqdm(total=len(futures), desc="Processing Tables", unit="batch") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                
                    pbar.update()
                    print_mem()

        # After all futures are done, calculate final weights
        energy_weight, force_weight = calc_E_F_weights(
            e_variance.n, f_variance.n, e_variance.std, f_variance.std
        )

        gram, ordinate = self.combine_weighted_gram(
            gram_e, gram_f, ord_e, ord_f, energy_weight, force_weight, weight
        )

        self.fit_with_gram(gram, ordinate)

    def load_tables_parallel(
        self,
        filenames: List[str],
        subset: Collection,
        num_cores: int = 2,
        progress: str = "bar",
        use_cur: bool = False
    ) -> pd.DataFrame:
        """
        Loads every (filename, table) that contains at least one row of the requested
        `subset` **once**, in parallel, and concatenates them into a single DataFrame
        whose index is the original *multi-index* [(filename, table_name), row_id].

        Parameters
        ----------
        filenames : list[str]
            HDF5 files to scan.
        subset : Collection
            Whatever you usually pass to `process.load_feature_db`.
        num_cores : int
            # threads for I/O.
        progress : {"bar", "none"}
            Show a tqdm bar or stay silent.

        Returns
        -------
        pd.DataFrame
            Concatenation of *all* tables (lazy-copy, so memory-cheap).
        """
        # ---- discover jobs ------------------------------------------------------
        table_jobs = []
        for fname in filenames:
            if not os.path.isfile(fname):
                raise FileNotFoundError(fname)
            n_table, _, table_names, _ = io.analyze_hdf_tables(fname)
            table_jobs.extend([(fname, tname) for tname in table_names])

        if not table_jobs:
            raise ValueError("No tables found in the supplied files.")

        # ---- I/O worker ---------------------------------------------------------
        def _load(job):
            fname, tname = job
            try:
                df = process.load_feature_db(fname, tname, subset)
                return df
            except Exception as exc:
                print(f"[WARN] Could not load {fname}:{tname} – {exc}")
            return None

        dfs = []
        with ThreadPoolExecutor(max_workers=max(1, num_cores)) as ex:
            futs = {ex.submit(_load, job): job for job in table_jobs}

            pbar = tqdm(total=len(futs), desc="Loading tables", unit="tbl",
                        disable=(progress != "bar"))
            for fut in as_completed(futs):
                out = fut.result()
                if out is not None:
                    dfs.append(out)
                pbar.update()
            pbar.close()

        if not dfs:
            raise RuntimeError("Every load failed – nothing to fit on.")

        return pd.concat(dfs, axis=0, copy=False)
    def drop_excluded_columns(self, exclude_elements, df):
        if exclude_elements is not None:
            sizes, offsets = self.bspline_config.get_interaction_partitions()
            col_idx = []
            frozen_c = []

            for interaction in sizes:
                if any(el in interaction for el in exclude_elements):
                    offset = offsets[interaction]
                    size = sizes[interaction]
                    col_idx.extend(range(offset, offset + size))
                    frozen_c.extend([0.0] * size)

            self.bspline_config.col_idx = np.array(col_idx, dtype=int)
            self.bspline_config.frozen_c = np.array(frozen_c, dtype=float)

            drop_columns = [
                col for col in df.columns
                if any(el.lower() in col.lower() for el in exclude_elements)
            ]
            df.drop(columns=drop_columns, inplace=True)
        return df
    
    
    def fit_from_dataframe_parallel(
        self,
        df: pd.DataFrame,
        subset: Collection,
        weight: float = 0.5,
        batch_size: int = 1000,
        sample_weights: Dict = None,
        energy_key: str = "energy",
        num_cores: int = 2,
        progress: str = "bar",
        exclude_elements: List[str] = None,
        outlier_config: Dict = None
    ):
        """
        Identical to `fit_from_files_parallel`, except that the data are
        already in memory.
        """
        import os
        print("NumPy config:")
        np.show_config()
        print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
        print("MKL_NUM_THREADS:", os.environ.get("MKL_NUM_THREADS"))
        from threadpoolctl import threadpool_info
        import mkl
        print("MKL threads:", mkl.get_max_threads())
        for lib in threadpool_info():
            print(f"{lib['internal_api']} | threads: {lib['num_threads']} | library: {lib['filepath']}")
            df = self.drop_excluded_columns(exclude_elements, df)
        gram_e, gram_f, ord_e, ord_f = self.initialize_gram_ordinate()
        e_var, f_var = VarianceRecorder(), VarianceRecorder()
        if(num_cores > 4):
            os.environ["MKL_NUM_THREADS"] = "4"
            os.environ["OMP_NUM_THREADS"] = "4"
            num_cores = num_cores//4
        idx = np.arange(len(df))
        splits = np.array_split(idx, max(max(1, num_cores), len(idx)/30000))
        
        data_splits = [df.iloc[i].copy(deep=False) for i in splits]
        del df
        model_config = {
            "mask": self.mask,
            "frozen_c": self.frozen_c,
            "col_idx": self.col_idx,
            "n_elements": len(self.bspline_config.element_list)
        }

        print("cpu count: ", multiprocessing.cpu_count())
        log_total_memory("Before parallel processing")

        # break data_splits into batches of size num_cores
        for i in range(0, len(data_splits), num_cores):
            batch = data_splits[i:i + num_cores]

            results = joblib.Parallel(n_jobs=num_cores, backend="loky")(
                joblib.delayed(process_chunk_serializable)(
                    chunk,
                    model_config,
                    energy_key,
                    batch_size,
                    sample_weights,
                    outlier_config
                ) for chunk in tqdm(batch, desc=f"Processing batch {i//num_cores + 1}", unit="chunk", disable=(progress != "bar"))
            )

            log_total_memory("After batch parallel processing")

            for result in results:
                if result is None:
                    continue
                g_e, g_f, o_e, o_f, e_mean, e_std, e_n, f_mean, f_std, f_n = result

                np.add(gram_e, g_e, out=gram_e)
                np.add(gram_f, g_f, out=gram_f)
                np.add(ord_e, o_e, out=ord_e)
                np.add(ord_f, o_f, out=ord_f)
                e_var.update_manual(e_mean, e_std, e_n)
                f_var.update_manual(f_mean, f_std, f_n)

                del g_e, g_f, o_e, o_f, e_mean, e_std, e_n, f_mean, f_std, f_n

            gc.collect()

        e_w, f_w = calc_E_F_weights(e_var.n, f_var.n, e_var.std, f_var.std)
        gram, ord_ = self.combine_weighted_gram(gram_e, gram_f, ord_e, ord_f, e_w, f_w, weight)
        self.fit_with_gram(gram, ord_)

    def fit_by_interaction(self,
                                         filenames: List[str],
                                         subset: Collection,
                                         target_interactions: List[Tuple[str, ...]],
                                         weight: float = 0.5,
                                         batch_size: int = 25000,
                                         sample_weights: Dict = None,
                                         energy_key: str = "energy",
                                         num_cores: int = 2,
                                         progress: str = "bar"):
        """
        Parallel version: fit only the coefficients for selected interaction terms
        (e.g., [('Al', 'Al'), ('Al', 'Al', 'N')]). All other coefficients are frozen.
        """
        component_sizes, component_offsets = self.bspline_config.get_interaction_partitions()
        n_coeff = self.n_feats

        # Sort interaction keys (normalize)
        sorted_targets = {composition.sort_interaction_symbols(k) for k in target_interactions}

        # Build frozen index mask
        col_idx = []
        frozen_c = []

        for interaction in component_sizes:
            sorted_key = composition.sort_interaction_symbols(interaction)
            if sorted_key not in sorted_targets:
                offset = component_offsets[interaction]
                size = component_sizes[interaction]
                col_idx.extend(range(offset, offset + size))
                frozen_c.extend(self.coefficients[offset:offset + size])

        # Apply freezing
        self.bspline_config.col_idx = np.array(col_idx, dtype=int)
        self.bspline_config.frozen_c = np.array(frozen_c, dtype=float)
        self.col_idx = self.bspline_config.col_idx
        self.frozen_c = self.bspline_config.frozen_c

        # Zero out coefficients of selected interactions (optional reset)
        for interaction in sorted_targets:
            offset = component_offsets[interaction]
            size = component_sizes[interaction]
            self.coefficients[offset:offset + size] = 0.0

        # Use existing parallel fit machinery
        self.fit_from_files_parallel(
            filenames=filenames,
            subset=subset,
            weight=weight,
            batch_size=batch_size,
            sample_weights=sample_weights,
            energy_key=energy_key,
            num_cores=num_cores,
            progress=progress
        )

    def fine_tune_from_dataframe_blend(
        self,
        df: pd.DataFrame,
        subset: Collection,
        alpha: float = 0.1,
        weight: float = 0.5,
        batch_size: int = 25000,
        sample_weights: Dict = None,
        energy_key: str = "energy",
        num_cores: int = 2,
        progress: str = "bar",
        exclude_elements: List[str] = None
    ):
        if self.coefficients is None:
            raise RuntimeError("Model must be fit before fine-tuning.")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be between 0 and 1")

        import os, gc, multiprocessing
        from tqdm import tqdm
        import joblib

        # keep thread pools sane when using process parallelism
        if num_cores > 4:
            os.environ["MKL_NUM_THREADS"] = "4"
            os.environ["OMP_NUM_THREADS"] = "4"
            num_cores = num_cores // 4

        df = self.drop_excluded_columns(exclude_elements, df)

        gram_e, gram_f, ord_e, ord_f = self.initialize_gram_ordinate()
        e_var, f_var = VarianceRecorder(), VarianceRecorder()

        idx = np.arange(len(df))
        splits = np.array_split(idx, max(max(1, num_cores), len(idx) / 30000))
        data_splits = [df.iloc[i].copy(deep=False) for i in splits]
        del df

        # minimal state needed by the worker, avoid pickling the whole model
        model_config = {
            "mask": self.mask,
            "frozen_c": self.frozen_c,
            "col_idx": self.col_idx,
            "n_elements": len(self.bspline_config.element_list)
        }

        # process in batches of size num_cores to bound memory
        for i in range(0, len(data_splits), num_cores):
            batch = data_splits[i:i + num_cores]

            results = joblib.Parallel(n_jobs=num_cores, backend="loky")(
                joblib.delayed(process_chunk_serializable)(
                    chunk,
                    model_config,
                    energy_key,
                    batch_size,
                    sample_weights,
                    None  # outlier_config not used in fine-tune
                )
                for chunk in tqdm(
                    batch,
                    desc=f"Processing batch {i // num_cores + 1}",
                    unit="chunk",
                    disable=(progress != "bar")
                )
            )

            # merge per-process outputs
            for result in results:
                if result is None:
                    continue
                g_e, g_f, o_e, o_f, e_mean, e_std, e_n, f_mean, f_std, f_n = result

                np.add(gram_e, g_e, out=gram_e)
                np.add(gram_f, g_f, out=gram_f)
                np.add(ord_e, o_e, out=ord_e)
                np.add(ord_f, o_f, out=ord_f)

                e_var.update_manual(e_mean, e_std, e_n)
                f_var.update_manual(f_mean, f_std, f_n)

                # free large arrays promptly
                del g_e, g_f, o_e, o_f

            gc.collect()

        # same weighting + solve as before
        e_w, f_w = calc_E_F_weights(e_var.n, f_var.n, e_var.std, f_var.std)
        gram, ordinate = self.combine_weighted_gram(
            gram_e, gram_f, ord_e, ord_f, e_w, f_w, weight
        )

        regularizer = freeze_regularizer(self.regularizer, self.mask)
        regularizer = np.dot(regularizer.T, regularizer)

        coeff_update = lu_factorization(gram + regularizer, ordinate)
        coeff_update = revert_frozen_coefficients(
            coeff_update, self.n_feats, self.mask, self.frozen_c, self.col_idx
        )

        # blended update
        self.coefficients = (1 - alpha) * self.coefficients + alpha * coeff_update

    def gram_from_df(self,
                     df: pd.DataFrame,
                     keys: Collection,
                     e_variance: VarianceRecorder = None,
                     f_variance: VarianceRecorder = None,
                     sample_weights: Dict = None,
                     energy_key: str = "energy",
                     batch_size: int = 500,
                     outlier_config: Dict = None):
        """
        Extract inputs and outputs from dataframe and compute
        moore-penrose components (gram matrices and ordinates).

        Args:
            df (pd.DataFrame): DataFrame of energy/force features.
            keys (list): keys to query from df (e.g. training subset).
            e_variance (VarianceRecorder): handler for accumulating
                statistics for energies (mean and variance).
            f_variance (VarianceRecorder): handler for accumulating
                statistics for forces (mean and variance).
            sample_weights (dict):
            energy_key (str): column name for energies, default "energy".
            batch_size (int): batch size, in rows, for matrix multiplication
                operations in constructing gram matrices.
        """
        n_elements = len(self.bspline_config.element_list)
        print("enter gram from df")
        print_mem()
        x_e, y_e, x_f, y_f = dataframe_to_tuples(df,
                                                 n_elements=n_elements,
                                                 energy_key=energy_key,
                                                 sample_weights=sample_weights)
        if len(x_e) == 0 or len(x_f) == 0:
            return None
        print("dataframe to tuples complete")
        print_mem()
        x_e, y_e = freeze_columns(x_e,
                                  y_e,
                                  self.mask,
                                  self.frozen_c,
                                  self.col_idx)
        x_f, y_f = freeze_columns(x_f,
                                  y_f,
                                  self.mask,
                                  self.frozen_c,
                                  self.col_idx)
        if e_variance is not None and f_variance is not None:
            e_variance.update(y_e)
            f_variance.update(y_f)
        print("columns frozen")
        print_mem()
        gram_e, ordinate_e = batched_moore_penrose(x_e,
                                                   y_e,
                                                   batch_size=batch_size)
        gram_f, ordinate_f = batched_moore_penrose(x_f,
                                                   y_f,
                                                   batch_size=batch_size)
        print("batched moore penrose complete")
        return gram_e, gram_f, ordinate_e, ordinate_f


    def batched_predict(self,
                        filename: str,
                        keys: List[str] = None,
                        table_names: List[str] = None,
                        score: bool = True,
                        drop_columns: List[str] = None,
                        use_elements: List[str] = None):
        """
        Extract inputs and outputs from HDF5 file and predict energies/forces.

        Args:
            filename: path to HDF5 file.
            keys (list): keys to query from df (e.g. training subset).
            table_names (list): list of table names in HDF5 to read.
            score (bool): whether to return root mean square error metrics.

        Returns:
            y_e (np.ndarray): target values for energies.
            p_e (np.ndarray): prediction values for energies.
            y_f (np.ndarray): target values for forces.
            p_f (np.ndarray): prediction values for forces.
            rmse_e (np.ndarray): RMSE across energy predictions.
            rmse_e (np.ndarray): RMSE across force predictions.
            drop_columns (list): list of columns to drop. Used when modifying
                the cutoffs of the feature vectors from HDF5 file. No internal
                checks are performed to see if dropping provided columns produce
                features of the intended cutoffs. Use with Caution.
        """
        n_elements = len(self.bspline_config.element_list)
        y_e, p_e, y_f, p_f, ids, force_ids = batched_prediction(self,
                                                filename,
                                                table_names=table_names,
                                                subset_keys=keys,
                                                n_elements=n_elements,
                                                drop_columns=drop_columns,
                                                use_elements=use_elements)
        if score:
            rmse_e = rmse_metric(y_e, p_e)
            rmse_f = rmse_metric(y_f, p_f)
            print(f"RMSE (energy): {rmse_e:.3F}")
            print(f"RMSE (forces): {rmse_f:.3F}")
            return y_e, p_e, y_f, p_f, rmse_e, rmse_f, ids, force_ids
        else:
            return y_e, p_e, y_f, p_f, ids
    
    def batched_predict_multiple_files(self,
                        filenames: [],
                        keys: List[str] = None,
                        table_names: List[str] = None,
                        score: bool = True,
                        drop_columns: List[str] = None,
                        use_elements: List[str] = None):
        """
        Extract inputs and outputs from HDF5 file and predict energies/forces.

        Args:
            filename: path to HDF5 file.
            keys (list): keys to query from df (e.g. training subset).
            table_names (list): list of table names in HDF5 to read.
            score (bool): whether to return root mean square error metrics.

        Returns:
            y_e (np.ndarray): target values for energies.
            p_e (np.ndarray): prediction values for energies.
            y_f (np.ndarray): target values for forces.
            p_f (np.ndarray): prediction values for forces.
            rmse_e (np.ndarray): RMSE across energy predictions.
            rmse_e (np.ndarray): RMSE across force predictions.
            drop_columns (list): list of columns to drop. Used when modifying
                the cutoffs of the feature vectors from HDF5 file. No internal
                checks are performed to see if dropping provided columns produce
                features of the intended cutoffs. Use with Caution.
        """
        n_elements = len(self.bspline_config.element_list)
        y_e, p_e, y_f, p_f, ids, force_labels = batched_prediction_multiple_files(self,
                                                filenames,
                                                table_names=table_names,
                                                subset_keys=keys,
                                                n_elements=n_elements,
                                                drop_columns=drop_columns,
                                                use_elements=use_elements)
        if score:
            rmse_e = rmse_metric(y_e, p_e)
            rmse_f = rmse_metric(y_f, p_f)
            print(f"RMSE (energy): {rmse_e:.3F}")
            print(f"RMSE (forces): {rmse_f:.3F}")
            return y_e, p_e, y_f, p_f, rmse_e, rmse_f, ids, force_labels
        else:
            return y_e, p_e, y_f, p_f

    def to_json(self, filename: str):
        """Save model (coefficients and knots map) to json file."""
        json_io.dump_interaction_map(self.as_dict(),
                                     filename=filename,
                                     write=True)

    def dump(self):
        """Legacy alias"""
        return self.as_dict()

    def load(self,
             solution: Dict = None,
             filename: str = None,
             bspline_config: bspline.BSplineBasis = None
             ):
        """
        Reflatten coefficients (e.g. obtained through arrange_coefficients)
        and load into model for prediction.

        Args:
            solution (dict): dictionary of 1B, 2B, ... terms
                organized as interaction: vector entries.
            filename (str): filename of json dump containing solution.
        """
        if filename is not None:
            if solution is not None:
                warnings.warn("Provided solutions ignored; loading file.")
            solution = json_io.load_interaction_map(filename)
        # if bspline_config is not None:
        #     self.bspline_config = bspline_config
        #     self.n_basis = np.sum(self.bspline_config.get_feature_partition_sizes())
        elif solution is None:
            raise ValueError("Neither solution nor filename were provided.")
        if "coefficients" in solution:
            solution = solution["coefficients"]
        elif "solution" in solution:
            # TODO: proper deprecation
            warnings.warn("'solution' should be renamed to 'coefficients'")
            solution = solution["solution"]
        for key in solution:
            if isinstance(key, tuple):
                sorted_key = composition.sort_interaction_symbols(key)
                if sorted_key != key:
                    solution[sorted_key] = solution[key]
        for trio in self.bspline_config.interactions_map.get(3, []):
            r_min = self.bspline_config.r_min_map[trio]
            r_max = self.bspline_config.r_max_map[trio]
            res = self.bspline_config.resolution_map[trio]
            self.bspline_config.symmetry[trio] = bspline.find_symmetry_3B(trio, r_min, r_max, res)
            print(f'Symmetry for {trio} set to {self.bspline_config.symmetry[trio]}')

        self.bspline_config.update_basis_functions()
        # consistency check with bspline_config
        component_len = self.bspline_config.get_interaction_partitions()[0]
        for pair in self.bspline_config.interactions_map[2]:
            n_target = component_len[pair]
            if pair not in solution:
                warnings.warn(f"{pair} not provided.")
                solution[pair] = np.zeros(n_target)
            n_provided = len(solution[pair])
            if n_provided != n_target:
                raise ValueError(
                    f"Incorrect shape: {pair}, {n_provided} != {n_target}"
                )
        for trio in self.bspline_config.interactions_map.get(3, []):
            n_target = component_len[trio]
            print(f"Checking trio {trio} with target size {n_target}")
            if trio not in solution:
                warnings.warn(f"{trio} not provided.")
            if trio in solution:
                # decompress if necessary
                component = np.array(solution[trio])
                if len(np.shape(component)) > 1:
                    print("model load compressing")
                    vector = self.bspline_config.compress_3B(component,
                                                             trio,
                                                             fitting = False)
                    solution[trio] = vector
            n_provided = len(solution[trio])
            if n_provided != n_target:
                print("nprovided != n_target")
                print(f"Provided: {n_provided}, target: {n_target}")
                component_len = self.bspline_config.get_interaction_partitions(uncompressed=True)[0]
                n_target = component_len[trio]
                if n_provided != n_target:
                    raise ValueError(
                        f"Incorrect shape: {trio}, {n_provided} != {n_target}"
                    )
        flattened_coefficients = []
        for element in self.bspline_config.element_list:
            value = solution[element]
            flattened_coefficients.append([value])
        for degree in range(2, self.bspline_config.degree + 1):
            interactions = self.bspline_config.interactions_map[degree]
            for interaction in interactions:
                values = solution[interaction]
                flattened_coefficients.append(values)
        # self-energies, pair interactions & trio interactions
        n_interactions = len(self.bspline_config.partition_sizes)
        # add self-energy as separate interactions
        n_coefficients = sum(self.bspline_config.partition_sizes)
        if len(flattened_coefficients) != n_interactions:
            error_line = "Incorrect interactions: {} provided, {} expected."
            error_line = error_line.format(len(flattened_coefficients),
                                           n_interactions)
            raise ValueError(error_line)
        flattened_coefficients = np.concatenate(flattened_coefficients)
        if len(flattened_coefficients) != n_coefficients:
            error_line = "Incorrect coefficients: {} provided, {} expected."
            error_line = error_line.format(len(flattened_coefficients),
                                           n_coefficients)
            raise ValueError(error_line)
        self.coefficients = np.array(flattened_coefficients)

    def fix_repulsion_2b(self, pair, r_target=None, min_curvature=2.0):
        components = self.bspline_config.get_interaction_partitions()
        component_sizes, component_offsets = components
        offset = component_offsets[pair]
        n_basis = component_sizes[pair]
        idx_subset = np.arange(offset, offset + n_basis)
        c_subset = self.coefficients[idx_subset]
        coverage = self.data_coverage[idx_subset]
        min_coverage = np.argmax(coverage == True)
        if min_coverage == 0:
            print(f"Coverage is sufficient; no fix applied to {pair}.")
        idx_fix = np.arange(self.bspline_config.leading_trim[2], min_coverage)

        knot_sequence = self.bspline_config.knots_map[pair]
        r_centers = knot_sequence[2: n_basis + 2]
        if r_target is None:
            r_target = r_centers[min_coverage]
        r_centers = r_centers[idx_fix]
        c_new = get_spline_taylor_expansion(r_target,
                                            r_centers,
                                            c_subset,
                                            knot_sequence,
                                            min_curvature=min_curvature)
        print(f"{pair} Correction: adjusted {len(idx_fix)} coefficients.")
        self.coefficients[idx_subset[idx_fix]] = c_new
    def initialize_accumulators(self):
        """
        Initialize the internal accumulators for incremental / fine-tuning fits.
        After calling this, you can call accumulate_gram_from_file multiple times
        and then finalize_fit_from_accumulator() to solve with all data.
        """
        self._acc_gram_e, self._acc_gram_f, self._acc_ord_e, self._acc_ord_f = \
            self.initialize_gram_ordinate()
        self._acc_e_variance = VarianceRecorder()
        self._acc_f_variance = VarianceRecorder()
    
    def initialize_gram_ordinate(self):
        """Initialize empty matrices for gram matrices and ordinates."""
        n_columns = self.n_feats - len(self.col_idx)
        print(f'n_columns: {n_columns} | nfeats: {self.n_feats} | col_idx: {self.col_idx}')
        gram_e = np.zeros((n_columns, n_columns),dtype=np.float64)
        ord_e = np.zeros(n_columns,dtype=np.float64)
        gram_f = np.zeros((n_columns, n_columns),dtype=np.float64)
        ord_f = np.zeros(n_columns,dtype=np.float64)
        return gram_e, gram_f, ord_e, ord_f

    def accumulate_gram_from_file(self,
                                  filename: str,
                                  subset: Collection,
                                  batch_size=2500,
                                  sample_weights: Dict = None,
                                  energy_key="energy"):
        """
        Read partial Gram from an HDF5 file, accumulate into the internal
        increment/fine-tuning accumulators (self._acc_gram_e, etc.).
        Does NOT solve yet. Call finalize_fit_from_accumulator() to solve.
        """
        if self._acc_gram_e is None:
            # if we forgot to init, do it automatically
            self.initialize_accumulators()

        # Summation approach
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)

        n_tables, _, table_names, _ = io.analyze_hdf_tables(filename)
        for j in range(n_tables):
            table_name = table_names[j]
            df = process.load_feature_db(filename, table_name, subset)
            keys = df.index.unique(level=0).intersection(subset)
            if len(keys) == 0:
                continue
            # local (per-table)
            local_e_var = VarianceRecorder()
            local_f_var = VarianceRecorder()

            g_e, g_f, o_e, o_f = self.gram_from_df(
                df, keys,
                e_variance=local_e_var,
                f_variance=local_f_var,
                sample_weights=sample_weights,
                energy_key=energy_key,
                batch_size=batch_size
            )
            # accumulate
            self._acc_gram_e += g_e
            self._acc_gram_f += g_f
            self._acc_ord_e += o_e
            self._acc_ord_f += o_f
            # also accumulate variances
            self._acc_e_variance.update_manual(
                local_e_var.mean, local_e_var.std, local_e_var.n)
            self._acc_f_variance.update_manual(
                local_f_var.mean, local_f_var.std, local_f_var.n)

    def finalize_fit_from_accumulator(self, weight: float = 0.5):
        """
        Once you have accumulated Gram & Ord from multiple data sources
        (via accumulate_gram_from_file), call this to do the final solve.
        """
        if self._acc_gram_e is None:
            raise ValueError("No accumulators initialized. Call "
                             "initialize_accumulators() or "
                             "accumulate_gram_from_file() first.")

        # Compute overall weighting
        e_var = self._acc_e_variance
        f_var = self._acc_f_variance
        energy_weight, force_weight = calc_E_F_weights(
            e_var.n, f_var.n, e_var.std, f_var.std
        )

        gram, ordinate = self.combine_weighted_gram(
            self._acc_gram_e, self._acc_gram_f,
            self._acc_ord_e, self._acc_ord_f,
            energy_weight, force_weight, weight
        )
        self.fit_with_gram(gram, ordinate)
    

    def streamed_gram_from_df(self,
                          df: pd.DataFrame,
                          keys: Collection,
                          e_variance: VarianceRecorder = None,
                          f_variance: VarianceRecorder = None,
                          sample_weights: Dict = None,
                          energy_key: str = "energy",
                          batch_size: int = 500,
                          outlier_config: Dict = None,
                          float_dtype: np.dtype = np.float64):
        """
        Memory-efficient streamed gram matrix computation.
        No full x/y matrices are built.
        """
        df = df.loc[keys]
        n_elements = len(self.bspline_config.element_list)
        n_features = self.n_feats - len(self.col_idx)  # exclude target column

        gram_e = np.zeros((n_features, n_features), dtype=float_dtype)
        ord_e = np.zeros(n_features, dtype=float_dtype)
        gram_f = np.zeros((n_features, n_features), dtype=float_dtype)
        ord_f = np.zeros(n_features, dtype=float_dtype)

        # Accumulators
        x_e_batch, y_e_batch = [], []
        x_f_batch, y_f_batch = [], []

        for is_energy, x, y in stream_dataframe_rows(df, energy_key, sample_weights, n_elements):
            if is_energy:
                x_e_batch.append(x)
                y_e_batch.append(y)
            else:
                x_f_batch.append(x)
                y_f_batch.append(y)

            # Process when batch is full
            if len(x_e_batch) >= batch_size:
                X = np.stack(x_e_batch)
                Y = np.stack(y_e_batch)
                X, Y = freeze_columns(X, Y, self.mask, self.frozen_c, self.col_idx)
                gram_e += X.T @ X
                ord_e += X.T @ Y
                if e_variance:
                    e_variance.update(Y)
                x_e_batch.clear()
                y_e_batch.clear()

            if len(x_f_batch) >= batch_size:
                X = np.stack(x_f_batch)
                Y = np.stack(y_f_batch)
                X, Y = freeze_columns(X, Y, self.mask, self.frozen_c, self.col_idx)
                gram_f += X.T @ X
                ord_f += X.T @ Y
                if f_variance:
                    f_variance.update(Y)
                x_f_batch.clear()
                y_f_batch.clear()

        # Final partials
        if x_e_batch:
            X = np.stack(x_e_batch)
            Y = np.stack(y_e_batch)
            X, Y = freeze_columns(X, Y, self.mask, self.frozen_c, self.col_idx)
            gram_e += X.T @ X
            ord_e += X.T @ Y
            if e_variance:
                e_variance.update(Y)

        if x_f_batch:
            X = np.stack(x_f_batch)
            Y = np.stack(y_f_batch)
            X, Y = freeze_columns(X, Y, self.mask, self.frozen_c, self.col_idx)
            gram_f += X.T @ X
            ord_f += X.T @ Y
            if f_variance:
                f_variance.update(Y)

        return gram_e, gram_f, ord_e, ord_f
    
def test_chunk(x):
    import time
    print(f'starting process {os.getpid()}')
    time.sleep(5)
    return x
def stream_dataframe_rows(
        df: pd.DataFrame,
        energy_key: str = "energy",
        sample_weights: Dict = None,
        n_elements: int = 0
    ):
        """
        Yields one row at a time as (name, is_energy, x, y) with optional weight/normalization.
        """
        for (name, index), row in df.iterrows():
            y = row.iloc[0]
            x = row.iloc[1:].to_numpy()

            # size normalization (assumes n_*)
            if n_elements > 0 and index == energy_key:
                norm = np.sum(x[:n_elements])
                if norm != 0:
                    x = x / norm
                    y = y / norm

            w = sample_weights.get(name, 1.0) if sample_weights else 1.0
            x = x * w
            y = y * w

            is_energy = (index == energy_key)
            yield is_energy, x.astype(np.float64, copy=False), np.float64(y)
from joblib import Parallel, delayed
def process_chunk_serializable_wrapper(args):
    return process_chunk_serializable(*args)
def process_chunk_serializable(chunk_df, model_config, energy_key, batch_size, sample_weights, outlier_config):
    print("starting chunk process")
    x_e, y_e, x_f, y_f = dataframe_to_tuples(chunk_df, model_config["n_elements"], energy_key, sample_weights)
    del chunk_df
    x_e, y_e = freeze_columns(x_e, y_e, model_config["mask"], model_config["frozen_c"], model_config["col_idx"])
    x_f, y_f = freeze_columns(x_f, y_f, model_config["mask"], model_config["frozen_c"], model_config["col_idx"])

    gram_e, ord_e = batched_moore_penrose(x_e, y_e, batch_size)
    gram_f, ord_f = batched_moore_penrose(x_f, y_f, batch_size)
    del x_e, x_f
    
    ev, fv = VarianceRecorder(), VarianceRecorder()
    ev.update(y_e)
    fv.update(y_f)
    del y_e, y_f

    return (gram_e, gram_f, ord_e, ord_f, ev.mean, ev.std, ev.n, fv.mean, fv.std, fv.n)
def get_spline_taylor_expansion(r_target,
                                r,
                                coefficients,
                                knot_sequence,
                                min_curvature=0.0):
    nd3 = ndsplines.NDSpline([knot_sequence], coefficients, 3)
    y_trace = nd3(r_target, nus=0)
    d1_trace = nd3(r_target, nus=1)
    d2_trace = nd3(r_target, nus=2)
    if min_curvature is not None:
        d2_trace = max(d2_trace, min_curvature)
    dr = r - r_target
    y = y_trace + (d1_trace * dr) + (0.5 * d2_trace * dr ** 2)
    return y


def dataframe_to_tuples_with_names(df_features,
                        n_elements=None,
                        energy_key='energy',
                        sample_weights=None):
    """
    Extract energy/force inputs/outputs from DataFrame.

    Args:
        df_features (pd.DataFrame): dataframe with target vector (y) as the
            first column and feature vectors (x) as remaining columns.
        n_elements (int): number of leading columns to consider for size
            normalization.
        energy_key (str): key for energy samples, used to slice df_features
            into energies and forces for weight generation.
        sample_weights (dict):

    Returns:
        x (np.ndarray): features for machine learning.
        y (np.ndarray): target vector.
        w (np.ndarray): weight vector for machine learning.
    """
    names = df_features.index.get_level_values(0)
    y_index = df_features.index.get_level_values(-1)
    energy_mask = (y_index == energy_key)
    force_mask = np.logical_not(energy_mask)
    data = df_features.to_numpy(dtype=np.float64)
    y = data[:, 0]
    x = data[:, 1:]
    y_e = y[energy_mask]
    y_f = y[force_mask]
    #test_names = df_features.index[force_mask]
    level_0_names = df_features.index[force_mask].get_level_values(0)
    #level_1_values = test_names.get_level_values(1) #force components 
    if n_elements is not None:
        s = np.sum(x[energy_mask, :n_elements], axis=1)
        x_e = np.divide(x[energy_mask].T, s).T
        y_e = y_e / s
    else:
        x_e = x[energy_mask]
    x_f = x[force_mask]
    if sample_weights is not None:
        w = np.array([sample_weights.get(name, 1.0) for name in names])
        w_e = w[energy_mask]
        w_f = w[force_mask]
        x_e = np.multiply(x_e.T, w_e).T
        y_e = np.multiply(y_e, w_e)
        x_f = np.multiply(x_f.T, w_f).T
        y_f = np.multiply(y_f, w_f)
    return x_e, y_e, x_f, y_f, level_0_names

def dataframe_to_tuples(df_features,
                        n_elements=None,
                        energy_key='energy',
                        sample_weights=None):
    """
    Extract energy/force inputs/outputs from DataFrame.

    Args:
        df_features (pd.DataFrame): dataframe with target vector (y) as the
            first column and feature vectors (x) as remaining columns.
        n_elements (int): number of leading columns to consider for size
            normalization.
        energy_key (str): key for energy samples, used to slice df_features
            into energies and forces for weight generation.
        sample_weights (dict):

    Returns:
        x (np.ndarray): features for machine learning.
        y (np.ndarray): target vector.
        w (np.ndarray): weight vector for machine learning.
    """
    names = df_features.index.get_level_values(0)
    y_index = df_features.index.get_level_values(-1)
    energy_mask = (y_index == energy_key)
    force_mask = np.logical_not(energy_mask)
    data = df_features.to_numpy()
    y = data[:, 0]
    x = data[:, 1:]
    y_e = y[energy_mask]
    y_f = y[force_mask]
    n_elements = sum(1 for col in df_features.columns[1:] if col.startswith("n_"))
    if n_elements > 0:
        s = np.sum(x[energy_mask, :n_elements], axis=1)
        x_e = np.divide(x[energy_mask].T, s).T
        y_e = y_e / s
    else:
        x_e = x[energy_mask]
    x_f = x[force_mask]
    if sample_weights is not None:
        w = np.array([sample_weights.get(name, 1.0) for name in names])
        w_e = w[energy_mask]
        w_f = w[force_mask]
        x_e = np.multiply(x_e.T, w_e).T
        y_e = np.multiply(y_e, w_e)
        x_f = np.multiply(x_f.T, w_f).T
        y_f = np.multiply(y_f, w_f)
    return x_e, y_e, x_f, y_f


def moore_penrose_components(x, y):
    """
    Compute gram matrix (x^T x) and ordinate (x^T y).

    Args:
        x (np.ndarray): input matrix of shape (n_samples, n_features).
        y (np.ndarray): output vector of length n_samples.

    Returns:
        a: Gram matrix (X'X)
        b: ordinate (X'y)
    """
    a = np.dot(x.T, x)
    b = np.dot(x.T, y)
    return a, b


def batched_moore_penrose(x, y, batch_size=500):
    n_samples, n_features = x.shape

    if n_samples <= batch_size:
        return moore_penrose_components(x, y)

    gram = np.zeros((n_features, n_features), dtype=np.float64)
    ordinate = np.zeros(n_features, dtype=np.float64)
    print("moore penrose setup")
    print_mem()
    t0 = time.perf_counter()
    loop_count = 0
    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        t1 = time.perf_counter()
        x_batch = x[start:stop]
        y_batch = y[start:stop]

        g, o = moore_penrose_components(x_batch, y_batch)
        np.add(gram, g, out=gram)
        np.add(ordinate, o, out=ordinate)
        t2 = time.perf_counter()
        print(f"Batch {loop_count} | samples {start}:{stop} | time: {t2 - t1:.4f}s")
        del x_batch, y_batch, g, o
        loop_count += 1
        
        if (loop_count) % 4 == 0:
            gc.collect()
    t3 = time.perf_counter()
    print(f"Complete moore penrose {loop_count} | time: {t3 - t0:.4f}s")
    return gram, ordinate


def lu_factorization(a, b):
    """
    LU factorization for least-squares solution using np.linalg.solve().

    Args:
        a: coefficients (X) or Gram matrix (X'X)
        b: ordinate (X'y)
    """
    return np.linalg.solve(a, b)


def linear_least_squares(x, y):
    """
    Solves the linear least-squares problem Ax=y. Regularizer matrix
    should be concatenated to x and zero-values padded to y.

    Args:
        x (np.ndarray): input matrix of shape (n_samples, n_features).
        y (np.ndarray): output vector of length n_samples.

    Returns:
        solution (np.ndarray): coefficients.
    """
    a, b = moore_penrose_components(x, y)
    return lu_factorization(a, b)


def weighted_least_squares(x, y, weights=None, regularizer=None):
    """
    Solves the linear least-squares problem with optional Tikhonov regularizer
    matrix and optional weighting.
    TODO: Remove (deprecated)

    Args:
        x (np.ndarray): input matrix.
        y (np.ndarray): output vector.
        weights (np.ndarray): sample weights (optional).
        regularizer (np.ndarray): Tikhonov regularizer matrix.

    Returns:
        solution (np.ndarray): coefficients.
        predictions (list of np.ndarray): predictions.
    """
    x_fit, y_fit = apply_weights(x, y, weights)
    n_feats = len(x[0])
    if regularizer is not None:  # append regularizer
        # validate_regularizer(regularizer, n_feats)
        reg_zeros = np.zeros(len(regularizer), dtype=np.float64)
        x_fit = np.concatenate([x_fit, regularizer])
        y_fit = np.concatenate([y_fit, reg_zeros])
    solution = linear_least_squares(x_fit, y_fit)
    return solution


def get_freezing_mask(n_feats: int, col_idx: np.ndarray) -> np.ndarray:
    """
    Freezing mask is the set difference between the range of feature indices
    and the indices to be excluded (col_idx).

    Args:
        n_feats (int): number of features.
        col_idx (list): list of indices to be masked.

    Returns:
        mask (np.ndarray): set of non-frozen indices.
    """
    mask = np.setdiff1d(np.arange(n_feats), col_idx)
    return mask


def freeze_columns(x: np.ndarray,
                   y: np.ndarray,
                   mask: np.ndarray,
                   frozen_c: np.ndarray,
                   col_idx: np.ndarray,
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Freeze coefficients of the solution (e.g. forcing to zero) by
    simultaneously eliminating columns of the input and their assumed
    contribution to the output.

    Args:
        x (np.ndarray): input matrix.
        y (np.ndarray): output vector.
        mask (np.ndarray): set of non-frozen indices.
        frozen_c (np.ndarray): values of coefficients to be frozen.
        col_idx (np.ndarray): indices of coefficients to be frozen.

    Returns:
        x (np.ndarray): input matrix without frozen columns.
        y (np.ndarray): output vector, minus frozen contributions.
    """
    x_fixed = x[:, col_idx]
    x = x[:, mask]
    y = np.subtract(y, np.dot(x_fixed, frozen_c))
    return x, y


def freeze_regularizer(regularizer: np.ndarray,
                       mask: np.ndarray) -> np.ndarray:
    """Apply freezing mask to regularizer, eliminating masked columns."""
    regularizer = regularizer[:, mask]
    return regularizer


def revert_frozen_coefficients(solution: np.ndarray,
                               n_coeff: int,
                               mask: Collection[bool],
                               frozen_c: Collection[float],
                               frozen_idx: Collection[int],) -> np.ndarray:
    """
    Reverse freezing operations by arranging learned coefficients
    and frozen coefficients using the mask.

    Args:
        solution: learned solution, excluding frozen coefficients
        n_coeff: number of columns in full (unfrozen) solution
        mask: indices of remaining columns in x.
        frozen_idx: column indices of fixed coefficients.
        frozen_c: frozen coefficients.

    Returns:
        full_solution (np.ndarray)
    """
    full_solution = np.zeros(n_coeff, dtype=np.float64)
    np.put_along_axis(full_solution, mask, solution, 0)
    np.put_along_axis(full_solution, frozen_idx, frozen_c, 0)
    return full_solution


def apply_weighted_gram(gram_matrix: np.ndarray,
                        weight: float) -> np.ndarray:
    """Deprecated utility function for weighting gram matrix."""
    return gram_matrix * weight**2


def apply_weights(x, y, weights):
    """Deprecated utility function for weighting inputs/outputs."""
    if weights is not None:
        if len(weights) != len(x):
            raise ValueError(
                'Number of weights does not match number of samples.')
        if not np.all(weights >= 0):
            raise ValueError('Negative weights provided.')
        w = np.sqrt(weights)
        x_fit = np.multiply(x.T, w).T
        y_fit = np.multiply(y, w)
    else:
        x_fit = x
        y_fit = y
    return x_fit, y_fit


def validate_regularizer(regularizer: np.ndarray, n_feats: int):
    """
    Check for consistency between regularizer matrix and number of features.

    Args:
        regularizer (np.ndarray): regularizer matrix.
        n_feats (int): number of features.
    """
    n_row, n_col = regularizer.shape
    if n_col != n_feats:
        shape_comparison = "N x {0}. Provided: {1} x {2}".format(n_feats,
                                                                 n_row,
                                                                 n_col)
        raise ValueError(
            "Expected regularizer shape: " + shape_comparison)


def subset_prediction(df: pd.DataFrame,
                      model: WeightedLinearModel,
                      subset_keys: Collection = None,
                      **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, list]:
    """
    Extract inputs/outputs from a subset of a DataFrame and make predictions.

    Args:
        df (pd.DataFrame): DataFrame containing input features and target values.
        model (WeightedLinearModel): Fitted model to generate predictions.
        subset_keys (list or set, optional): Keys to filter the DataFrame.

    Returns:
        y_e (np.ndarray): Target values for energies.
        p_e (np.ndarray): Predicted values for energies.
        y_f (np.ndarray): Target values for forces.
        p_f (np.ndarray): Predicted values for forces.
        ids (np.ndarray): Sample identifiers.
    """
    if subset_keys is not None:
        idx = df.index.unique(level=0).intersection(subset_keys)
        if len(idx) == 0:
            # Return empty arrays if no matching keys are found
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), []
        df = df.loc[idx]
    else:
        print("SUBSET KEYS IS NONE")
        return np.array([]), np.array([]), np.array([]), np.array([]), [], []

    # Extract input features and target values
    x_e, y_e, x_f, y_f, force_labels = dataframe_to_tuples_with_names(df, **kwargs)

    # Generate predictions using the fitted model
    p_e = model.predict(x_e)
    p_f = model.predict(x_f)

    # Extract identifiers (e.g., sample IDs)
    ids = df.index.unique(level=0)

    return y_e, p_e, y_f, p_f, list(ids), list(force_labels)
def get_elements_in_feature_data(feature_data):
    columns = feature_data.columns
    elements = set()
    for col in columns:
        if col.startswith('n_'):
            element = col.split('_')[1]
            elements.add(element)
    return elements

def batched_prediction_multiple_files(
    model: WeightedLinearModel,
    filenames: List[str],
    table_names: Collection = None,
    subset_keys: Collection = None,
    drop_columns: List[str] = None,
    use_elements: List[str] = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function for optimization workflow. Read inputs/outputs
    from multiple HDF5 files and predict using fitted model.

    Args:
        filenames (list): list of paths to HDF5 files.
        model (WeightedLinearModel): fitted model.
        table_names (list): list of table names to query from HDF5 file.
        subset_keys (list): list of keys to query from DataFrame.
        drop_columns (list): list of columns to drop. Used when modifying
            the cutoffs of the feature vectors from HDF5 file. No internal
            checks are performed to see if dropping provided columns produces
            features of the intended cutoffs. Use with Caution.

    Returns:
        y_e (np.ndarray): target values for energies.
        p_e (np.ndarray): prediction values for energies.
        y_f (np.ndarray): target values for forces.
        p_f (np.ndarray): prediction values for forces.
    """
    
    def dataframe_batch_loader_multiple_files(filenames: List[str], table_names: Collection):
        """
        Iterator for reading DataFrames from multiple HDF5 files in batches.
        """
        for filename in filenames:
            if not os.path.isfile(filename):
                raise FileNotFoundError(f"{filename} not found.")
            
            tables_to_read = table_names or io.analyze_hdf_tables(filename)[2]
            for table_name in tables_to_read:
                df = pd.read_hdf(filename, table_name)
                yield df

    y_e, p_e, y_f, p_f, ids, force_ids = [], [], [], [], [], []

    # Load DataFrame batches and process predictions
    for df in dataframe_batch_loader_multiple_files(filenames, table_names):
        drop_columns_final = []
        if use_elements is not None:
            elements_in_data = get_elements_in_feature_data(df)
            exclude_elements = [el for el in elements_in_data if el not in use_elements]
            drop_columns_final = [
                col for col in df.columns
                if any(ex_el.lower() in col.lower() for ex_el in exclude_elements)
            ]
        if drop_columns:
            drop_columns_final = drop_columns_final.extend(drop_columns) if drop_columns_final else drop_columns
        if drop_columns_final:
            df.drop(columns=drop_columns_final, inplace=True)
        # Perform predictions using the fitted model
        results = subset_prediction(df, model, subset_keys=subset_keys, **kwargs)
        
        # Append energy and force predictions and targets
        y_e.extend(results[0])
        p_e.extend(results[1])
        y_f.extend(results[2])
        p_f.extend(results[3])
        ids.extend(results[4])
        force_ids.extend(results[5])

    return np.array(y_e), np.array(p_e), np.array(y_f), np.array(p_f),list(ids), list(force_ids)


def batched_prediction(model: WeightedLinearModel,
                       filename: str,
                       table_names: Collection = None,
                       subset_keys: Collection = None,
                       drop_columns: List[str] = None,
                       use_elements: List[str] = None,
                       **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, list]:
    """
    Convenience function for optimization workflow. Read inputs/outputs
    from HDF5 file and predict using fitted model.

    Returns:
        y_e (np.ndarray): target values for energies.
        p_e (np.ndarray): prediction values for energies.
        y_f (np.ndarray): target values for forces.
        p_f (np.ndarray): prediction values for forces.
        ids (np.ndarray): identifiers of the samples.
    """
    if table_names is None:
        _, _, table_names, _ = io.analyze_hdf_tables(filename)
    df_batches = io.dataframe_batch_loader(filename, table_names)
    y_e, p_e, y_f, p_f, ids,force_ids = [], [], [], [], [],[]

    for df in df_batches:
        drop_columns_final = None
        if use_elements is not None:
            elements_in_data = get_elements_in_feature_data(df)
            exclude_elements = [el for el in elements_in_data if el not in use_elements]

            drop_columns_final = [
                col for col in df.columns
                if any(ex_el.lower() in col.lower() for ex_el in exclude_elements)
            ]
        if drop_columns:
            drop_columns_final = drop_columns + drop_columns_final
        if drop_columns_final:
            df.drop(columns=drop_columns_final, inplace=True)

        results = subset_prediction(df, model, subset_keys=subset_keys, **kwargs)
        y_e.extend(results[0])
        p_e.extend(results[1])
        y_f.extend(results[2])
        p_f.extend(results[3])
        ids.extend(results[4])
        force_ids.extend(results[5])

    return np.array(y_e), np.array(p_e), np.array(y_f), np.array(p_f),list(ids), list(force_ids)



def rmse_metric(predicted: Collection,
                actual: Collection) -> float:
    """
    Root-mean-square error metric.

    Args:
        predicted (list): prediction values.
        actual (list): reference values.

    Returns:
        root-mean-square-error metric.
    """
    return np.sqrt(np.mean(np.subtract(predicted, actual) ** 2))


def mae_metric(predicted, actual):
    """
    Mean-absolute error metric.

    Args:
        predicted (list): prediction values.
        actual (list): reference values.

    Returns:
        mean-absolute error metric.
    """
    return np.mean(np.abs(np.subtract(predicted, actual)))


def arrange_coefficients(coefficients, bspline_config):
    """
    Arrange coefficients by degree of interaction.

    Args:
        coefficients (np.ndarray): Flattened vector of coefficients.
            Partitioned by provided bspline_config per degree.
        bspline_config (bspline.BSplineBasis)

    Returns:
        solutions (dict): fit coefficients per degree.
    """
    split_indices = np.cumsum(bspline_config.partition_sizes)[:-1]
    solutions_list = np.array_split(coefficients,
                                    split_indices)
    element_list = bspline_config.element_list
    solutions = {element: value[0] for element, value
                 in zip(element_list, solutions_list[:len(element_list)])}
    solutions_list = solutions_list[len(element_list):]
    j = 0
    for d in range(2, bspline_config.degree + 1):
        interactions_map = bspline_config.interactions_map[d]
        for interaction in interactions_map:
            solutions[interaction] = solutions_list[j]
            j += 1
    return solutions


def postprocess_coefficients_2b(coefficients,
                                core_hardness=2.0,
                                min_core=2.0,
                                min_slope=0.1,
                                rounding_factor=3,
                                smooth_cutoff=False,
                                in_place=False):
    """
    Postprocess 2B coefficients to enforce repulsive core.

    Args:
        coefficients (np.ndarray): vector of 2B coefficients.
        core_hardness (float): power base factor for hard-core correction.
        min_core (float): minimum energy barrier at the lower-bound (eV).
        min_slope (float): minimum core slope at peak (eV).
        rounding_factor (float): decimal for rounding in extrema search.
        smooth_cutoff (bool): whether to fix the last two coefficients to
            zero, forcing the second derivative to be zero at the upper bound.
        in_place (bool): whether to modify in-place or make a copy.

    Returns:
        coefficients (np.ndarray): new vector of coefficients.
    """
    if not in_place:  # apply corrections to a copy
        coefficients = np.array(coefficients)
    well_idx = find_pair_potential_well(coefficients, rounding_factor)
    if well_idx > 1:
        # search for maximum left of potential well, rounding to meV (default)
        peak_search = np.round(coefficients[:well_idx], rounding_factor)
        # bias towards well with imperceptible slope to deal with plateau
        peak_search += np.arange(len(peak_search)) * 10**(-2 * rounding_factor)
        gradient = np.gradient(peak_search)
        peak_idx = np.argmax(peak_search)
        if np.all(gradient[:peak_idx] >= 0):
            # correction for case where lower-bound is far below
            # observations and coefficients are nearly zero.
            for i in np.arange(peak_idx)[::-1]:
                value = np.abs(coefficients[i + 1]) * core_hardness
                value = max(value, min_slope)
                coefficients[i] = value
    if coefficients[0] < min_core:
        # fail-safe hard core by simply fixing the first coefficient
        coefficients[0] = min_core
    if smooth_cutoff:
        coefficients[-2:] = 0
    return coefficients


def find_pair_potential_well(coefficients, rounding_factor):
    """
    Identify coefficient index corresponding to possible potential well.
    Intermediate function for postprocess_coefficients_2b().

    Args:
        coefficients: vector of two-body coefficients.
        rounding_factor: decimal for rounding in extrema search.

    Returns:
        well_idx: approximate location of potential well in coefficients
    """
    peak_idx = np.argmax(coefficients)
    well_idx = np.argmin(coefficients)
    if well_idx < peak_idx:
        # if well is left of peak, either core may not be well-defined
        # or well may not be well-defined
        well_search = np.round(coefficients[:peak_idx], rounding_factor)
        if np.ptp(well_search) < 10 ** -(rounding_factor - 1):
            # no actual well
            well_idx = peak_idx + 1
    return well_idx


def calc_E_F_weights(n_e, n_f, std_e, std_f):
    """
    Calculates weights applied to energy and force components of the
    least-squares problem (excluding kappa, which is applied in
    self.combine_weighted_gram()).

    Args:
        n_e (int): number of energy samples.
        n_f (int): number of force samples.
        e_stddev (float): standard deviation of energy samples.
        f_stddev (float): standard deviation of force samples.

    Returns:
        energy_weight (float): weight applied to energy components.
        force_weight (float): weight applied to force components.
    """
    if n_e == 0 or std_e == 0:
        print("n_e or std_e are 0!")
        energy_weight = 0.0
    else:
        energy_weight = 1.0 / (np.sqrt(n_e) * std_e)

    # If we have no forces or zero std, set that weight to 0
    if n_f == 0 or std_f == 0:
        print("n_f or std_f are 0!")
        force_weight = 0.0
    else:
        force_weight = 1.0 / (np.sqrt(n_f) * std_f)

    return energy_weight, force_weight

    
def filter_residual_zscore(x, y, coeffs, std, threshold=4.0):
        if coeffs is None or std == 0:
            return x, y  # nothing to compare against

        residuals = y - np.dot(x, coeffs)
        z_scores = np.abs(residuals / (std + 1e-8))
        mask = z_scores < threshold
        return x[mask], y[mask]
def filter_mahalanobis(x, y=None, *, threshold=4.0):
    if len(x) < 2:
        return (x, y) if y is not None else x

    mean = np.mean(x, axis=0)
    cov = np.cov(x, rowvar=False)
    try:
        inv_cov = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-6)
    except np.linalg.LinAlgError:
        return (x, y) if y is not None else x

    diffs = x - mean
    dists = np.sqrt(np.sum(diffs @ inv_cov * diffs, axis=1))
    mask = dists < threshold
    return (x[mask], y[mask]) if y is not None else x[mask]
def filter_by_feature_norm(x, y, threshold=4.0, bins=20):
    norms = np.linalg.norm(x, axis=1)
    bins_idx = np.digitize(norms, np.linspace(norms.min(), norms.max(), bins))

    keep_mask = np.ones(len(y), dtype=bool)

    for b in np.unique(bins_idx):
        idx = (bins_idx == b)
        if np.sum(idx) < 5:
            continue
        y_bin = y[idx]
        mu, std = np.mean(y_bin), np.std(y_bin)
        z = np.abs((y_bin - mu) / (std + 1e-8))
        keep_mask[idx] = z < threshold

    return x[keep_mask], y[keep_mask]

def remove_bad_features(x, y):
    mask = np.isfinite(x).all(axis=1)
    return x[mask], y[mask]
from sklearn.neighbors import NearestNeighbors
import numpy as np

def filter_global_percentile(X, y, lower=0.1, upper=99.9):
    lo, hi = np.percentile(y, [lower, upper])
    mask = (y >= lo) & (y <= hi)
    return X[mask], y[mask]

def filter_global_iqr(X, y, k=1.5):
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    mask = (y >= lo) & (y <= hi)
    return X[mask], y[mask]

def filter_residual_zscore(X, y, coeffs, std=None, threshold=4.0):
    if coeffs is None:
        return X, y
    residual = y - np.dot(X, coeffs)
    std = std if std is not None else np.std(residual)
    z = np.abs(residual / (std + 1e-8))
    mask = z < threshold
    return X[mask], y[mask]

def filter_mahalanobis(X, y, threshold=4.0):
    mu = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    try:
        inv_cov = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-6)
    except np.linalg.LinAlgError:
        return X, y
    delta = X - mu
    dist = np.sqrt(np.sum(delta @ inv_cov * delta, axis=1))
    mask = dist < threshold
    return X[mask], y[mask]

def filter_feature_norm_vs_label(X, y, bins=30, threshold=4.0):
    norms = np.linalg.norm(X, axis=1)
    bin_idx = np.digitize(norms, np.linspace(norms.min(), norms.max(), bins))
    mask = np.ones(len(y), dtype=bool)
    for b in np.unique(bin_idx):
        idx = (bin_idx == b)
        if np.sum(idx) < 5:
            continue
        yb = y[idx]
        mu, std = np.mean(yb), np.std(yb)
        z = np.abs((yb - mu) / (std + 1e-8))
        mask[idx] = z < threshold
    return X[mask], y[mask]
def apply_global_filters(X, y, coeffs=None, kind="energy", config=None):
    if config is None:
        config = {}

    X, y = filter_global_percentile(X, y,
        lower=config.get("percentile_lo", 0.1),
        upper=config.get("percentile_hi", 99.9))

    X, y = filter_global_iqr(X, y,
        k=config.get("iqr_k", 1.5))

    X, y = filter_mahalanobis(X, y,
        threshold=config.get("mahalanobis_thresh", 5.0))

    X, y = filter_feature_norm_vs_label(X, y,
        bins=config.get("norm_bins", 30),
        threshold=config.get("norm_dev_thresh", 4.0))

    if coeffs is not None:
        X, y = filter_residual_zscore(X, y, coeffs,
            std=config.get("resid_std", None),
            threshold=config.get("resid_thresh", 5.0))

    return X, y
def restore_feature_columns(X, full_dim):
    if X.shape[1] == full_dim:
        return X
    X_full = np.zeros((X.shape[0], full_dim), dtype=X.dtype)
    X_full[:, :X.shape[1]] = X
    return X_full