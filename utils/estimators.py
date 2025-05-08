import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import minimize
from scipy.linalg import fractional_matrix_power


def get_odds(p, smooth=0.001):
    return np.where(p==0, smooth, p / (1-p))


def odds_ratio(q, p, **kwargs):
    return np.where(p == 0, 1, get_odds(q, **kwargs) / get_odds(p, **kwargs))


def apply_odds_ratio(p, odds_ratio, **kwargs):
    odds = get_odds(p, **kwargs) * odds_ratio
    return odds / (1 + odds)
    

def get_transfer_matrix(start, end, heat=0):
    assert len(start) == (n := len(end))
    diff = end - start
    gain = diff.clip(min=0)
    gain_frac = gain/gain.sum()
    lose = -diff.clip(max=0)
    lose_frac = np.divide(lose, start, where=start!=0)
    keep_frac = 1 - lose_frac
    pre_heat = np.array([[keep_frac[i] - heat if i == j else (lose_frac[i] * gain_frac[j]) for j in range(n)] for i in range(n)])
    return pre_heat + (heat * start)


class TransitionMatrixEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, max_transfer=1.0):
        self.transition_matrix_ = None
        self.max_transfer = max_transfer

    def fit(self, X, y):

        def estimate_switching_matrix(initial_pop, final_pop):
            n_cats = initial_pop.shape[1]
    
            def residuals(flat_matrix):
                S = flat_matrix.reshape((n_cats, n_cats))
                predicted = initial_pop @ S
                return np.sum((predicted - final_pop).values ** 2)
        
            def row_sum_constraint(flat_matrix):
                S = flat_matrix.reshape((n_cats, n_cats))
                return np.sum(S, axis=1) - 1
        
            initial_guess = np.eye(n_cats).flatten()
            bounds = [(1-self.max_transfer, 1) if i == j else (0, self.max_transfer) for i in range(n_cats) for j in range(n_cats)]
            constraints = [{'type': 'eq', 'fun': row_sum_constraint}]
            result = minimize(residuals, initial_guess, bounds=bounds, constraints=constraints)
            return result.x.reshape((n_cats, n_cats))
        
        self.transition_matrix_ = estimate_switching_matrix(X, y)
        
        return self

    def predict(self, X, power=1):
        if self.transition_matrix_ is None:
            raise ValueError('The model has not been fitted yet.')
        transition_matrix = fractional_matrix_power(self.transition_matrix_, power)
        predictions = X @ transition_matrix
        row_sums = predictions.values.sum(axis=1, keepdims=True)
        normalized_predictions = predictions / row_sums
        return normalized_predictions


class BasicTransitionMatrixEstimator(TransitionMatrixEstimator):
    def __init__(self, heat=0):
        self.transition_matrix_ = None
        self.heat = heat
        
    def fit(self, X, y):
        self.transition_matrix_ = get_transfer_matrix(X.values.mean(axis=0), y.values.mean(axis=0), heat=self.heat)
        return self
        

class OddsMultiplierEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, smooth=0.001):
        self.odds_ratios_ = None
        self.smooth=smooth

    def fit(self, X, y):
        self.odds_ratios_ = np.median(odds_ratio(y, X, smooth=self.smooth), axis=0)
        return self

    def predict(self, X, power=1):
        if self.odds_ratios_ is None:
            raise ValueError('The model has not been fitted yet.')

        odds_ratios = self.odds_ratios_ ** power
        predictions = apply_odds_ratio(X, odds_ratios, smooth=self.smooth)
        row_sums = predictions.sum(axis=1, keepdims=True)
        normalized_predictions = predictions / row_sums

        return pd.DataFrame(normalized_predictions, X.index, X.columns)


class LinearTrendEstimator(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.velocity_ = None

    def fit(self, X, y):
        self.velocity_ = (y - X).mean(axis=0)
        return self

    def predict(self, X, power=1):
        if self.velocity_ is None:
            raise ValueError('The model has not been fitted yet.')

        velocity = self.velocity_ * power
        predictions = np.clip(X + velocity, a_min=0, a_max=1)
        row_sums = predictions.values.sum(axis=1, keepdims=True)
        normalized_predictions = predictions / row_sums

        return normalized_predictions


class ExponentialEstimator(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.factor_ = None

    def fit(self, X, y):
        self.factor_ = (y.sum(axis=0) / X.sum(axis=0)).fillna(1).replace({np.inf: 1})
        return self

    def predict(self, X, power=1):
        if self.factor_ is None:
            raise ValueError('The model has not been fitted yet.')

        factor = self.factor_ ** power
        predictions = np.clip(X * factor, a_min=0, a_max=1)
        row_sums = predictions.values.sum(axis=1, keepdims=True)
        normalized_predictions = predictions / row_sums

        return normalized_predictions


class IndividualLinearEstimator(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.velocity_array_ = None

    def fit(self, X, y):
        assert X.index.equals(y.index), "Indexes must be equal"
        assert X.columns.equals(y.columns), "Columns must be equal"
        self.velocity_array_ = y - X
        self.X_index_ = X.index
        return self

    def predict(self, X, power=1):
        if self.velocity_array_ is None:
            raise ValueError('The model has not been fitted yet.')
        assert X.index.equals(self.velocity_array_.index), "Index must match the training set index"

        velocity_array = self.velocity_array_ * power
        predictions = (X + velocity_array).clip(0, 1)
        normalized_predictions = predictions.apply(lambda x: x/x.sum(), axis='columns')
        return normalized_predictions


class IndividualExponentialEstimator(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.factor_array_ = None

    def fit(self, X, y):
        assert X.index.equals(y.index), "Indexes must be equal"
        assert X.columns.equals(y.columns), "Columns must be equal"
        self.factor_array_ = (y / X).fillna(1).replace({np.inf: 1})
        self.X_index_ = X.index
        return self

    def predict(self, X, power=1):
        if self.factor_array_ is None:
            raise ValueError('The model has not been fitted yet.')
        assert X.index.equals(self.factor_array_.index), "Index must match the training set index"

        factor_array = self.factor_array_ ** power
        predictions = (X * factor_array).clip(0, 1)
        normalized_predictions = predictions.apply(lambda x: x/x.sum(), axis='columns')
        return normalized_predictions


class IndividualOddsRatioEstimator(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.odds_ratio_array_ = None

    def fit(self, X, y):
        assert X.index.equals(y.index), "Indexes must be equal"
        assert X.columns.equals(y.columns), "Columns must be equal"
        self.odds_ratio_array_ = pd.DataFrame(
            odds_ratio(y.values, X.values),
            index=X.index,
            columns=X.columns,
        )
        return self

    def predict(self, X, power=1):
        if self.odds_ratio_array_ is None:
            raise ValueError('The model has not been fitted yet.')
        assert X.index.equals(self.odds_ratio_array_.index), "Index must match the training set index"

        odds_ratio_array = self.odds_ratio_array_ ** power
        return pd.DataFrame(
            apply_odds_ratio(X.values, odds_ratio_array),
            index=X.index,
            columns=X.columns,
        ).apply(lambda x: x/x.sum(), axis='columns')


class IndividualBasicTransitionMatrixEstimator(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.transition_matrices_ = None
        self.columns_ = None

    def fit(self, X, y):
        assert X.index.equals(y.index), "Indexes must be equal"
        assert X.columns.equals(y.columns), "Columns must be equal"
        self.columns_ = X.columns
        self.transition_matrices_ = pd.Series([get_transfer_matrix(start, end) for start, end in zip(X.values, y.values)],
                                               index=X.index,
                                               name='matrices')
        return self

    def predict(self, X, power=1):
        if self.transition_matrices_ is None:
            raise ValueError('The model has not been fitted yet.')
        assert X.index.equals(self.transition_matrices_.index), "Index must match the training set index"
        assert self.columns_.equals(X.columns), "Columns must match the training set columns"

        def frac_power(matrix):
            return fractional_matrix_power(matrix, power)
            
        transition_matrices = self.transition_matrices_.apply(frac_power)
        return pd.DataFrame(
            (pd.concat([X, transition_matrices], axis='columns')
             .apply(lambda row: row.iloc[:-1].values @ row.iloc[-1], axis='columns')
             .to_list()),
            index=X.index,
            columns=X.columns,
        )
        