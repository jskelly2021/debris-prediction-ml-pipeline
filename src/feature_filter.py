import pandas as pd

from logger import Log


log = Log()


class FeatureFilter:
    def __init__(
        self,
        enabled=True,
        drop_constant=True,
        min_binary_positive_count=0,
        head_name="features",
    ):
        self.enabled = enabled
        self.drop_constant = drop_constant
        self.min_binary_positive_count = min_binary_positive_count
        self.head_name = head_name

        self.kept_columns_ = None
        self.dropped_constant_columns_ = []
        self.dropped_sparse_binary_columns_ = []

    def _print_fit_report(self):
        log.h2(f"{self.head_name} feature filtering")
        log.body(f"Filtering enabled         : {self.enabled}")
        log.body(f"Dropped constant columns  : {len(self.dropped_constant_columns_)}")
        if self.dropped_constant_columns_:
            log.body(f"  {self.dropped_constant_columns_}")

        log.body(f"Dropped sparse binary cols: {len(self.dropped_sparse_binary_columns_)}")
        if self.dropped_sparse_binary_columns_:
            log.body(f"  {self.dropped_sparse_binary_columns_}")

        log.body(f"Output feature count      : {len(self.kept_columns_)}")
        print()

    def _is_binary_column(self, series):
        non_null_unique = set(series.dropna().unique())
        return non_null_unique.issubset({0, 1})

    def fit(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureFilter expects a pandas DataFrame.")

        self.dropped_constant_columns_ = []
        self.dropped_sparse_binary_columns_ = []

        if not self.enabled:
            self.kept_columns_ = X.columns.tolist()
            self._print_fit_report()
            return self

        dropped_constant = set()
        if self.drop_constant:
            dropped_constant = {
                col
                for col in X.columns
                if X[col].nunique(dropna=False) <= 1
            }

        dropped_sparse_binary = set()
        if self.min_binary_positive_count > 0:
            for col in X.columns:
                if col in dropped_constant:
                    continue

                if not self._is_binary_column(X[col]):
                    continue

                positive_count = int((X[col] == 1).sum())
                if positive_count < self.min_binary_positive_count:
                    dropped_sparse_binary.add(col)

        self.dropped_constant_columns_ = sorted(dropped_constant)
        self.dropped_sparse_binary_columns_ = sorted(dropped_sparse_binary)
        dropped_columns = dropped_constant | dropped_sparse_binary
        self.kept_columns_ = [col for col in X.columns if col not in dropped_columns]

        self._print_fit_report()
        return self

    def transform(self, X):
        if self.kept_columns_ is None:
            raise ValueError("FeatureFilter must be fitted before transform.")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureFilter expects a pandas DataFrame.")

        return X.reindex(columns=self.kept_columns_, fill_value=0)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
