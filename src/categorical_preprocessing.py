import pandas as pd

from logger import Log


log = Log()

SUPPORTED_ENCODINGS = {"onehot", "target", "none"}
MISSING_CATEGORY = "__MISSING__"


def _resolve_encoding_config(categorical_encoding):
    if categorical_encoding is None:
        return "onehot", {}

    if isinstance(categorical_encoding, str):
        return categorical_encoding, {}

    default = categorical_encoding.get("default", "onehot")
    overrides = categorical_encoding.get("overrides", {})
    return default, overrides


def resolve_column_encodings(categorical_cols, categorical_encoding):
    default_encoding, overrides = _resolve_encoding_config(categorical_encoding)

    if default_encoding not in SUPPORTED_ENCODINGS:
        raise ValueError(f"Unsupported default categorical encoding: {default_encoding}")

    encoding_map = {}
    for col in categorical_cols:
        encoding = overrides.get(col, default_encoding)
        if encoding not in SUPPORTED_ENCODINGS:
            raise ValueError(f"Unsupported categorical encoding '{encoding}' for column '{col}'.")
        encoding_map[col] = encoding

    unexpected = sorted(set(overrides) - set(categorical_cols))
    if unexpected:
        raise ValueError(f"categorical_encoding overrides reference non-categorical columns: {unexpected}")

    return encoding_map


class TargetEncoder:
    def __init__(self, smoothing=10.0):
        self.smoothing = smoothing
        self.global_mean_ = None
        self.mapping_ = {}
        self.preview_ = {}

    def fit(self, series, y):
        series = pd.Series(series, copy=True).fillna(MISSING_CATEGORY)
        y = pd.Series(y, index=series.index, copy=True)

        self.global_mean_ = float(y.mean())

        grouped = (
            pd.DataFrame({"category": series, "target": y})
            .groupby("category")["target"]
            .agg(["mean", "count"])
        )

        smooth = (
            grouped["mean"] * grouped["count"] + self.global_mean_ * self.smoothing
        ) / (grouped["count"] + self.smoothing)

        self.mapping_ = smooth.to_dict()
        self.preview_ = smooth.sort_values(ascending=False).head(3).to_dict()
        return self

    def transform(self, series):
        series = pd.Series(series, copy=True).fillna(MISSING_CATEGORY)
        return series.map(self.mapping_).fillna(self.global_mean_).astype(float)


class CategoricalPreprocessor:
    def __init__(
        self,
        categorical_cols,
        categorical_encoding=None,
        target_smoothing=10.0,
        head_name="features",
    ):
        self.categorical_cols = list(categorical_cols or [])
        self.categorical_encoding = categorical_encoding or {}
        self.target_smoothing = target_smoothing
        self.head_name = head_name

        self.column_encodings_ = {}
        self.available_categorical_cols_ = []
        self.passthrough_cols_ = []
        self.onehot_cols_ = []
        self.target_cols_ = []
        self.onehot_output_columns_ = []
        self.target_encoders_ = {}
        self.feature_names_ = []

    def _print_fit_report(self):
        log.h2(f"{self.head_name} categorical preprocessing")
        log.body(f"Passthrough columns : {len(self.passthrough_cols_)}")
        if self.passthrough_cols_:
            log.body(f"  {self.passthrough_cols_}")

        log.body(f"One-hot columns     : {len(self.onehot_cols_)}")
        if self.onehot_cols_:
            log.body(f"  {self.onehot_cols_}")
            log.body(f"One-hot features    : {len(self.onehot_output_columns_)}")

        log.body(f"Target-encoded cols : {len(self.target_cols_)}")
        if self.target_cols_:
            log.body(f"  {self.target_cols_}")
            for col in self.target_cols_:
                encoder = self.target_encoders_[col]
                log.body(
                    f"  {col} -> {col}_te | fallback={encoder.global_mean_:.4f} | "
                    f"preview={encoder.preview_}"
                )

        log.body(f"Output feature count: {len(self.feature_names_)}")
        print()

    def fit(self, X, y):
        X = X.copy()

        self.available_categorical_cols_ = [col for col in self.categorical_cols if col in X.columns]
        self.column_encodings_ = resolve_column_encodings(
            self.available_categorical_cols_,
            self.categorical_encoding,
        )

        self.onehot_cols_ = [col for col, encoding in self.column_encodings_.items() if encoding == "onehot"]
        self.target_cols_ = [col for col, encoding in self.column_encodings_.items() if encoding == "target"]
        passthrough_cats = [col for col, encoding in self.column_encodings_.items() if encoding == "none"]
        self.passthrough_cols_ = [col for col in X.columns if col not in self.available_categorical_cols_] + passthrough_cats

        if self.onehot_cols_:
            onehot_train = pd.get_dummies(
                X[self.onehot_cols_].copy(),
                columns=self.onehot_cols_,
                drop_first=False,
                dummy_na=True,
            )
            self.onehot_output_columns_ = onehot_train.columns.tolist()

        self.target_encoders_ = {}
        for col in self.target_cols_:
            encoder = TargetEncoder(smoothing=self.target_smoothing).fit(X[col], y)
            self.target_encoders_[col] = encoder

        transformed = self.transform(X)
        self.feature_names_ = transformed.columns.tolist()

        self._print_fit_report()

        return self

    def transform(self, X):
        X = X.copy()
        parts = []

        if self.passthrough_cols_:
            parts.append(X[self.passthrough_cols_].copy())

        if self.target_cols_:
            target_parts = []
            for col in self.target_cols_:
                encoder = self.target_encoders_[col]
                transformed = encoder.transform(X[col]).rename(f"{col}_te")

                unseen = int(
                    (~pd.Series(X[col], copy=False).fillna(MISSING_CATEGORY).isin(encoder.mapping_.keys())).sum()
                )
                if unseen:
                    log.info(f"{self.head_name}: '{col}' used global-mean fallback for {unseen} rows.")

                target_parts.append(transformed)

            parts.append(pd.concat(target_parts, axis=1))

        if self.onehot_cols_:
            onehot = pd.get_dummies(
                X[self.onehot_cols_].copy(),
                columns=self.onehot_cols_,
                drop_first=False,
                dummy_na=True,
            )
            unseen_columns = sorted(set(onehot.columns) - set(self.onehot_output_columns_))
            if unseen_columns:
                log.info(f"{self.head_name}: unseen one-hot categories encountered, dropped columns={unseen_columns[:5]}")

            parts.append(onehot.reindex(columns=self.onehot_output_columns_, fill_value=0))

        if not parts:
            return pd.DataFrame(index=X.index)

        return pd.concat(parts, axis=1)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
