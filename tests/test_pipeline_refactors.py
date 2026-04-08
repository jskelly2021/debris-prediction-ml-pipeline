import sys
import tempfile
import unittest
import math
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

HAS_YAML = importlib.util.find_spec("yaml") is not None
HAS_IMBLEARN = importlib.util.find_spec("imblearn") is not None
HAS_XGBOOST = importlib.util.find_spec("xgboost") is not None
HAS_PIPELINE_DEPS = HAS_YAML and HAS_IMBLEARN and HAS_XGBOOST


class DummyClassifier:
    def predict_proba(self, X):
        return [[0.3, 0.7] for _ in range(len(X))]


class DummyRegressor:
    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, X):
        return self.predictions[:len(X)]


class DummyConfig:
    class_params_dist = {}
    reg_params_dist = {}
    class_default_params = {}
    reg_default_params = {}
    smote = False
    scale_pos_weight = False
    log_target_reg = False
    positive_only_regression = False
    categorical_cols = []
    categorical_encoding = {}
    target_encoding_smoothing = 10.0
    feature_filtering = {}


class ConfigTests(unittest.TestCase):
    def _write_config(self, payload):
        temp_dir = tempfile.TemporaryDirectory()
        path = Path(temp_dir.name) / "config.yaml"
        lines = []
        for key, value in payload.items():
            if isinstance(value, list):
                lines.append(f"{key}:")
                lines.extend([f"  - {item}" for item in value])
            elif isinstance(value, bool):
                lines.append(f"{key}: {'true' if value else 'false'}")
            else:
                lines.append(f"{key}: {value}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.addCleanup(temp_dir.cleanup)
        return path

    @unittest.skipUnless(HAS_YAML, "PyYAML is required to load config files.")
    def test_load_config_builds_label_specs_by_order(self):
        from config import load_config

        path = self._write_config({
            "data_path": "data.csv",
            "class_target_cols": ["Bin_CD", "Bin_VG", "Bin_Both"],
            "reg_target_cols": ["VolCD_sum", "VolVG_sum", "VolBoth_sum"],
            "label_names": ["VG", "CD", "BT"],
            "output_path": "outputs",
        })

        config = load_config(str(path))

        self.assertEqual(
            [(spec.label_name, spec.class_target_col, spec.reg_target_col) for spec in config.label_specs],
            [
                ("VG", "Bin_CD", "VolCD_sum"),
                ("CD", "Bin_VG", "VolVG_sum"),
                ("BT", "Bin_Both", "VolBoth_sum"),
            ],
        )

    @unittest.skipUnless(HAS_YAML, "PyYAML is required to load config files.")
    def test_load_config_preserves_feature_filtering_section(self):
        from config import load_config

        temp_dir = tempfile.TemporaryDirectory()
        path = Path(temp_dir.name) / "config.yaml"
        path.write_text(
            "\n".join([
                "data_path: data.csv",
                "output_path: outputs",
                "label_names:",
                "  - CD",
                "class_target_cols:",
                "  - Bin_CD",
                "reg_target_cols:",
                "  - VolCD_sum",
                "feature_filtering:",
                "  enabled: true",
                "  drop_constant: true",
                "  min_binary_positive_count: 20",
                "  max_dominant_value_fraction: 0.995",
            ]) + "\n",
            encoding="utf-8",
        )
        self.addCleanup(temp_dir.cleanup)

        config = load_config(str(path))

        self.assertEqual(
            config.feature_filtering,
            {
                "enabled": True,
                "drop_constant": True,
                "min_binary_positive_count": 20,
                "max_dominant_value_fraction": 0.995,
            },
        )

    @unittest.skipUnless(HAS_YAML, "PyYAML is required to load config files.")
    def test_load_config_fails_for_length_mismatch(self):
        from config import load_config

        path = self._write_config({
            "data_path": "data.csv",
            "class_target_cols": ["Bin_CD"],
            "reg_target_cols": ["VolCD_sum", "VolVG_sum"],
            "label_names": ["CD"],
            "output_path": "outputs",
        })

        with self.assertRaises(ValueError):
            load_config(str(path))


class PipelineTests(unittest.TestCase):
    @unittest.skipUnless(HAS_PIPELINE_DEPS, "pipeline tests require yaml, imblearn, and xgboost.")
    def test_tuned_threshold_is_used_for_predictions(self):
        from classifier import ClassifierTrainingResult
        from regressor import RegressorTrainingResult
        from two_head_pipeline import TwoHeadPipeline

        pipeline = TwoHeadPipeline(DummyConfig())
        splits = SimpleNamespace()

        with patch("two_head_pipeline.train_classifier") as mock_train_classifier, patch("two_head_pipeline.train_regressor") as mock_train_regressor, patch.object(pipeline, "_prepare_class_splits", return_value=splits), patch.object(pipeline, "_prepare_reg_splits", return_value=splits):
            mock_train_classifier.return_value = ClassifierTrainingResult(
                estimator=DummyClassifier(),
                training_time=0.0,
                best_threshold=0.8,
                best_f1=0.9,
            )
            mock_train_regressor.return_value = RegressorTrainingResult(
                estimator=DummyRegressor([2.0]),
                training_time=0.0,
            )

            pipeline.train(splits=splits, class_target_col="Bin_CD", reg_target_col="VolCD_sum")

        preds = pipeline.predict(pd.DataFrame({"x": [1]}))

        self.assertEqual(pipeline.threshold, 0.8)
        self.assertEqual(int(preds.class_pred[0]), 0)

    @unittest.skipUnless(HAS_PIPELINE_DEPS, "pipeline tests require yaml, imblearn, and xgboost.")
    def test_log_target_predictions_are_inverse_transformed(self):
        from two_head_pipeline import TwoHeadPipeline

        config = DummyConfig()
        config.log_target_reg = True

        pipeline = TwoHeadPipeline(config)
        pipeline.head1 = DummyClassifier()
        pipeline.head2 = DummyRegressor([0.0, math.log1p(3.0)])
        pipeline.is_fitted = True

        preds = pipeline.predict(pd.DataFrame({"x": [1, 2]}))

        self.assertAlmostEqual(float(preds.reg_pred[0]), 0.0, places=6)
        self.assertAlmostEqual(float(preds.reg_pred[1]), 3.0, places=6)

    @unittest.skipUnless(HAS_PIPELINE_DEPS, "pipeline tests require yaml, imblearn, and xgboost.")
    def test_pipeline_applies_feature_filter_to_train_val_test_and_predict(self):
        from classifier import ClassifierTrainingResult
        from regressor import RegressorTrainingResult
        from two_head_pipeline import TwoHeadPipeline

        config = DummyConfig()
        config.feature_filtering = {
            "enabled": True,
            "drop_constant": True,
            "min_binary_positive_count": 2,
        }

        pipeline = TwoHeadPipeline(config)
        splits = SimpleNamespace(
            X_train_class=pd.DataFrame({"x": [1, 2, 3]}),
            X_val_class=pd.DataFrame({"x": [4, 5]}),
            X_test_class=pd.DataFrame({"x": [6, 7]}),
            y_train_class=pd.Series([0, 1, 0]),
            y_val_class=pd.Series([1, 0]),
            y_test_class=pd.Series([0, 1]),
            X_train_reg=pd.DataFrame({"x": [1, 2, 3]}),
            X_val_reg=pd.DataFrame({"x": [4, 5]}),
            X_test_reg=pd.DataFrame({"x": [6, 7]}),
            y_train_reg=pd.Series([1.0, 2.0, 3.0]),
            y_val_reg=pd.Series([4.0, 5.0]),
            y_test_reg=pd.Series([6.0, 7.0]),
        )

        encoded_train = pd.DataFrame({
            "num": [10, 20, 30],
            "constant_onehot": [0, 0, 0],
            "rare_flag": [0, 0, 1],
            "common_flag": [1, 0, 1],
        })
        encoded_val = pd.DataFrame({
            "num": [40, 50],
            "constant_onehot": [0, 0],
            "rare_flag": [1, 0],
            "common_flag": [0, 1],
        })
        encoded_test = pd.DataFrame({
            "num": [60, 70],
            "constant_onehot": [0, 0],
            "rare_flag": [0, 0],
            "common_flag": [1, 1],
        })
        encoded_predict = pd.DataFrame({
            "num": [80],
            "constant_onehot": [0],
            "rare_flag": [1],
            "common_flag": [1],
        })

        class_transforms = [encoded_train, encoded_val, encoded_test, encoded_predict]
        reg_transforms = [encoded_train, encoded_val, encoded_test, encoded_predict]

        class RecordingClassifier(DummyClassifier):
            def __init__(self):
                self.predict_inputs = []

            def predict_proba(self, X):
                self.predict_inputs.append(X.copy())
                return super().predict_proba(X)

        class RecordingRegressor(DummyRegressor):
            def __init__(self, predictions):
                super().__init__(predictions)
                self.predict_inputs = []

            def predict(self, X):
                self.predict_inputs.append(X.copy())
                return super().predict(X)

        classifier_estimator = RecordingClassifier()
        regressor_estimator = RecordingRegressor([2.0])

        with patch("two_head_pipeline.CategoricalPreprocessor") as mock_preprocessor_cls, patch("two_head_pipeline.train_classifier") as mock_train_classifier, patch("two_head_pipeline.train_regressor") as mock_train_regressor:
            class_preprocessor = Mock()
            reg_preprocessor = Mock()
            mock_preprocessor_cls.side_effect = [class_preprocessor, reg_preprocessor]
            class_preprocessor.fit_transform.return_value = class_transforms[0]
            class_preprocessor.transform.side_effect = class_transforms[1:]
            reg_preprocessor.fit_transform.return_value = reg_transforms[0]
            reg_preprocessor.transform.side_effect = reg_transforms[1:]

            mock_train_classifier.return_value = ClassifierTrainingResult(
                estimator=classifier_estimator,
                training_time=0.0,
                best_threshold=0.5,
                best_f1=0.9,
            )
            mock_train_regressor.return_value = RegressorTrainingResult(
                estimator=regressor_estimator,
                training_time=0.0,
            )

            pipeline.train(splits=splits, class_target_col="Bin_CD", reg_target_col="VolCD_sum")
            pipeline.predict(pd.DataFrame({"x": [9]}))

        class_train_splits = mock_train_classifier.call_args.kwargs["splits"]
        reg_train_splits = mock_train_regressor.call_args.kwargs["splits"]

        expected_columns = ["num", "common_flag"]
        self.assertListEqual(class_train_splits.X_train_class.columns.tolist(), expected_columns)
        self.assertListEqual(class_train_splits.X_val_class.columns.tolist(), expected_columns)
        self.assertListEqual(class_train_splits.X_test_class.columns.tolist(), expected_columns)
        self.assertListEqual(reg_train_splits.X_train_reg.columns.tolist(), expected_columns)
        self.assertListEqual(reg_train_splits.X_val_reg.columns.tolist(), expected_columns)
        self.assertListEqual(reg_train_splits.X_test_reg.columns.tolist(), expected_columns)
        self.assertListEqual(classifier_estimator.predict_inputs[-1].columns.tolist(), expected_columns)
        self.assertListEqual(regressor_estimator.predict_inputs[-1].columns.tolist(), expected_columns)


class ConditionalRegressionTests(unittest.TestCase):
    @unittest.skipUnless(HAS_IMBLEARN, "imblearn is required because split.py imports resampling.")
    def test_positive_only_regression_splits_keep_only_positive_rows(self):
        from split import make_label_specific_splits

        X = pd.DataFrame({"f1": [1, 2, 3, 4, 5, 6]})
        y_class = pd.DataFrame({"Bin_CD": [0, 1, 0, 1, 0, 1]})
        y_reg = pd.DataFrame({"VolCD_sum": [0, 10, 0, 20, 0, 30]})
        label_specs = [SimpleNamespace(label_name="CD", class_target_col="Bin_CD", reg_target_col="VolCD_sum")]

        splits = make_label_specific_splits(
            X=X,
            y_class=y_class,
            y_reg=y_reg,
            label_specs=label_specs,
            outlier_threshold=None,
            positive_only_regression=True,
            holdout_size=0.5,
            random_state=1,
        )

        cd_split = splits["CD"]

        self.assertTrue((cd_split.y_train_reg > 0).all())
        self.assertTrue((cd_split.y_val_reg > 0).all())
        self.assertTrue((cd_split.y_test_reg > 0).all())

    def test_evaluation_uses_conditional_regression_predictions(self):
        from evaluation import evaluate_multilabel_model

        split = SimpleNamespace(
            X_test_class=pd.DataFrame({"f1": [1]}),
            X_test_reg=pd.DataFrame({"f1": [1]}),
            y_test_class=pd.Series([1]),
            y_test_reg=pd.Series([10.0]),
        )

        class DummyPipeline:
            def predict_df(self, X, prefix):
                return pd.DataFrame({
                    f"{prefix}_class_prob": [1.0],
                    f"{prefix}_class_pred": [1],
                    f"{prefix}_reg_pred": [10.0],
                    f"{prefix}_expected_volume_pred": [5.0],
                }, index=X.index)

        model = SimpleNamespace(
            is_fitted=True,
            train_config=SimpleNamespace(positive_only_regression=True),
            models={"CD": {"pipeline": DummyPipeline(), "class_col": "Bin_CD", "reg_col": "VolCD_sum"}},
        )

        metrics = evaluate_multilabel_model(model, {"CD": split})

        self.assertEqual(metrics["CD"].regression_display_name, "Conditional Volume")
        self.assertAlmostEqual(metrics["CD"].regression.rmse, 0.0, places=6)


class SplitConsistencyTests(unittest.TestCase):
    @unittest.skipUnless(HAS_IMBLEARN and HAS_YAML, "split tests require importable split/config modules.")
    def test_single_label_and_multi_label_runs_share_same_base_indices(self):
        from config import TrainConfig
        from split import make_label_specific_splits

        X = pd.DataFrame({"f1": range(30)})
        y_class = pd.DataFrame({
            "Bin_CD": [i % 2 for i in range(30)],
            "Bin_VG": [(i // 2) % 2 for i in range(30)],
            "Bin_Both": [(i // 3) % 2 for i in range(30)],
        })
        y_reg = pd.DataFrame({
            "VolCD_sum": [float(i) for i in range(30)],
            "VolVG_sum": [float(i + 1) for i in range(30)],
            "VolBoth_sum": [float(i + 2) for i in range(30)],
        })

        full_cfg = TrainConfig(
            data_path=Path("dummy.csv"),
            class_target_cols=["Bin_CD", "Bin_VG", "Bin_Both"],
            reg_target_cols=["VolCD_sum", "VolVG_sum", "VolBoth_sum"],
            label_names=["CD", "VG", "BT"],
        )
        single_cfg = TrainConfig(
            data_path=Path("dummy.csv"),
            class_target_cols=["Bin_CD"],
            reg_target_cols=["VolCD_sum"],
            label_names=["CD"],
        )

        full_splits = make_label_specific_splits(
            X=X,
            y_class=y_class,
            y_reg=y_reg,
            label_specs=full_cfg.label_specs,
            outlier_threshold=None,
            positive_only_regression=False,
            random_state=12,
        )
        single_splits = make_label_specific_splits(
            X=X,
            y_class=y_class[["Bin_CD"]],
            y_reg=y_reg[["VolCD_sum"]],
            label_specs=single_cfg.label_specs,
            outlier_threshold=None,
            positive_only_regression=False,
            random_state=12,
        )

        self.assertListEqual(
            list(full_splits["CD"].X_train_class.index),
            list(single_splits["CD"].X_train_class.index),
        )
        self.assertListEqual(
            list(full_splits["CD"].X_val_class.index),
            list(single_splits["CD"].X_val_class.index),
        )
        self.assertListEqual(
            list(full_splits["CD"].X_test_class.index),
            list(single_splits["CD"].X_test_class.index),
        )


class CategoricalEncodingTests(unittest.TestCase):
    def test_target_encoder_uses_train_statistics_only(self):
        from categorical_preprocessing import CategoricalPreprocessor

        X_train = pd.DataFrame({"cat": ["a", "a", "b"], "num": [1, 2, 3]})
        y_train = pd.Series([0.0, 1.0, 1.0])
        X_test = pd.DataFrame({"cat": ["a", "c"], "num": [4, 5]})

        encoder = CategoricalPreprocessor(
            categorical_cols=["cat"],
            categorical_encoding={"default": "target"},
            target_smoothing=1.0,
            head_name="test head",
        )

        train_out = encoder.fit_transform(X_train, y_train)
        test_out = encoder.transform(X_test)

        self.assertIn("cat_te", train_out.columns)
        self.assertAlmostEqual(float(test_out.loc[1, "cat_te"]), float(y_train.mean()), places=6)

    def test_onehot_encoding_produces_consistent_columns(self):
        from categorical_preprocessing import CategoricalPreprocessor

        X_train = pd.DataFrame({"cat": ["a", "b"], "num": [1, 2]})
        y_train = pd.Series([0, 1])
        X_test = pd.DataFrame({"cat": ["b", "c"], "num": [3, 4]})

        encoder = CategoricalPreprocessor(
            categorical_cols=["cat"],
            categorical_encoding={"default": "onehot"},
            head_name="test head",
        )

        train_out = encoder.fit_transform(X_train, y_train)
        test_out = encoder.transform(X_test)

        self.assertListEqual(train_out.columns.tolist(), test_out.columns.tolist())


class FeatureFilterTests(unittest.TestCase):
    def test_feature_filter_drops_constant_and_sparse_binary_columns(self):
        from feature_filter import FeatureFilter

        X = pd.DataFrame({
            "city_A": [1, 0, 0],
            "city_B": [0, 1, 0],
            "city_C": [0, 0, 0],
            "rare_flag": [0, 0, 1],
            "num": [5, 6, 7],
        })

        feature_filter = FeatureFilter(
            enabled=True,
            drop_constant=True,
            min_binary_positive_count=2,
        )

        transformed = feature_filter.fit_transform(X)

        self.assertListEqual(feature_filter.dropped_constant_columns_, ["city_C"])
        self.assertListEqual(feature_filter.dropped_sparse_binary_columns_, ["city_A", "city_B", "rare_flag"])
        self.assertListEqual(feature_filter.dropped_near_constant_columns_, [])
        self.assertListEqual(transformed.columns.tolist(), ["num"])

    def test_feature_filter_keeps_non_binary_numeric_columns(self):
        from feature_filter import FeatureFilter

        X = pd.DataFrame({
            "num": [1, 2, 3],
            "ratio": [0.0, 0.5, 1.0],
        })

        feature_filter = FeatureFilter(
            enabled=True,
            drop_constant=True,
            min_binary_positive_count=3,
        )

        transformed = feature_filter.fit_transform(X)

        self.assertListEqual(feature_filter.dropped_sparse_binary_columns_, [])
        self.assertListEqual(feature_filter.dropped_near_constant_columns_, [])
        self.assertListEqual(transformed.columns.tolist(), ["num", "ratio"])

    def test_feature_filter_treats_all_zero_and_all_one_as_constant_only(self):
        from feature_filter import FeatureFilter

        X = pd.DataFrame({
            "all_zero": [0, 0, 0],
            "all_one": [1, 1, 1],
            "keep_me": [0, 1, 0],
        })

        feature_filter = FeatureFilter(
            enabled=True,
            drop_constant=True,
            min_binary_positive_count=2,
        )

        transformed = feature_filter.fit_transform(X)

        self.assertListEqual(feature_filter.dropped_constant_columns_, ["all_one", "all_zero"])
        self.assertListEqual(feature_filter.dropped_sparse_binary_columns_, ["keep_me"])
        self.assertListEqual(feature_filter.dropped_near_constant_columns_, [])
        self.assertListEqual(transformed.columns.tolist(), [])

    def test_feature_filter_transform_reindexes_to_kept_columns(self):
        from feature_filter import FeatureFilter

        X_train = pd.DataFrame({
            "keep_a": [1, 0, 1],
            "drop_constant": [0, 0, 0],
            "keep_b": [2, 3, 4],
        })
        X_new = pd.DataFrame({
            "keep_b": [10],
            "extra": [99],
        })

        feature_filter = FeatureFilter(enabled=True, drop_constant=True, min_binary_positive_count=0)
        feature_filter.fit(X_train)

        transformed = feature_filter.transform(X_new)

        self.assertListEqual(transformed.columns.tolist(), ["keep_a", "keep_b"])
        self.assertEqual(int(transformed.loc[0, "keep_a"]), 0)
        self.assertEqual(int(transformed.loc[0, "keep_b"]), 10)

    def test_feature_filter_disabled_is_no_op(self):
        from feature_filter import FeatureFilter

        X = pd.DataFrame({
            "constant": [1, 1],
            "binary": [0, 1],
        })

        feature_filter = FeatureFilter(
            enabled=False,
            drop_constant=True,
            min_binary_positive_count=10,
            max_dominant_value_fraction=0.995,
        )

        transformed = feature_filter.fit_transform(X)

        self.assertListEqual(feature_filter.kept_columns_, ["constant", "binary"])
        self.assertListEqual(feature_filter.dropped_constant_columns_, [])
        self.assertListEqual(feature_filter.dropped_sparse_binary_columns_, [])
        self.assertListEqual(feature_filter.dropped_near_constant_columns_, [])
        self.assertListEqual(transformed.columns.tolist(), ["constant", "binary"])

    def test_feature_filter_drops_non_binary_near_constant_columns(self):
        from feature_filter import FeatureFilter

        X = pd.DataFrame({
            "dominant_num": [5, 5, 5, 5, 7],
            "binary_flag": [0, 0, 0, 1, 1],
            "varying_num": [1, 2, 3, 4, 5],
        })

        feature_filter = FeatureFilter(
            enabled=True,
            drop_constant=True,
            min_binary_positive_count=2,
            max_dominant_value_fraction=0.8,
        )

        transformed = feature_filter.fit_transform(X)

        self.assertListEqual(feature_filter.dropped_constant_columns_, [])
        self.assertListEqual(feature_filter.dropped_sparse_binary_columns_, [])
        self.assertListEqual(feature_filter.dropped_near_constant_columns_, ["dominant_num"])
        self.assertListEqual(transformed.columns.tolist(), ["binary_flag", "varying_num"])

    def test_feature_filter_near_constant_rule_skips_binary_columns(self):
        from feature_filter import FeatureFilter

        X = pd.DataFrame({
            "binary_flag": [0, 0, 0, 0, 1],
            "dominant_num": [9, 9, 9, 9, 8],
        })

        feature_filter = FeatureFilter(
            enabled=True,
            drop_constant=True,
            min_binary_positive_count=0,
            max_dominant_value_fraction=0.8,
        )

        transformed = feature_filter.fit_transform(X)

        self.assertListEqual(feature_filter.dropped_sparse_binary_columns_, [])
        self.assertListEqual(feature_filter.dropped_near_constant_columns_, ["dominant_num"])
        self.assertListEqual(transformed.columns.tolist(), ["binary_flag"])

    def test_feature_filter_near_constant_rule_is_disabled_when_unset(self):
        from feature_filter import FeatureFilter

        X = pd.DataFrame({
            "dominant_num": [5, 5, 5, 5, 7],
            "varying_num": [1, 2, 3, 4, 5],
        })

        feature_filter = FeatureFilter(
            enabled=True,
            drop_constant=True,
            min_binary_positive_count=0,
        )

        transformed = feature_filter.fit_transform(X)

        self.assertListEqual(feature_filter.dropped_near_constant_columns_, [])
        self.assertListEqual(transformed.columns.tolist(), ["dominant_num", "varying_num"])


if __name__ == "__main__":
    unittest.main()
