import sys
import tempfile
import unittest
import math
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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

        with patch("two_head_pipeline.train_classifier") as mock_train_classifier, patch("two_head_pipeline.train_regressor") as mock_train_regressor:
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
            apply_smote=False,
            outlier_threshold=None,
            positive_only_regression=False,
            random_state=12,
        )
        single_splits = make_label_specific_splits(
            X=X,
            y_class=y_class[["Bin_CD"]],
            y_reg=y_reg[["VolCD_sum"]],
            label_specs=single_cfg.label_specs,
            apply_smote=False,
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


if __name__ == "__main__":
    unittest.main()
