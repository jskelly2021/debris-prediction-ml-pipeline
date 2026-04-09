# xgb-training

This project trains a multi-label two-head XGBoost model from a YAML config file.

## Run

From the repository root:

```bash
python src/main.py config/bt_only.yaml
```

You can also run a different config, for example:

```bash
python src/main.py config/hex9_v4.yaml
```

## Optional flags

```bash
python src/main.py config/bt_only.yaml --save --plots --feature_importance --run-id my_run
```

- `--save` saves metrics output.
- `--plots` saves dashboard plots.
- `--feature_importance` saves feature-importance output.
- `--run-id` sets the run name used in saved output.

## Config

Config files live in `config/`. Each config points to:

- `data_path`: input dataset file
- `output_path`: where outputs are written
- label definitions and model settings

Example configs:

- `config/bt_only.yaml`
- `config/hex9_v4.yaml`

## Notes

- Run commands from the repository root.
- Make sure the dataset path in the config exists before running.
