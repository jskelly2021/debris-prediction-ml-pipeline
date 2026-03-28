
import pandas as pd

from imblearn.over_sampling import SMOTE
from logger import Log


log = Log()


def apply_smote_single_label(X_train, y_train, label_name, random_state=12, k_neighbors=5):
    X_train = X_train.copy()
    y_train = pd.Series(y_train).copy()

    log.h2(f"SMOTE REPORT FOR {label_name}")

    before_pos = int((y_train == 1).sum())
    before_neg = int((y_train == 0).sum())
    log.body(f"  Before SMOTE: X shape={X_train.shape}, pos={before_pos}, neg={before_neg}")

    if before_pos < 2:
        log.warn("Skipping SMOTE: not enough minority samples.")
        return X_train, y_train

    X_train = X_train.fillna(0)

    k = min(k_neighbors, before_pos - 1)
    if k < 1:
        log.warn("Skipping SMOTE: k_neighbors would be invalid.")
        return X_train, y_train

    smote = SMOTE(random_state=random_state, k_neighbors=k)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name=y_train.name)

    after_pos = int((y_res == 1).sum())
    after_neg = int((y_res == 0).sum())
    log.body(f"  After  SMOTE: X shape={X_res.shape}, pos={after_pos}, neg={after_neg}")
    log.body(f"  Added samples: {len(X_res) - len(X_train)}")
    print()

    return X_res, y_res
