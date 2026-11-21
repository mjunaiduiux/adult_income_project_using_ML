import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RAW_NUM = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
RAW_CAT = ["workclass", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]
RAW_COLS = ["age", "workclass", "education_num", "marital_status", "occupation", "relationship",
            "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country"]

NUM_COLS = RAW_NUM + ["net_capital", "has_capital_gain", "has_capital_loss", "overtime", "is_us"]
CAT_COLS = RAW_CAT + ["age_bin"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in RAW_CAT:
        df[c] = df[c].astype(object)
        df[c] = df[c].where(df[c].notna(), np.nan)
    df["net_capital"] = df["capital_gain"] - df["capital_loss"]
    df["has_capital_gain"] = (df["capital_gain"] > 0).astype(int)
    df["has_capital_loss"] = (df["capital_loss"] > 0).astype(int)
    df["overtime"] = (df["hours_per_week"] > 40).astype(int)
    df["is_us"] = (df["native_country"] == "United-States").astype(int)
    df["age_bin"] = pd.cut(df["age"], bins=[0, 25, 35, 45, 55, 65, 150],
                           labels=["<=25", "26-35", "36-45", "46-55", "56-65", "65+"]).astype(str)
    return df


def build_preprocessor() -> Pipeline:
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    encode = ColumnTransformer([("num", num, NUM_COLS), ("cat", cat, CAT_COLS)])
    return Pipeline([("features", FunctionTransformer(add_features)), ("encode", encode)])
