import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from typing import Tuple


def read_csv(file: str, class_index: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(file)

    data = np.ascontiguousarray(df.drop([class_index], axis=1, inplace=False).values)
    classes = np.ascontiguousarray(df[class_index].values)

    data_num_columns = np.shape(data)[1]
    data_indexs_str = [
        i for i in range(data_num_columns) if isinstance(data[0, i], str)
    ]
    column_transformer_data = ColumnTransformer(
        [("encoder", OneHotEncoder(), data_indexs_str)], remainder="passthrough"
    )
    try:
        data = column_transformer_data.fit_transform(data).toarray()
    except AttributeError:
        data = column_transformer_data.fit_transform(data)

    label_encoder_classes = LabelEncoder()
    classes = label_encoder_classes.fit_transform(classes)

    imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
    imp_mean.fit_transform(data)

    return data, classes
