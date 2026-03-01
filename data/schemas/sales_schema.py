import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema
import pandas as pd

sales_schema = DataFrameSchema(
    {
        "id": Column(str),
        "item_id": Column(str),
        "dept_id": Column(str),
        "cat_id": Column(str),
        "store_id": Column(str),
        "state_id": Column(str),
    },
    coerce=True,
)