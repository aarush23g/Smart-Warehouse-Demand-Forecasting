from pydantic import BaseModel


class ForecastRequest(BaseModel):
    item_id: str
    store_id: str
    lag_7: float
    lag_28: float
    rolling_mean_7: float
    rolling_std_28: float
    sell_price: float
    event_name_1: str
    event_type_1: str


class ForecastResponse(BaseModel):
    p50: float
    p90: float
    safety_stock: float
    reorder_point: float
    order_qty: float