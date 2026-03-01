import numpy as np


def compute_safety_stock(p50, p90, lead_time=7, service_level=0.95):
    """
    Compute safety stock using demand uncertainty and service level.
    """
    Z = {
        0.90: 1.28,
        0.95: 1.64,
        0.99: 2.33,
    }.get(service_level, 1.64)

    uncertainty = p90 - p50
    sigma = uncertainty

    safety_stock = Z * sigma * np.sqrt(lead_time)

    return safety_stock


def compute_reorder_point(p50, safety_stock, lead_time=7):
    """
    ROP = expected demand during lead time + safety stock
    """
    return p50 * lead_time + safety_stock


def compute_order_quantity(rop, current_inventory):
    """
    Order only if inventory below ROP.
    """
    return max(0, rop - current_inventory)


def compute_inventory_decision(
    p50,
    p90,
    current_inventory,
    lead_time=7,
    service_level=0.95,
):
    """
    Full inventory decision pipeline.
    """
    safety_stock = compute_safety_stock(
        p50, p90, lead_time=lead_time, service_level=service_level
    )

    rop = compute_reorder_point(p50, safety_stock, lead_time=lead_time)

    order_qty = compute_order_quantity(rop, current_inventory)

    return {
        "safety_stock": safety_stock,
        "reorder_point": rop,
        "order_qty": order_qty,
    }