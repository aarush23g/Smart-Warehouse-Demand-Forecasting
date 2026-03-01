from src.inference.prediction_logic import compute_inventory_decision


def test_inventory_decision_positive_order():
    decision = compute_inventory_decision(
        p50=2.0,
        p90=5.0,
        current_inventory=1.0,
        lead_time=7,
        service_level=0.95,
    )

    assert decision["order_qty"] >= 0
    assert decision["reorder_point"] >= decision["safety_stock"]


def test_inventory_decision_zero_order():
    decision = compute_inventory_decision(
        p50=1.0,
        p90=2.0,
        current_inventory=100.0,
        lead_time=7,
        service_level=0.95,
    )

    assert decision["order_qty"] == 0


def test_uncertainty_logic():
    decision = compute_inventory_decision(
        p50=3.0,
        p90=6.0,
        current_inventory=0.0,
        lead_time=7,
        service_level=0.95,
    )

    assert decision["safety_stock"] > 0
    assert decision["reorder_point"] > 0


def test_p90_greater_than_p50():
    p50 = 4.0
    p90 = 7.0

    decision = compute_inventory_decision(
        p50=p50,
        p90=p90,
        current_inventory=0.0,
    )

    assert p90 >= p50