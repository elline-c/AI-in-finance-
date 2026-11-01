"""Compatibility safety net for Node 1 (Requirement 1.2, 1.3).

The expected values below were captured from the pre-multi-layout Node 1 on the
statement made by ``generate_clean`` with its default arguments. They are a literal
on purpose: this test must pass now and after every later parser change.
"""
from __future__ import annotations

from datetime import date

from fraud_detect.nodes import ingestion
from fraud_detect.synthetic import generate_clean

EXPECTED_HEADER = {
    "account_holder": "Meridian Trading Pte Ltd",
    "uen": "201812345A",
    "account_number": "0123456789",
    "statement_period_start": date(2025, 3, 1),
    "statement_period_end": date(2025, 12, 1),
    "opening_balance": 42000.0,
    "closing_balance": 43395.71,
    "declared_currency": "SGD",
}

# (index, date, date_raw, description, amount, balance, page)
EXPECTED_TXNS = [
    (0, '2025-03-03', '03/03/2025', 'M1 Limited', -5675.18, 36324.82, 1),
    (1, '2025-03-06', '06/03/2025', 'Ninja Van SG', 6968.18, 43293.0, 1),
    (2, '2025-03-12', '12/03/2025', 'Ninja Van SG', 2284.62, 45577.62, 1),
    (3, '2025-03-13', '13/03/2025', 'Office Rental', -2911.72, 42665.9, 1),
    (4, '2025-03-14', '14/03/2025', 'Payroll Run', -2084.41, 40581.49, 1),
    (5, '2025-03-16', '16/03/2025', 'IRAS GST', -984.92, 39596.57, 1),
    (6, '2025-03-17', '17/03/2025', 'SP Services', 6021.14, 45617.71, 1),
    (7, '2025-03-21', '21/03/2025', 'Supplier Payment', -6792.57, 38825.14, 1),
    (8, '2025-03-22', '22/03/2025', 'Comfort Transport', -5416.8, 33408.34, 1),
    (9, '2025-03-27', '27/03/2025', 'DBS Merchant Payout', 2462.37, 35870.71, 1),
    (10, '2025-03-31', '31/03/2025', 'Comfort Transport', -6507.59, 29363.12, 1),
    (11, '2025-04-05', '05/04/2025', 'Supplier Payment', -970.0, 28393.12, 1),
    (12, '2025-04-11', '11/04/2025', 'IRAS GST', -1109.96, 27283.16, 1),
    (13, '2025-04-17', '17/04/2025', 'Comfort Transport', -3655.36, 23627.8, 1),
    (14, '2025-04-22', '22/04/2025', 'IRAS GST', -4791.05, 18836.75, 1),
    (15, '2025-04-25', '25/04/2025', 'Payroll Run', -6212.61, 12624.14, 1),
    (16, '2025-04-30', '30/04/2025', 'Ninja Van SG', 7752.95, 20377.09, 1),
    (17, '2025-05-02', '02/05/2025', 'Supplier Payment', -3008.73, 17368.36, 1),
    (18, '2025-05-04', '04/05/2025', 'Sales Settlement', 8003.22, 25371.58, 1),
    (19, '2025-05-10', '10/05/2025', 'Customer Payment INV', 4231.17, 29602.75, 1),
    (20, '2025-05-11', '11/05/2025', 'Comfort Transport', -3790.29, 25812.46, 1),
    (21, '2025-05-14', '14/05/2025', 'SP Services', 751.39, 26563.85, 1),
    (22, '2025-05-17', '17/05/2025', 'Office Rental', -6887.36, 19676.49, 1),
    (23, '2025-05-22', '22/05/2025', 'Customer Payment INV', 1947.97, 21624.46, 1),
    (24, '2025-05-24', '24/05/2025', 'Office Rental', -5462.51, 16161.95, 1),
    (25, '2025-05-29', '29/05/2025', 'Customer Payment INV', 4867.06, 21029.01, 1),
    (26, '2025-06-01', '01/06/2025', 'Customer Payment INV', 2788.88, 23817.89, 1),
    (27, '2025-06-06', '06/06/2025', 'Payroll Run', -338.25, 23479.64, 1),
    (28, '2025-06-12', '12/06/2025', 'Payroll Run', -1165.97, 22313.67, 1),
    (29, '2025-06-17', '17/06/2025', 'Ninja Van SG', 8572.61, 30886.28, 1),
    (30, '2025-06-20', '20/06/2025', 'Office Rental', -1638.92, 29247.36, 1),
    (31, '2025-06-24', '24/06/2025', 'Sheng Siong Distribution', 3532.71, 32780.07, 1),
    (32, '2025-06-25', '25/06/2025', 'Office Rental', -4477.38, 28302.69, 1),
    (33, '2025-06-28', '28/06/2025', 'DBS Merchant Payout', 737.79, 29040.48, 1),
    (34, '2025-07-04', '04/07/2025', 'SP Services', 5467.92, 34508.4, 1),
    (35, '2025-07-09', '09/07/2025', 'IRAS GST', -5770.13, 28738.27, 1),
    (36, '2025-07-10', '10/07/2025', 'M1 Limited', -4809.86, 23928.41, 1),
    (37, '2025-07-11', '11/07/2025', 'Sheng Siong Distribution', 6910.23, 30838.64, 1),
    (38, '2025-07-13', '13/07/2025', 'DBS Merchant Payout', 7276.95, 38115.59, 2),
    (39, '2025-07-16', '16/07/2025', 'Sheng Siong Distribution', 1923.5, 40039.09, 2),
    (40, '2025-07-21', '21/07/2025', 'Supplier Payment', -4699.6, 35339.49, 2),
    (41, '2025-07-27', '27/07/2025', 'Ninja Van SG', 6472.32, 41811.81, 2),
    (42, '2025-07-31', '31/07/2025', 'Sheng Siong Distribution', 3152.04, 44963.85, 2),
    (43, '2025-08-04', '04/08/2025', 'IRAS GST', -2031.42, 42932.43, 2),
    (44, '2025-08-10', '10/08/2025', 'Office Rental', -1700.91, 41231.52, 2),
    (45, '2025-08-14', '14/08/2025', 'M1 Limited', -439.48, 40792.04, 2),
    (46, '2025-08-18', '18/08/2025', 'DBS Merchant Payout', 6610.1, 47402.14, 2),
    (47, '2025-08-22', '22/08/2025', 'Office Rental', -4843.78, 42558.36, 2),
    (48, '2025-08-24', '24/08/2025', 'Supplier Payment', -4526.17, 38032.19, 2),
    (49, '2025-08-29', '29/08/2025', 'SP Services', 4953.11, 42985.3, 2),
    (50, '2025-09-02', '02/09/2025', 'Supplier Payment', -5682.68, 37302.62, 2),
    (51, '2025-09-08', '08/09/2025', 'DBS Merchant Payout', 6768.34, 44070.96, 2),
    (52, '2025-09-10', '10/09/2025', 'Payroll Run', -2352.79, 41718.17, 2),
    (53, '2025-09-11', '11/09/2025', 'Sales Settlement', 8483.95, 50202.12, 2),
    (54, '2025-09-13', '13/09/2025', 'SP Services', 2645.04, 52847.16, 2),
    (55, '2025-09-18', '18/09/2025', 'Payroll Run', -554.04, 52293.12, 2),
    (56, '2025-09-20', '20/09/2025', 'CPF Board', -4120.88, 48172.24, 2),
    (57, '2025-09-26', '26/09/2025', 'CPF Board', -550.74, 47621.5, 2),
    (58, '2025-09-29', '29/09/2025', 'Ninja Van SG', 5373.75, 52995.25, 2),
    (59, '2025-10-04', '04/10/2025', 'Sales Settlement', 1387.61, 54382.86, 2),
    (60, '2025-10-08', '08/10/2025', 'Sheng Siong Distribution', 4748.3, 59131.16, 2),
    (61, '2025-10-13', '13/10/2025', 'M1 Limited', -6327.84, 52803.32, 2),
    (62, '2025-10-15', '15/10/2025', 'Customer Payment INV', 3285.8, 56089.12, 2),
    (63, '2025-10-17', '17/10/2025', 'Sales Settlement', 2310.75, 58399.87, 2),
    (64, '2025-10-18', '18/10/2025', 'IRAS GST', -6545.66, 51854.21, 2),
    (65, '2025-10-24', '24/10/2025', 'Office Rental', -5445.27, 46408.94, 2),
    (66, '2025-10-26', '26/10/2025', 'DBS Merchant Payout', 843.4, 47252.34, 2),
    (67, '2025-10-28', '28/10/2025', 'Sheng Siong Distribution', 5074.77, 52327.11, 2),
    (68, '2025-10-31', '31/10/2025', 'IRAS GST', -5935.59, 46391.52, 2),
    (69, '2025-11-03', '03/11/2025', 'Customer Payment INV', 2975.33, 49366.85, 2),
    (70, '2025-11-09', '09/11/2025', 'Office Rental', -1206.82, 48160.03, 2),
    (71, '2025-11-10', '10/11/2025', 'Ninja Van SG', 3956.09, 52116.12, 2),
    (72, '2025-11-12', '12/11/2025', 'Office Rental', -2584.01, 49532.11, 2),
    (73, '2025-11-17', '17/11/2025', 'Office Rental', -2832.87, 46699.24, 2),
    (74, '2025-11-22', '22/11/2025', 'Sheng Siong Distribution', 1194.82, 47894.06, 2),
    (75, '2025-11-25', '25/11/2025', 'M1 Limited', -4081.79, 43812.27, 2),
    (76, '2025-11-29', '29/11/2025', 'M1 Limited', -1021.8, 42790.47, 2),
    (77, '2025-12-01', '01/12/2025', 'Sales Settlement', 605.24, 43395.71, 2),
]


def test_legacy_names_importable():
    from fraud_detect.nodes.ingestion import (  # noqa: F401
        _PERIOD_RE,
        _ROW_RE,
        _parse_date,
        _parse_money,
    )


def test_generator_statement_unchanged(tmp_path):
    ing = ingestion.ingest(generate_clean(str(tmp_path)).path)
    for name, expected in EXPECTED_HEADER.items():
        assert getattr(ing, name) == expected, name
    actual = [
        (t.index, t.date.isoformat() if t.date else None, t.date_raw,
         t.description, t.amount, t.balance, t.page)
        for t in ing.transactions
    ]
    assert actual == EXPECTED_TXNS
