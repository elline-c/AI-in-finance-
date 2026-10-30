"""Unit tests for the Money_Reader (Requirements 3.1-3.7).

Test-first: ``fraud_detect.nodes.ingest_parse.money`` is implemented in task 3.2,
so this module fails with ImportError until then. All values are fictional.
"""
import pytest

from fraud_detect.nodes.ingest_parse.money import Money, parse_money


@pytest.mark.parametrize(
    "raw, value",
    [
        ("S$5,365.00", 5365.00),   # 3.1 S$ prefix
        ("SGD 28,249.88", 28249.88),  # 3.1 SGD prefix
        ("$1,000.50", 1000.50),    # 3.1 $ prefix
        ("1,234.56", 1234.56),     # 3.1 no prefix
        ("42", 42.0),
    ],
)
def test_prefixes_and_plain_numbers_have_no_sign(raw, value):
    m = parse_money(raw)
    assert isinstance(m, Money)
    assert m.value == pytest.approx(value)
    assert m.explicit_sign is None
    assert m.suffix is None


def test_minus_before_prefix():
    m = parse_money("-SGD 13.47")
    assert m is not None
    assert m.value == pytest.approx(-13.47)
    assert m.explicit_sign == -1
    assert m.suffix is None


def test_plus_before_prefix():
    m = parse_money("+SGD 11.08")
    assert m is not None
    assert m.value == pytest.approx(11.08)
    assert m.explicit_sign == 1
    assert m.suffix is None


def test_parentheses_are_negative():
    m = parse_money("(1,200.00)")
    assert m is not None
    assert m.value == pytest.approx(-1200.00)
    assert m.explicit_sign == -1


def test_cr_suffix_keeps_sign_and_records_suffix():
    m = parse_money("3,900.18 CR")
    assert m is not None
    assert m.value == pytest.approx(3900.18)
    assert m.suffix == "CR"
    assert m.explicit_sign is None


def test_cr_suffix_does_not_flip_an_explicit_negative():
    m = parse_money("-3,900.18 CR")
    assert m is not None
    assert m.value == pytest.approx(-3900.18)
    assert m.suffix == "CR"


def test_dr_suffix_makes_number_negative():
    m = parse_money("3,900.18 DR")
    assert m is not None
    assert m.value == pytest.approx(-3900.18)
    assert m.suffix == "DR"


@pytest.mark.parametrize(
    "raw", ["", "   ", "abc", "SGD", "N/A", "12.3.4", "BALANCE BROUGHT FORWARD", "--5.00"]
)
def test_invalid_text_returns_none(raw):
    assert parse_money(raw) is None


@pytest.mark.parametrize(
    "raw, value",
    [("10.126", 10.13), ("10.124", 10.12), ("-2.999", -3.0)],
)
def test_values_rounded_to_two_decimals(raw, value):
    m = parse_money(raw)
    assert m is not None
    assert m.value == pytest.approx(value, abs=1e-9)
    assert round(m.value, 2) == m.value
