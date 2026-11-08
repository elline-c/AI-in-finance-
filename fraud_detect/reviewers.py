"""Static reviewer / senior-officer rosters for the analyst console.

ITEM 5 + ITEM 6. Greenlink Digital Bank would, in production, pull the list of
people allowed to review and sign off on a case from its internal staff
directory / IAM. This prototype has no such directory, so the console presents a
small, fixed roster instead of a free-text name box: a reviewer picks their name
from a dropdown, and a junior reviewer who escalates a complex case picks the
senior officer to route it to from a second dropdown.

IMPORTANT: every name below is MOCK / demo data invented for the prototype. None
of these are real Greenlink Digital Bank staff; there is no real staff directory
behind this module. The rosters exist only so the dropdowns have plausible,
validatable values.

Display strings use a plain hyphen ("Name - Title"), never an em dash.
"""
from __future__ import annotations

from typing import NamedTuple


class Officer(NamedTuple):
    """One roster entry: a person's name and their title / position."""
    name: str
    title: str

    def display(self) -> str:
        """The exact string the dropdown shows and the server validates against.

        Format: "<name> - <title>" with a single plain hyphen separator.
        """
        return f"{self.name} - {self.title}"


# --------------------------------------------------------------------------- #
# Rosters (MOCK demo data - not real staff)
# --------------------------------------------------------------------------- #
# Front-line fraud / credit review officers who sign off on a case.
REVIEWERS: list[Officer] = [
    Officer("Jane Tan", "Credit Risk Analyst"),
    Officer("Marcus Lee", "Fraud Review Officer"),
    Officer("Priya Nair", "KYC Analyst"),
    Officer("Daniel Wong", "Transaction Monitoring Analyst"),
    Officer("Siti Rahman", "Credit Review Officer"),
]

# Senior officers a junior reviewer can escalate a complex case to.
SENIOR_OFFICERS: list[Officer] = [
    Officer("Rajesh Menon", "Senior Fraud Investigator"),
    Officer("Evelyn Koh", "Head of Credit Risk"),
    Officer("Arjun Pillai", "Senior Financial Crime Manager"),
    Officer("Grace Lim", "Head of Fraud Operations"),
]


# --------------------------------------------------------------------------- #
# Dropdown helpers
# --------------------------------------------------------------------------- #
def reviewer_choices() -> list[str]:
    """Display strings for the reviewer dropdown (e.g. 'Jane Tan - Credit Risk Analyst')."""
    return [o.display() for o in REVIEWERS]


def senior_choices() -> list[str]:
    """Display strings for the senior-officer escalation dropdown."""
    return [o.display() for o in SENIOR_OFFICERS]


def is_valid_reviewer(value: str) -> bool:
    """True when ``value`` exactly matches a reviewer roster display string.

    The server uses this to reject any submitted reviewer that is not on the
    roster, so a tampered form cannot sign off under an arbitrary name.
    """
    return value in set(reviewer_choices())


def is_valid_senior(value: str) -> bool:
    """True when ``value`` exactly matches a senior-officer roster display string."""
    return value in set(senior_choices())
