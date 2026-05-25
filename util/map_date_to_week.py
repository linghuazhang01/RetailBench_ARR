#!/usr/bin/env python3
"""
Utility: map a date to week index where 09/07/89 is week 0 start and 09/14/89 is week 1 start.
"""

from datetime import datetime, date
from typing import Optional


def calc_week_from_base(target: date | str) -> int:
    """
    Given a date (date object or string), return week index relative to 09/07/89.
    09/07/89 is the first day of week 0; 09/14/89 starts week 1.
    """
    base = datetime.strptime("09/07/89", "%m/%d/%y").date()
    if not isinstance(target, date):
        target = datetime.strptime(str(target), "%m/%d/%y").date()
    delta_days = (target - base).days
    return delta_days // 7


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python map_date_to_week.py <MM/DD/YY>")
        raise SystemExit(1)

    dt = datetime.strptime(sys.argv[1], "%m/%d/%y").date()
    print(calc_week_from_base(dt))
