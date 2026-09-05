"""Regex components for inferring `strptime` formats from string data."""

from __future__ import annotations

DATE_RE = r"(?P<date>\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\d{8})"
SEP_RE = r"(?P<sep>\s|T)"
TIME_RE = r"(?P<time>\d{2}:\d{2}(?::\d{2})?|\d{6}?)"
HMS_RE = r"^(?P<hms>\d{2}:\d{2}:\d{2})$"
HM_RE = r"^(?P<hm>\d{2}:\d{2})$"
HMS_RE_NO_SEP = r"^(?P<hms_no_sep>\d{6})$"
TZ_RE = r"(?P<tz>Z|[+-]\d{2}:?\d{2})"  # 'Z', '+02:00', '+0200'
FULL_RE = rf"{DATE_RE}{SEP_RE}?{TIME_RE}?{TZ_RE}?$"

YMD_RE = r"^(?P<year>(?:[12][0-9])?[0-9]{2})(?P<sep1>[-/.])(?P<month>0[1-9]|1[0-2])(?P<sep2>[-/.])(?P<day>0[1-9]|[12][0-9]|3[01])$"
DMY_RE = r"^(?P<day>0[1-9]|[12][0-9]|3[01])(?P<sep1>[-/.])(?P<month>0[1-9]|1[0-2])(?P<sep2>[-/.])(?P<year>(?:[12][0-9])?[0-9]{2})$"
MDY_RE = r"^(?P<month>0[1-9]|1[0-2])(?P<sep1>[-/.])(?P<day>0[1-9]|[12][0-9]|3[01])(?P<sep2>[-/.])(?P<year>(?:[12][0-9])?[0-9]{2})$"
YMD_RE_NO_SEP = r"^(?P<year>(?:[12][0-9])?[0-9]{2})(?P<month>0[1-9]|1[0-2])(?P<day>0[1-9]|[12][0-9]|3[01])$"

DATE_FORMATS = (
    (YMD_RE_NO_SEP, "%Y%m%d"),
    (YMD_RE, "%Y-%m-%d"),
    (DMY_RE, "%d-%m-%Y"),
    (MDY_RE, "%m-%d-%Y"),
)
TIME_FORMATS = ((HMS_RE, "%H:%M:%S"), (HM_RE, "%H:%M"), (HMS_RE_NO_SEP, "%H%M%S"))
