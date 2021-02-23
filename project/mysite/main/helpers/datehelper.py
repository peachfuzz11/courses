import datetime

from django.utils import timezone


class Datehelper:

    def __init__(self):
        self._date = None

    def of(self, date):
        self._date = date
        return self

    def now(self):
        self._date = datetime.datetime.now()
        self.make_aware()
        return self

    def minus_days(self, days_to_subtract):
        self._date = self._date - datetime.timedelta(days=days_to_subtract)
        return self

    def get_value(self):
        return self._date

    def make_aware(self):
        self._date = timezone.make_aware(self._date, timezone.get_current_timezone())
        return self
