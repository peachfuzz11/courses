import base64
from io import BytesIO

from matplotlib import pyplot


class Figurehelper:

    def __init__(self, figure):
        self._figure = figure

    def to_png(self):
        buf = BytesIO()
        self._figure.savefig(buf, format="png")
        img = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
        pyplot.close(self._figure)
        return img
