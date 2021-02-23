import os

import numpy
import scipy
from scipy.io import netcdf

data_dir = os.path.dirname(os.path.abspath(__file__))


class CourseDataLoader:

    @staticmethod
    def load_netcdf(file):
        return netcdf.NetCDFFile(os.path.join(data_dir, file))

    @staticmethod
    def load_dat(file):
        return numpy.genfromtxt(os.path.join(data_dir, file),
                                skip_header=0,
                                skip_footer=0,
                                delimiter='')

    @staticmethod
    def load_mat(file):
        return scipy.io.loadmat(os.path.join(data_dir, file))

    @staticmethod
    def load_soi_txt() -> ([], [], [], []):
        path = os.path.join(data_dir, 'soi.txt')
        month = numpy.genfromtxt(path, skip_header=3, max_rows=1, dtype=str)[1:]
        year = numpy.genfromtxt(path, skip_header=4, max_rows=69 - 4)[:, 0].astype(int)
        anomaly = numpy.genfromtxt(path, skip_header=4, max_rows=69 - 4)[:, 1:]
        standardized = numpy.genfromtxt(path, skip_header=78, max_rows=143 - 78)[:, 1:]
        return month, year, anomaly, standardized
