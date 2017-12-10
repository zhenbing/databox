import os

import netCDF4

from .CacheManager import lru_cache


class netCDF4_wrapper(object):
    def __init__(self, filename):
        #         print("__init__", filename)
        self.filename = filename
        self.netcdf = netCDF4.Dataset(filename, mode="r")

    def __del__(self):
        self.netcdf.close()

    #         print("__del__")

    def __call__(self):
        return self.netcdf


@lru_cache(maxsize=256, timeout=120)
def get_filter_files(folder, ext):
    files = list(filter(lambda a: a.endswith(ext), os.listdir(folder)))
    return files


@lru_cache(maxsize=256, timeout=120)
def get_ncfile_handler(ncfile):
    return netCDF4_wrapper(ncfile)
