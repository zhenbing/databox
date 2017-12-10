'''
Created on Oct 2, 2017

@author: root
'''
import os
import netCDF4


def test_r(filename, times=1000):
    netcdf = netCDF4.Dataset(filename, mode="r")
    print(netcdf)
    netcdf.close()


def test_w(filename, times=1000):
    if os.path.exists(filename):
        netcdf = netCDF4.Dataset(filename, mode="r+")
    else:
        netcdf = netCDF4.Dataset(filename, mode="w")

    for i in range(times):
        netcdf.grid_crs = "This is a test crs: %s" % i

    netcdf.close()


if __name__ == '__main__':
    filename = "/tmp/test.nc"
    test_w(filename)
    test_r(filename)
