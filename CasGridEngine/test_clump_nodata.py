'''
Created on Aug 22, 2017

@author: root
'''

import os, sys
import ogr, osr, gdal, gdalconst
import numpy as np

# import numpy.ma as ma

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(sys.argv[0], " infile outfile")
        sys.exit()

    if os.path.exists(sys.argv[2]):
        print("exists:", sys.argv[2])
        sys.exit()

    ds_in = gdal.Open(sys.argv[1])
    if ds_in is None:
        print("Failed open:", sys.argv[1])
        sys.exit()

    print(sys.argv[1], "-->", sys.argv[2])

    xsize = ds_in.RasterXSize
    ysize = ds_in.RasterYSize

    driver = gdal.GetDriverByName("GTiff")
    options = ["COMPRESS=DEFLATE"]

    ds_out = driver.Create(sys.argv[2], xsize, ysize, bands=1, eType=gdalconst.GDT_UInt16, options=options)

    proj = ds_in.GetProjectionRef()
    ds_out.SetProjection(proj)

    geom = ds_in.GetGeoTransform()
    ds_out.SetGeoTransform(geom)

    band_in = ds_in.GetRasterBand(1)
    band_out = ds_out.GetRasterBand(1)

    nodata_in = band_in.GetNoDataValue()
    gdal.TermProgress_nocb(0.0)

    for y in range(ysize):
        data = band_in.ReadAsArray(0, y, xsize, 1)
        cond = np.logical_and(data < 10000, data > 0, data != nodata_in)
        data = np.where(cond, data, 0)
        band_out.WriteArray(data, 0, y)

        gdal.TermProgress_nocb(y / (ysize * 1.0))

    gdal.TermProgress_nocb(1.0)

    band_out.SetNoDataValue(0)
    band_out.ComputeStatistics(False)

    ds_out.FlushCache()
