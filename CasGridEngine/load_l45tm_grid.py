'''
Created on Aug 22, 2017

@author: root
'''

import datetime
import json
import math
import os, sys
import sqlite3

from netCDF4 import num2date, date2num
import netCDF4
from netcdftime import utime

import rasterio

from databox.geomtrans import GeomTrans
from databox.gridtools import get_grid_by_xy, adjust_bbox, crs_to_proj4, paste_ndarray
import numpy as np


class MetaDB(object):
    def __del__(self):
        self.db.close()

    def __init__(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.db = sqlite3.connect(os.path.join(path, "meta.db"))
        sql = '''
CREATE TABLE IF NOT EXISTS db_datasets (dataid varchar(32) not null primary key, bands text, ctime float, cyear int, cmonth int, cday int, minx float, miny float, maxx float, maxy float);
CREATE INDEX IF NOT EXISTS db_datasets_ctime on db_datasets ( ctime );    
CREATE INDEX IF NOT EXISTS db_datasets_cyear on db_datasets ( cyear );    
CREATE INDEX IF NOT EXISTS db_datasets_cmonth on db_datasets ( cmonth );    
CREATE INDEX IF NOT EXISTS db_datasets_cday on db_datasets ( cday );    
    '''
        sqls = list(filter(lambda a: len(a.strip()) > 0, sql.split(";")))
        for sql in sqls:
            self.db.execute(sql)

    def insert_dataid(self, dataid, bandid, ctime, minx, miny, maxx, maxy):
        sql = "select bands from db_datasets where dataid = ? limit 1"
        c = self.db.execute(sql, [dataid])
        datas = c.fetchall()
        if len(datas) == 0:
            bands = [bandid]
            sql = "insert into db_datasets (dataid, bands, ctime, cyear, cmonth, cday, minx, miny, maxx, maxy) values ( ?, ?, ?, ?, ?, ?, ?, ?, ?, ? )"
            self.db.execute(sql,
                            [dataid, json.dumps(bands, ensure_ascii=False), ctime.timestamp(), ctime.year, ctime.month,
                             ctime.day, minx, miny, maxx, maxy])
        else:
            bands = json.loads(datas[0][0])
            if bandid not in bands:
                bands.append(bandid)
                bands.sort()
                sql = "update db_datasets set bands = ? where dataid = ?"
                self.db.execute(sql, [json.dumps(bands, ensure_ascii=False), dataid])

    def commit(self):
        self.db.commit()


class GridCubeBuilder(object):

    def __init__(self, raster_file, grid_root, product, sensor, ctime, dataid, bandid, gsize):
        self.raster_file = raster_file
        self.grid_root = grid_root

        self.product = product.upper()
        self.sensor = sensor.upper()
        self.ctime = ctime
        self.dataid = dataid
        self.bandid = bandid.upper()
        self.gsize = gsize

        self.metadb = MetaDB(os.path.join(grid_root, sensor))

        print("process:", raster_file)

    def _get_ncfile(self, sensor, grid_y, grid_x, bandid):
        ncfile = os.path.join(self.grid_root, "%s/%s/%s/%s/%s/%s.nc" % (
        sensor, grid_y // 256, grid_y % 256, grid_x // 256, grid_x % 256, bandid))
        return ncfile

    def save_to_cube(self, grid_meta, grid_image, grid_crs, grid_bounds, grid_size, grid_res, grid_xy):
        ctime = grid_meta["ctime"]
        dtype = grid_meta["dtype"]
        dataid = grid_meta["dataid"]

        ncfile = self._get_ncfile(self.sensor, grid_xy[1], grid_xy[0], self.bandid)

        if os.path.exists(ncfile):
            ncdataset = netCDF4.Dataset(ncfile, mode="r+")
        else:
            ncdir = os.path.dirname(ncfile)
            if not os.path.exists(ncdir):
                os.makedirs(ncdir)

            ncdataset = netCDF4.Dataset(ncfile, mode="w")

            ncdataset.grid_crs = grid_crs
            ncdataset.grid_bounds = grid_bounds
            ncdataset.grid_res = grid_res
            ncdataset.grid_xy = grid_xy
            ncdataset.grid_size = grid_size

        #         x_min_0, y_min_0, x_max_0, y_max_0 = grid_bounds
        #         pixel_xsize, pixel_ysize = grid_res
        #         x_size , y_size = grid_size

        dims = ncdataset.dimensions.keys()

        def get_dimension(k, ndim=None):
            if k not in dims:
                return ncdataset.createDimension(k, ndim)
            else:
                return ncdataset.dimensions[k]

        get_dimension("dataid", None)  # 数据标识维度
        get_dimension("time", None)  # 时间维度
        get_dimension("year", None)  # 年维度
        get_dimension("month", None)  # 月维度
        get_dimension("day", None)  # 日维度
        get_dimension("hour", None)  # 时维度
        get_dimension("minute", None)  # 分维度
        get_dimension("x", grid_size[0])  # 经度维度
        get_dimension("y", grid_size[1])  # 纬度维度

        nc_vars = ncdataset.variables.keys()

        if "times" not in nc_vars:
            grid_times = ncdataset.createVariable("times", datatype="f8", dimensions=("time",))
            grid_times.units = "hours since 0001-01-01 00:00:00.0"
            grid_times.calendar = "gregorian"
        else:
            grid_times = ncdataset.variables["times"]

        cdftime = utime(grid_times.units)
        dtime = cdftime.date2num(ctime)
        if len(grid_times[grid_times == dtime]) > 0:
            print("--------data exists in cube, skipit--------", ctime)
            return

        if "dataids" not in nc_vars:
            grid_dataids = ncdataset.createVariable("dataids", datatype=str, dimensions=("dataid",))
        else:
            grid_dataids = ncdataset.variables["dataids"]

        if "years" not in nc_vars:
            grid_years = ncdataset.createVariable("years", datatype="u4", dimensions=("year",))
        else:
            grid_years = ncdataset.variables["years"]

        if "months" not in nc_vars:
            grid_months = ncdataset.createVariable("months", datatype="u4", dimensions=("month",))
        else:
            grid_months = ncdataset.variables["months"]

        if "days" not in nc_vars:
            grid_days = ncdataset.createVariable("days", datatype="u4", dimensions=("day",))
        else:
            grid_days = ncdataset.variables["days"]

        if "hours" not in nc_vars:
            grid_hours = ncdataset.createVariable("hours", datatype="u4", dimensions=("hour",))
        else:
            grid_hours = ncdataset.variables["hours"]

        if "minutes" not in nc_vars:
            grid_minutes = ncdataset.createVariable("minutes", datatype="u4", dimensions=("minute",))
        else:
            grid_minutes = ncdataset.variables["minutes"]

        if "values" not in nc_vars:
            fill_value = grid_image.get_fill_value()  # "u4"
            grid_vars = ncdataset.createVariable("values", datatype="u4", dimensions=("time", "y", "x",), zlib=True,
                                                 fill_value=fill_value)
            grid_vars.grid_dtype = dtype
        else:
            grid_vars = ncdataset.variables["values"]

        idx = len(grid_times)

        grid_times[idx] = date2num(ctime, units=grid_times.units, calendar=grid_times.calendar)
        grid_dataids[idx] = dataid
        grid_years[idx] = ctime.year
        grid_months[idx] = ctime.month
        grid_days[idx] = ctime.day
        grid_hours[idx] = ctime.hour
        grid_minutes[idx] = ctime.minute

        u4_grid_image = grid_image[0]

        '''
        只存储一个波段数据
        '''
        grid_vars[idx, :, :] = u4_grid_image.astype(np.uint32)
        ncdataset.close()

    def make_cube(self, ):

        with rasterio.open(self.raster_file, "r") as ds:
            proj = crs_to_proj4(ds.crs.wkt)

            utm2wgs = GeomTrans(proj, "EPSG:4326")
            wgs2utm = GeomTrans("EPSG:4326", proj)

            bbox_utm = ds.bounds

            extent_utm = [[bbox_utm.left, bbox_utm.top], [bbox_utm.left, bbox_utm.bottom],
                          [bbox_utm.right, bbox_utm.bottom],
                          [bbox_utm.right, bbox_utm.top]]  # get_extent(geom, cols, rows)

            #   the grid should be extended large by one pixel
            pixel_xsize, pixel_ysize = ds.res
            extent_wgs = utm2wgs.transform_points(extent_utm)

            ex = np.array(extent_wgs)

            x_max = ex[:, 0].max()
            x_min = ex[:, 0].min()
            y_max = ex[:, 1].max()
            y_min = ex[:, 1].min()

            self.metadb.insert_dataid(self.dataid, self.bandid, self.ctime, x_min, y_min, x_max, y_max)

            x_min = math.floor(x_min) - self.gsize
            x_max = math.ceil(x_max) + self.gsize
            y_min = math.floor(y_min) - self.gsize
            y_max = math.ceil(y_max) + self.gsize

            y = y_min
            while y < y_max:
                if y < -90 or y >= 90: continue
                x = x_min
                while x < x_max:
                    if x < -180 or x >= 180: continue
                    grid_xy = get_grid_by_xy(x, y, self.gsize)

                    corner_points = wgs2utm.transform_points([
                        [x, y],
                        [x, y + self.gsize],
                        [x + self.gsize, y + self.gsize],
                        [x + self.gsize, y]])

                    ex_0 = np.array(corner_points)

                    # 往外扩增1个像素
                    x_min_0 = ex_0[:, 0].min() - pixel_xsize
                    x_max_0 = ex_0[:, 0].max() + pixel_xsize

                    y_min_0 = ex_0[:, 1].min() - pixel_ysize
                    y_max_0 = ex_0[:, 1].max() + pixel_ysize

                    ref_bbox = [bbox_utm.left, bbox_utm.bottom, bbox_utm.right, bbox_utm.top]
                    mask_bounds, mask_size = adjust_bbox((x_min_0, y_min_0, x_max_0, y_max_0), ds.res,
                                                         ref_bbox=ref_bbox)

                    invert_y = ds.transform.e > 0
                    source_bounds = ds.bounds
                    if invert_y:
                        source_bounds = [source_bounds[0], source_bounds[3],
                                         source_bounds[2], source_bounds[1]]

                    if rasterio.coords.disjoint_bounds(source_bounds, mask_bounds):
                        x = x + self.gsize
                        print(grid_xy, "not overlap, skipit")
                        continue

                    if invert_y:
                        mask_bounds = [mask_bounds[0], mask_bounds[3],
                                       mask_bounds[2], mask_bounds[1]]

                    window = ds.window(*mask_bounds)
                    window = window.crop(ds.height, ds.width)  # 最新rasterio需要自己crop
                    #                     out_transform = ds.window_transform(window)
                    out_bounds = ds.window_bounds(window)

                    out_image = ds.read(window=window, masked=True)

                    v = out_image[~out_image.mask]
                    if len(v) == 0:  # all nodata
                        x = x + self.gsize
                        print(grid_xy, "all nodata, skipit")
                        continue

                    print(grid_xy, "saving data ...")

                    pout_image, pout_bounds, pout_size = paste_ndarray(None, mask_bounds, out_image, out_bounds, ds.res)

                    out_meta = ds.meta.copy()
                    out_meta.update({
                        "dataid": self.dataid,
                        "ctime": self.ctime,
                        "width": pout_size[0],
                        "height": pout_size[1],

                    })

                    self.save_to_cube(out_meta, pout_image, proj, pout_bounds, pout_size, ds.res, grid_xy)

                    x = x + self.gsize
                y = y + self.gsize

            self.metadb.commit()


def parse_landsat_ctime(dataid):
    year = int(dataid[9:13])
    days = int(dataid[13:16])
    stime = datetime.datetime(year=year, month=1, day=1)
    return datetime.timedelta(days=days - 1) + stime


def process(afile, root, gsize):
    items = afile.split("/")
    dataid = items[-2]
    bandid = items[-1].split(".")[0]

    ctime = parse_landsat_ctime(dataid)
    product = "LANDSAT"
    sensor = "L45TM"

    gcb = GridCubeBuilder(afile, root, product, sensor, ctime, dataid, bandid, gsize)
    gcb.make_cube()


def main():
    if len(sys.argv) != 2:
        print(sys.argv[0], "<bandfile>")
        return

    afile = sys.argv[1]
    if not os.path.exists(afile):
        print("File not exists:", afile)
        return

    from databox import QueryProvider
    qp = QueryProvider()

    typeMap = qp.typemaps["LANDSAT"]

    root = typeMap["ROOT"]
    gsize = typeMap["GSIZE"]

    if not os.path.exists(root):
        os.makedirs(root)

    process(afile, root, gsize)


#     testit(root, gsize)

if __name__ == '__main__':
    main()
