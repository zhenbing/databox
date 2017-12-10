import json
import os

from netcdftime import utime
import ogr
import rasterio
from rasterio.features import geometry_mask

from databox.geomtrans import GeomTrans
from databox.netcdf4 import get_ncfile_handler, get_filter_files
from databox.timeslice import TimeSlice, DATETIME_FMT
import numpy as np
import numpy.ma as ma

from .CacheManager import lru_cache
from .gridtools import get_grid_by_xy, map_bbox_win, adjust_bbox, crs_to_proj4
from .gridtools import get_grids_by_bbox, get_grid_bbox, bbox_polygon

try:
    import cPickle as pickle
except:
    import pickle

EPSG_4326 = "EPSG:4326"

GEOM_MAX_SIZE = 8 * 1024 * 1024


class EDatabox(Exception):
    def __init__(self, code, reason):
        self.code = code
        self.reason = reason


class EInvalidBBox(EDatabox):
    def __init__(self, *args):
        EDatabox.__init__(self, -1, "Invalid BBox")


class EInvalidGeom(EDatabox):
    def __init__(self, *args):
        EDatabox.__init__(self, -2, "Invalid Geometry")


class EGeomTooLarge(EDatabox):
    def __init__(self, *args):
        EDatabox.__init__(self, -3, "Geometry too large")


class ETimeSlice(EDatabox):
    def __init__(self, *args):
        EDatabox.__init__(self, -4, " ".join(args))


def _ndobject_to_str(o):
    if isinstance(o, str):
        return o
    f = getattr(o, "tolist", None)
    if f: return f()
    return str(o)


class DataBoxQuery(object):
    def __init__(self, root, gsize):
        self.root = root
        self.gsize = gsize

    def _get_ncfile(self, sensor, grid_y, grid_x, bandid):
        ncfile = os.path.join(self.root, "%s/%s/%s/%s/%s/%s.nc" % (
        sensor, grid_y // 256, grid_y % 256, grid_x // 256, grid_x % 256, bandid))
        return ncfile

    def _get_ncfile_path(self, sensor, grid_y, grid_x):
        ncfile = os.path.join(self.root,
                              "%s/%s/%s/%s/%s" % (sensor, grid_y // 256, grid_y % 256, grid_x // 256, grid_x % 256))
        return ncfile

    @lru_cache(maxsize=256, timeout=300, args_base=1)
    def _info_by_grid_xy(self, sensor, grid_x, grid_y, times=None):
        g_sensor = sensor.upper()

        ncfile_path = self._get_ncfile_path(g_sensor, grid_y, grid_x)
        if not os.path.exists(ncfile_path):
            return {}

        ncfiles = get_filter_files(ncfile_path, ".nc")
        if len(ncfiles) == 0:
            return {}

        res0 = {}
        #         res = { "bands" :  list(map(lambda a: a[:-3], ncfiles)) }
        for ncfile in ncfiles:
            bandid = ncfile[:-3]
            first_nc = os.path.join(ncfile_path, ncfile)

            res = {}
            ncdataset_wrapper = get_ncfile_handler(first_nc, )
            ncdataset = ncdataset_wrapper()

            grid_crs = ncdataset.grid_crs
            grid_bounds = ncdataset.grid_bounds
            grid_res = ncdataset.grid_res
            grid_size = ncdataset.grid_size

            res["crs"] = crs_to_proj4(grid_crs)
            res["bbox"] = list(map(lambda a: float(a), grid_bounds.tolist()))
            res["res"] = list(map(lambda a: float(a), grid_res.tolist()))
            res["size"] = list(map(lambda a: float(a), grid_size.tolist()))

            grid_dataids = ncdataset.variables["dataids"]
            grid_times = ncdataset.variables["times"]
            cdftime = utime(grid_times.units)

            t_slices = []
            if times is not None:
                if isinstance(times, TimeSlice) == False:
                    times = TimeSlice(times)
                try:
                    t_slices = times.get_slices(ncdataset, cdftime)
                except ValueError as e:
                    raise ETimeSlice(str(e))

            if len(t_slices) == 0:
                grid_datas_t = cdftime.num2date(grid_times[:])
                grid_dataids_t = grid_dataids[:]
            else:
                if len(t_slices[t_slices == True]) == 0:
                    grid_datas_t = np.ndarray(shape=(0))
                    grid_dataids_t = np.ndarray(shape=(0))
                else:
                    grid_datas_t = cdftime.num2date(grid_times[t_slices])
                    grid_dataids_t = grid_dataids[t_slices]

            res["nctimes"] = [d.strftime(DATETIME_FMT) for d in grid_datas_t]
            res["dataids"] = [_ndobject_to_str(d) for d in grid_dataids_t]

            res0[bandid] = res

        ret1 = {}
        ret1["bands"] = res0
        ret1["xy"] = [grid_x, grid_y]

        return ret1

    @lru_cache(maxsize=256, timeout=300, args_base=1)
    def info_by_bbox(self, sensor, minx, miny, maxx, maxy, crs=None, times=None, fmt="json"):
        '''
        返回空间范围所覆盖的数据切片信息，，返回：bands, crs, bbox, res, size, nctimes, geometry
        sensor：数据产品名称
        minx, miny, maxx, maxy：bbox范围  
        crs：矢量掩膜对应投影信息，如为：None，默认：EPSG:4326
        times：TimeSlice 可识别的时间条件 
        '''
        geom = bbox_polygon(minx, miny, maxx, maxy)
        try:
            return self.info_by_geom(sensor, geom, crs, times, fmt)
        except EInvalidGeom:
            raise EInvalidBBox(minx, miny, maxx, maxy)

    @lru_cache(maxsize=256, timeout=300, args_base=1)
    def info_by_geom(self, sensor, geom, crs=None, times=None, fmt="json"):
        '''
        返回空间范围所覆盖的数据切片信息，，返回：bands, crs, bbox, res, size, nctimes, geometry
        sensor：数据产品名称
        geom：矢量掩膜范围，支持 wkt 或 geojson  
        crs：矢量掩膜对应投影信息，如为：None，默认：EPSG:4326
        times：TimeSlice 可识别的时间条件 
        '''
        if len(geom) > GEOM_MAX_SIZE:
            raise EGeomTooLarge()

        geom_4326 = GeomTrans(crs, EPSG_4326).transform_geom(geom)  # self._get_geom_4326(geom, crs)
        if geom_4326.IsValid() == False:
            raise EInvalidGeom()

        xmin0, xmax0, ymin0, ymax0 = geom_4326.GetEnvelope()

        grids_xy = get_grids_by_bbox(xmin0, ymin0, xmax0, ymax0, self.gsize)
        grids_info = []

        for (grid_x, grid_y) in grids_xy:
            r = self._info_by_grid_xy(sensor, grid_x, grid_y, times).copy()
            if len(r.keys()) == 0:  # 存在 nc 文件
                continue

            t_bbox = get_grid_bbox(grid_x, grid_y, self.gsize)
            grid_geom = ogr.CreateGeometryFromWkt(bbox_polygon(*t_bbox))

            out_geom = grid_geom.Intersection(geom_4326)
            if out_geom is None:  # 不相交
                continue

            if out_geom.IsEmpty():
                continue

            r["geometry"] = out_geom.ExportToWkt()
            grids_info.append(r)

        if fmt == "json":
            return grids_info, json

        out_bytes = pickle.dumps(grids_info)
        return out_bytes, "bytes"

    @lru_cache(maxsize=256, timeout=300, args_base=1)
    def info_by_point(self, sensor, x0, y0, crs=None, times=None, fmt="json"):
        '''
        返回坐标点位置的数据切片信息，返回：bands, crs, bbox, res, size, nctimes
        sensor：数据产品名称
        x, y：坐标
        crs：坐标对应投影信息，如为：None，默认：EPSG:4326
        times：TimeSlice 可识别的时间条件 
        '''
        x, y = GeomTrans(crs, EPSG_4326).transform_point([x0, y0])

        grid_x, grid_y = get_grid_by_xy(x, y, self.gsize)
        r = self._info_by_grid_xy(sensor, grid_x, grid_y, times).copy()

        r["point"] = [x, y]

        if fmt == "json":
            return r, "json"

        out_bytes = pickle.dumps(r)
        return out_bytes, "bytes"

    @lru_cache(maxsize=256, timeout=300, args_base=1)
    def query_by_point(self, sensor, bandid, x0, y0, crs=None, times=None, fmt="json"):
        '''
        获取坐标点的数据。
                
        sensor    ：数据产品名称
        bandid   ：波段名称
        x, y：坐标
        crs：坐标对应投影信息，如为：None，默认：EPSG:4326
        times：TimeSlice 可识别的时间条件 
        '''

        x, y = GeomTrans(crs, EPSG_4326).transform_point([x0, y0])
        grid_x, grid_y = get_grid_by_xy(x, y, self.gsize)

        g_sensor = sensor.upper()

        ncfile = self._get_ncfile(g_sensor, grid_y, grid_x, bandid)
        if not os.path.exists(ncfile):
            return {}, "json"

        ncdataset_wrapper = get_ncfile_handler(ncfile)
        ncdataset = ncdataset_wrapper()

        grid_crs = ncdataset.grid_crs
        grid_bounds = ncdataset.grid_bounds
        grid_res = ncdataset.grid_res
        grid_size = list(map(lambda a: int(a), ncdataset.grid_size))

        x_proj, y_proj = GeomTrans(EPSG_4326, grid_crs).transform_point([x, y])

        gminx, gminy, gmaxx, gmaxy = map_bbox_win(grid_bounds, [x_proj, y_proj, x_proj, y_proj, ], grid_res)

        gymin1 = grid_size[1] - gmaxy
        gymax1 = grid_size[1] - gminy

        #         print(grid_x, grid_y)
        #         print(gminx, gymin1, gmaxx, gymax1)

        grid_values = ncdataset.variables["values"]
        grid_times = ncdataset.variables["times"]

        cdftime = utime(grid_times.units)

        grid_dtype = grid_values.grid_dtype
        np_otype = np.typeDict.get(grid_dtype)
        fill_value = grid_values._FillValue

        t_slices = []
        if times is not None:
            if isinstance(times, TimeSlice) == False:
                times = TimeSlice(times)
            try:
                t_slices = times.get_slices(ncdataset, cdftime)
            except ValueError as e:
                raise ETimeSlice(str(e))

        if len(t_slices) == 0:
            grid_datas = grid_values[:, gymin1, gminx]
            grid_datas_t = cdftime.num2date(grid_times[:])
        else:
            if len(t_slices[t_slices == True]) == 0:
                grid_datas = np.ndarray(shape=(1, 1, 0), dtype=np_otype)
                grid_datas_t = np.ndarray(shape=(0))
            else:
                grid_datas = grid_values[t_slices, gymin1, gminx]
                grid_datas_t = cdftime.num2date(grid_times[t_slices])

        ret = {
            "times": [d.strftime(DATETIME_FMT) for d in grid_datas_t],
            "nodata": float(fill_value),
            "q_xy": [float(x0), float(y0)],
            "g_xy": [gminx, gymin1],
            "g_no": [grid_x, grid_y]
        }

        if fmt == "json":
            ret["values"] = grid_datas.tolist()
            return ret, "json"

        ret["values"] = grid_datas
        out_bytes = pickle.dumps(ret)
        return out_bytes, "bytes"

    def query_by_geom(self, sensor, bandid, mask_geom, grid_x, grid_y, times=None, fmt="json"):
        '''
        获取空间范围内的数据，在调用该函数之前先调用 info_by_geom 或 info_by_bbox，将返回的结果中的 geometry 和 xy 属性作为参数。
                
        sensor    ：数据产品名称
        bandid   ：波段名称
        geom_info:矢量掩膜范围，必须包含 geometry 和 xy, geometry 必须为经纬度的 wkt 或 geojson，例如：  
            { 
                'geometry': 'POLYGON ((115.0 40.222408112063,115.0 40.3643188487053,115.312252122152 40.5,115.5 40.5,115.5 40.3046486467168,115.387540328212 40.2805104337424,115.0 40.222408112063))', 
                'xy': [590, 260], 
            }
        times：TimeSlice 可识别的时间条件 
        '''
        #         geom = geom_info.get("geometry", None)
        #         grid_xy = geom_info.get("xy", None)

        if len(mask_geom) > GEOM_MAX_SIZE:
            raise EGeomTooLarge()

        return self._query_by_geom(sensor, bandid, mask_geom, grid_x, grid_y, times, fmt)

    @lru_cache(maxsize=256, timeout=300, args_base=1)
    def _query_by_geom(self, sensor, bandid, mask_geom, grid_x, grid_y, times=None, fmt="json"):
        geom_4326 = GeomTrans(EPSG_4326, EPSG_4326).transform_geom(
            mask_geom)  # self._get_geom_4326(mask_geom, EPSG_4326)
        if geom_4326.IsValid() == False:
            raise EInvalidGeom()

        g_sensor = sensor.upper()

        ncfile = self._get_ncfile(g_sensor, grid_y, grid_x, bandid)
        if not os.path.exists(ncfile):
            return {}, "json"

        ncdataset_wrapper = get_ncfile_handler(ncfile, )
        ncdataset = ncdataset_wrapper()

        grid_crs = ncdataset.grid_crs
        grid_bounds = ncdataset.grid_bounds
        grid_res = ncdataset.grid_res
        grid_size = list(map(lambda a: int(a), ncdataset.grid_size))

        geom_proj = GeomTrans(EPSG_4326, grid_crs).transform_geom(mask_geom)

        xmin0, xmax0, ymin0, ymax0 = geom_proj.GetEnvelope()
        g_win_bbox, g_win_size = adjust_bbox([xmin0, ymin0, xmax0, ymax0], grid_res, ref_bbox=grid_bounds)

        xmin0, ymin0, xmax0, ymax0 = g_win_bbox
        xsize, ysize = g_win_size

        gminx, gminy, gmaxx, gmaxy = map_bbox_win(grid_bounds, g_win_bbox, grid_res)

        gymin1 = grid_size[1] - gmaxy
        gymax1 = grid_size[1] - gminy

        if gminx < 0 or gminy < 0 or gmaxx > grid_size[0] or gmaxy > grid_size[1]:
            return {}, "json"

        grid_values = ncdataset.variables["values"]
        grid_times = ncdataset.variables["times"]

        cdftime = utime(grid_times.units)

        t_slices = []
        if times is not None:
            if isinstance(times, TimeSlice) == False:
                times = TimeSlice(times)
            try:
                t_slices = times.get_slices(ncdataset, cdftime)
            except ValueError as e:
                raise ETimeSlice(str(e))

        grid_dtype = grid_values.grid_dtype
        np_otype = np.typeDict.get(grid_dtype)
        fill_value = grid_values._FillValue

        if len(t_slices) == 0:
            grid_datas = grid_values[:, gymin1:gymax1, gminx:gmaxx]
            grid_datas_t = cdftime.num2date(grid_times[:])
        else:
            if len(t_slices[t_slices == True]) == 0:
                out_image = grid_datas = np.ndarray(shape=(1, 1, 0), dtype=np_otype)
                grid_datas_t = np.ndarray(shape=(0))
            else:
                grid_datas = grid_values[t_slices, gymin1:gymax1, gminx:gmaxx]
                grid_datas_t = cdftime.num2date(grid_times[t_slices])

        if len(grid_datas_t) > 0:
            # begin apply geom mask
            geom_json = json.loads(geom_proj.ExportToJson())
            transform = rasterio.transform.from_bounds(xmin0, ymin0, xmax0, ymax0, xsize, ysize)

            out_shape = grid_datas[0].shape

            all_touched = False
            invert = False

            with rasterio.Env():
                geom_mask_2d = geometry_mask([geom_json], out_shape, transform, all_touched=all_touched, invert=invert)

            nc_mask = getattr(grid_datas, "mask", None)
            nc_data = getattr(grid_datas, "data", grid_datas)

            geom_mask_nd = np.ndarray(shape=grid_datas.shape, dtype=np.bool)
            geom_mask_nd[:] = geom_mask_2d

            if nc_mask is not None:
                geom_mask = np.logical_or(geom_mask_nd, nc_mask)
            else:
                geom_mask = nc_mask

            out_image = ma.masked_array(nc_data, mask=geom_mask, dtype=np_otype, fill_value=fill_value)

        ret = {
            "nodata": float(fill_value),
            "shape": list(map(lambda a: int(a), out_image.shape)),
            "times": [d.strftime(DATETIME_FMT) for d in grid_datas_t],
            "g_bbox": [gminx, gymin1, gmaxx, gymax1],
            "g_no": [grid_x, grid_y, ]
        }

        if fmt == "json":
            ret["values"] = out_image.tolist(fill_value)
            return ret, "json"

        if getattr(out_image, "filled", None):
            out_image = out_image.filled(fill_value)

        ret["values"] = out_image

        out_bytes = pickle.dumps(ret)
        return out_bytes, "bytes"
