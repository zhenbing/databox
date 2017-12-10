# distutils: language=c++

cimport cython
from libc.math cimport round, fmax, fmin, floor, ceil
from libc.stdint cimport uint64_t, int32_t, uint32_t
from libc.string cimport const_char
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair

import sys, os

import numpy as np
import numpy.ma as ma
cimport numpy as np

import ogr, osr

import datetime
import re
import json

import netCDF4
from netcdftime import utime

from collections import namedtuple
from functools import update_wrapper
from threading import RLock
import time

import rasterio
from rasterio.features import geometry_mask

try:
    import cPickle as pickle
except:
    import pickle

cdef extern from "DataBoxEngineHelper.hpp" namespace "GDAL":
    string _bbox_polygon(float minx, float miny, float maxx, float maxy) nogil;

    string _crs_to_proj4(string & crs) nogil;
# 
#     void c_get_grid_by_xy(float x, float y, float grid_size, int & grid_x, int & grid_y) nogil ; 
#     
#     void c_adjust_bbox(float * bbox, float * res, float * ref_bbox, int has_ref) nogil;

#     string float_encode_0(string & data, string & salt) nogil ;
#      
#     string float_decode_0(string & data, string & salt) nogil ;

cdef bytes decode_to_bytes(name):
    if name is None:
        return bytes()

    if isinstance(name, bytes):
        return name

    if not isinstance(name, unicode):
        raise TypeError(
            "'name' arg must be a byte string or a unicode string")

    encoding = sys.getfilesystemencoding() or 'ascii'
    try:
        return name.encode(encoding)
    except UnicodeEncodeError as exc:
        raise ValueError(
            "Cannot convert unicode 'name' to a file system name: %s" % exc)

# cdef array.array list_to_array(const_char typecode, list data):  
#     cdef array.array r = array.array(typecode, [ ])   
#     r.fromlist(data)
#     return r   


# cdef fix_float(float v): 
#     return round(v * 1000000.0) / 1000000.0 ;

def get_grid_by_xy(float x, float y, float grid_size):
    cdef float grid_x = ((x + 180) % 360) / grid_size  # 0 - 360 / grid_size, 0 base from west to east   
    cdef float grid_y = ((y + 90) % 360) / grid_size  # 0 - 180 / grid_size,  0 base from south to north           
    return int(grid_x), int(grid_y)

def get_grids_by_bbox(float minx0, float miny0, float maxx0, float maxy0, float grid_size):
    cdef float minx1, miny1, maxx1, maxy1
    cdef float grid_x, grid_y

    minx1 = minx0
    miny1 = miny0
    maxx1 = maxx0
    maxy1 = maxy0

    cdef list grids = []
    while miny0 < maxy1:
        while minx0 < maxx1:
            grid_x = (minx0 + 180) / grid_size  # 0 - 360 / grid_size, 0 base from west to east   
            grid_y = (miny0 + 90) / grid_size  # 0 - 180 / grid_size,  0 base from south to north   

            grids.append([int(grid_x), int(grid_y)])

            minx0 = minx0 + grid_size

        miny0 = miny0 + grid_size
        minx0 = minx1

    return grids

def get_grid_bbox(float grid_x, float grid_y, float grid_size):
    cdef float minx, miny, maxx, maxy

    minx = grid_x * grid_size - 180
    miny = grid_y * grid_size - 90

    maxx = minx + grid_size
    maxy = miny + grid_size

    return minx, miny, maxx, maxy

def adjust_bbox(bbox, res, ref_bbox=None):
    cdef float minx, miny, maxx, maxy
    cdef float xres, yres
    cdef float minx0, miny0, maxx0, maxy0
    cdef int xoff, yoff, xsize, ysize

    minx, miny, maxx, maxy = bbox
    xres, yres = res
    if ref_bbox is None:
        minx = round(minx / 10) * 10
        miny = round(miny / 10) * 10
    else:
        minx0, miny0, maxx0, maxy0 = ref_bbox

        xoff = int(round((minx - minx0) / xres))
        minx = minx0 + xoff * xres

        yoff = int(round((miny - miny0) / yres))
        miny = miny0 + yoff * yres

    xsize = int(round((maxx - minx) / xres))
    ysize = int(round((maxy - miny) / yres))

    maxx = minx + xsize * xres
    maxy = miny + ysize * yres

    return (minx, miny, maxx, maxy), (xsize, ysize)

def intersect_bbox(this_bbox, other_bbox):
    cdef float xmin0, ymin0, xmax0, ymax0
    cdef float xmin1, ymin1, xmax1, ymax1

    xmin0, ymin0, xmax0, ymax0 = this_bbox
    xmin1, ymin1, xmax1, ymax1 = other_bbox

    cdef float xmin = fmax(xmin0, xmin1)  # bboxs[:, 0].max()
    cdef float ymin = fmax(ymin0, ymin1)  # bboxs[:, 1].max()
    cdef float xmax = fmin(xmax0, xmax1)  # bboxs[:, 2].min()
    cdef float ymax = fmin(ymax0, ymax1)  # bboxs[:, 3].min()

    return xmin, ymin, xmax, ymax

def union_bbox(this_bbox, other_bbox):  # minx, miny, maxx, maxy  

    cdef float xmin0, ymin0, xmax0, ymax0
    cdef float xmin1, ymin1, xmax1, ymax1

    xmin0, ymin0, xmax0, ymax0 = this_bbox
    xmin1, ymin1, xmax1, ymax1 = other_bbox

    cdef float xmin = fmin(xmin0, xmin1)  # bboxs[:, 0].max()
    cdef float ymin = fmin(ymin0, ymin1)  # bboxs[:, 1].max()
    cdef float xmax = fmax(xmax0, xmax1)  # bboxs[:, 2].min()
    cdef float ymax = fmax(ymax0, ymax1)  # bboxs[:, 3].min()

    return xmin, ymin, xmax, ymax

def map_bbox_win(dst_bbox, src_bbox, res):  # minx, miny, maxx, maxy
    cdef float xres, yres
    cdef float xmin0, ymin0, xmax0, ymax0
    cdef float xmin1, ymin1, xmax1, ymax1
    cdef int xmin, ymin, xmax, ymax
    cdef float xoff_f, yoff_f, xsize_f, ysize_f

    xres, yres = res
    xmin0, ymin0, xmax0, ymax0 = dst_bbox
    xmin1, ymin1, xmax1, ymax1 = src_bbox

    xoff_f = (xmin1 - xmin0) / xres
    yoff_f = (ymin1 - ymin0) / yres

    xmin = int(floor(xoff_f))  # 往左对齐
    ymin = int(ceil(yoff_f))  # 往上对齐

    xsize_f = (xmax1 - xmin1) / xres
    ysize_f = (ymax1 - ymin1) / yres

    xmax = int(floor(xsize_f + xoff_f))  # 往左边对齐
    ymax = int(ceil(ysize_f + yoff_f))  # 往上对齐

    return xmin, ymin, xmax, ymax

def paste_ndarray(dst_array, dst_bbox, src_array, src_bbox, res):  # minx, miny, maxx, maxy
    tmp_bbox = union_bbox(dst_bbox, src_bbox)
    max_bbox, max_size = adjust_bbox(tmp_bbox, res, ref_bbox=dst_bbox)

    #     print(src_bbox)
    #     print(dst_bbox)
    #     print(max_bbox)

    dst_window = map_bbox_win(max_bbox, dst_bbox, res)
    src_window = map_bbox_win(max_bbox, src_bbox, res)

    #     print(src_window, dst_window)

    cdef int ndim0 = 1
    src_shape = src_array.shape
    if len(src_shape) == 3:
        ndim0 = src_shape[0]

    cdef int ndim1 = 1
    if dst_array is not None:
        dst_shape = dst_array.shape  # z, y, x
        if len(dst_shape) == 3:
            ndim1 = dst_shape[0]

    cdef int ndim = max(ndim0, ndim1)
    cdef int xsize = max_size[0]
    cdef int ysize = max_size[1]

    cdef np.ndarray res_array = np.ndarray([ndim, ysize, xsize, ])
    res_array = ma.masked_equal(res_array, 0)

    if dst_array is not None:
        for dim in range(ndim1):
            if isinstance(dst_array, np.ndarray):
                t_array = dst_array[dim, :, :]
            else:
                t_array = dst_array
        xmin0, ymin0, xmax0, ymax0 = dst_window
        res_array[dim, ymin0:ymax0, xmin0:xmax0] = t_array

    for dim in range(ndim0):
        if isinstance(src_array, np.ndarray):
            t_array = src_array[dim, :, :]
        else:
            t_array = src_array

        xmin0, ymin0, xmax0, ymax0 = src_window
        ymin1 = ysize - ymax0
        ymax1 = ysize - ymin0
        res_array[dim, ymin1:ymax1, xmin0:xmax0] = t_array  # z, y, x

    return res_array, max_bbox, max_size

def bbox_polygon(float minx, float miny, float maxx, float maxy):
    cdef string wkt = _bbox_polygon(minx, miny, maxx, maxy)
    return wkt.decode("utf-8")

def crs_to_proj4(crs):
    cdef bytes b_crs = decode_to_bytes(crs)
    cdef string proj4 = _crs_to_proj4(b_crs)
    return proj4.decode("utf-8")

####################GeomTrans#######################
EPSG_4326 = "EPSG:4326"


class GeomTrans(object):
    def __init__(self, in_proj, out_proj):
        self.transform = None

        if in_proj:
            self.inSpatialRef = osr.SpatialReference()
            self.inSpatialRef.SetFromUserInput(in_proj)
        else:
            return

        if out_proj:
            self.outSpatialRef = osr.SpatialReference()
            self.outSpatialRef.SetFromUserInput(out_proj)
        else:
            return

        if self.inSpatialRef.IsSame(self.outSpatialRef) == 0:
            self.transform = osr.CoordinateTransformation(self.inSpatialRef, self.outSpatialRef)

    def transform_point(self, point):
        if self.transform is None:
            return point

        geom = ogr.Geometry(ogr.wkbPoint)
        geom.AddPoint(point[0], point[1])
        geom.Transform(self.transform)

        return geom.GetX(), geom.GetY()

    def transform_points(self, points):
        return [self.transform_point(point) for point in points]

    def transform_geom(self, geometry):
        if geometry.find('{') >= 0:
            geom = ogr.CreateGeometryFromJson(geometry)
        else:
            geom = ogr.CreateGeometryFromWkt(geometry)

        if self.transform is not None:
            geom.Transform(self.transform)

        return geom

    def transform_wkt(self, geometry):
        return self.transform_geom(geometry).ExportToWkt()

    def transform_json(self, geometry):
        return self.transform_geom(geometry).ExportToJson()


####################TimeSlice#######################

DATETIME_FMT = "%Y-%m-%d %H:%M:%S"
DATE_FMT = "%Y-%m-%d"

# raw_sql = '''    not (year < 10 or date == "world") or (day>10) and not ( day == 10 and day ==23  )  and month in [24,    45, 654]   '''

# raw_sql = ''' (date == 0.5) and ( year not in [24, 654]  ) or not ( day == 1 and month > 3)  '''

# raw_sql = ''' x>1 and ((x>1) or ( not (    x   <   2) and x==34) or (z==1))'''


COND_TOKENS = ["date", "year", "month", 'day', "hour", "minute"]
COND_OPCODES = ["in", "notin", ">=", "<=", ">", "<", "==", "!="]
COND_LOGICALS = ["and", "or"]


class CondParser(object):
    def __init__(self, raw_sql):

        raw_sql = re.sub("==", " == ", raw_sql)
        raw_sql = re.sub(">", " > ", raw_sql)
        raw_sql = re.sub("<", " < ", raw_sql)
        raw_sql = re.sub(">=", " >= ", raw_sql)
        raw_sql = re.sub("<=", " <= ", raw_sql)

        raw_sql = re.sub("\s+", " ", raw_sql)
        raw_sql = re.sub(r",\s+", ",", raw_sql)
        raw_sql = re.sub(r":\s+", ":", raw_sql)
        raw_sql = re.sub(r"not\s+in", "notin", raw_sql)
        raw_sql = re.sub(r"\(\s+", "(", raw_sql)
        raw_sql = re.sub(r"\s+\)", ")", raw_sql)

        #         print(raw_sql)
        self.raw_sql = raw_sql

    def _get_brackets(self, s):
        left = 1
        for idx, c in enumerate(s):
            if c == "(": left += 1
            if c == ")": left -= 1
            if left == 0: return idx, s[:idx]
        raise ValueError("invalid expr: (" + s.replace("notin", "not in"))

    def parse(self):
        rst = []
        self._parse(self.raw_sql, rst)
        return rst

    def validate(self, s, cond):
        l = len(cond)
        if l == 1:
            raise ValueError("invalid expr: " + s)
        if l == 2:
            opcode, value = cond
            if opcode != "not":
                raise ValueError("invalid expr: " + " ".join(cond))
            if not isinstance(value, list):
                raise ValueError("invalid expr: " + " ".join(cond))
            return

        if l > 3:
            cond_r = cond[2:]
            token = cond[0];
            opcode = cond[1]
            #             cond.clear( )
            cond[:] = [cond[0], cond[1], " ".join(cond_r)]
            l = len(cond)

        if l == 3:
            token, opcode, value = cond
            if token not in COND_TOKENS or opcode not in COND_OPCODES:
                raise ValueError("invalid expr: " + " ".join(cond))
            done = False;
            value = cond[-1];
            if len(value) > 1:
                t0 = value[0];
                t1 = value[-1]
                if t0 in ['"', "'"] and t1 in ['"', "'"]:
                    value = value[1:-1]  # is string
                    done = True
                elif t0 == "[" and t1 == "]":
                    try:
                        if value.find("{") >= 0: raise
                        value = value.replace("'", '"')
                        value = json.loads(value)  # is array
                    except:
                        raise ValueError("invalid expr: " + value)
                    if opcode not in ["in", "notin"]:
                        raise ValueError("invalid expr: " + opcode + " " + json.dumps(value))
                    done = True
            if done == False:
                if value.find(".") >= 0:
                    try:
                        value = float(value)  # is float
                    except:
                        raise ValueError("invalid expr: " + value)
                else:
                    try:
                        value = int(value)  # is int
                    except:
                        raise ValueError("invalid expr: " + value)
            cond[-1] = value
            if opcode in ["in", "notin"]:
                if not isinstance(value, list):
                    raise ValueError("invalid expr: " + opcode + " " + json.dumps(value))
            if opcode == "notin": cond[1] = "not in"
            return
        raise ValueError("invalid expr: " + s)

    def _parse(self, s, rst):
        s = s.strip();
        l = len(s)
        beg, idx, tmp = 0, 0, []
        while idx < l:
            c = s[idx]
            if c == "(":
                pos, token = self._get_brackets(s[idx + 1:])
                sub_cond = []
                self._parse(token, sub_cond)
                tmp.append(sub_cond)
                idx = idx + pos + 3;
                beg = idx
            elif c == " ":
                token = s[beg:idx]
                idx += 1;
                beg = idx
                tmp.append(token)
            else:
                idx += 1

        if beg < idx:
            tmp.append(s[beg:idx])

        idx = 0;
        l = len(tmp)
        if l == 0: return
        if l == 1:
            token = tmp[0]
            if isinstance(token, list):
                rst.extend(token)
                return
            else:
                raise ValueError("invalid expr: " + s.replace("notin", "not in"))

        cond = [];
        token = None
        while idx < l:
            token = tmp[idx]
            if isinstance(token, list):
                rst.append(token)
                if idx + 1 < l:
                    opcode = tmp[idx + 1]
                    if opcode not in COND_LOGICALS:
                        raise ValueError("invalid expr: " + json.dumps(token) + " " + opcode)
                    rst.append(opcode)
                idx += 2
            elif token == "not":
                value = tmp[idx + 1]
                if not isinstance(value, list):
                    raise ValueError("invalid expr: not " + value)
                rst.append(["not", value])
                if idx + 2 < l:
                    opcode = tmp[idx + 2]
                    if opcode not in COND_LOGICALS:
                        raise ValueError("invalid expr: " + json.dumps(token) + " " + opcode)
                    rst.append(opcode)
                idx += 3
            elif token in COND_LOGICALS:
                self.validate(s, cond)
                rst.append(cond)
                rst.append(token)
                cond = []
                idx += 1
            else:
                cond.append(token)
                idx += 1

        if len(cond) > 0:
            self.validate(s, cond)
            if len(rst) == 0:
                rst.extend(cond)
            else:
                rst.append(cond)

        if len(rst) > 0:
            last = rst[-1]
            if last in COND_LOGICALS:
                raise ValueError("invalid expr: " + s.replace("notin", "not in"))


def _parse_dates(times):
    '''
    如果 times 为 datetime，返回 [ datetime ]
    如果 times 为 %Y-%m-%d %H:%M:%S，返回 [ datetime ]
    '''
    if times is None: return []

    if not isinstance(times, (tuple, list)):
        times = [times]

    rets = []
    for atime in times:
        if isinstance(atime, datetime.datetime):
            rets.append(atime)
        else:
            stime = atime.strip()
            try:
                atime = datetime.datetime.strptime(stime, DATE_FMT)
            except:
                atime = datetime.datetime.strptime(stime, DATETIME_FMT)
            rets.append(atime)
    return rets


# print(CondParser(raw_sql).parse())

class TimeSlice(object):
    def __init__(self, timeval=None, ):
        '''
        timeval 为 datetime, datestr, timerange 对象 或 列表    
        [ ["key", "opcode", "val"], "and/or", [ "not", ["key", "opcode", "val"]] ]
        if key == "in", val 可以为 ["value1", "value2"]
        else:  val 可以为 value 或 ["key", "opcode", "val"]
        
        key    : "date", "year", "month", 'day', "hour", "minute"
        opcode3: "in", "not in", ">=", "<=", ">", "<", "==", "!="
        opcode2: "not"
                
        '''
        self.timeval = timeval

    def cmp_oper(self, key, opcode, val, ncdataset, cdftime):
        ''' 
        key    : "date", "year", "month", 'day', "hour", "minute"
        opcode3: "in", "not in", ">=", "<=", ">", "<", "==", "!="
        opcode2: "not"
        '''

        grid_times = ncdataset.variables["times"]
        grid_years = ncdataset.variables["years"]
        grid_months = ncdataset.variables["months"]
        grid_days = ncdataset.variables["days"]
        grid_hours = ncdataset.variables["hours"]
        grid_minutes = ncdataset.variables["minutes"]

        def get_cubeval(key):
            if key == "date":
                return grid_times[:]
            elif key == "year":
                return grid_years[:]
            elif key == "month":
                return grid_months[:]
            elif key == "days":
                return grid_days[:]
            elif key == "hours":
                return grid_hours[:]
            elif key == "minutes":
                return grid_minutes[:]
            raise ValueError("invalid key: %s" % key)

        if opcode in ["in", "not in"]:  # 如果是 in/not in 操作符，val 为数值数组
            if not isinstance(val, (tuple, list)):
                raise ValueError("require list/tuple: %s %s %s" % (key, opcode, str(val)))

            cubeval = get_cubeval(key)
            if key == "date":
                timeval = np.array(_parse_dates(val))
                num_timeval = cdftime.date2num(timeval)
            else:
                num_timeval = np.array(list(map(lambda a: int(a), val)))

            ix = np.in1d(cubeval.ravel(), num_timeval).reshape(cubeval.shape)
            if opcode == "in":
                return ix
            else:
                return np.logical_not(ix)

        if isinstance(val, (tuple, list)):
            raise ValueError("require string: %s %s %s" % (key, opcode, str(val)))

        cubeval = get_cubeval(key)
        if key == "date":
            timeval = np.array(_parse_dates(val))
            num_timeval = cdftime.date2num(timeval)[0]
        else:
            num_timeval = int(val)

        if opcode == ">=":
            return cubeval >= num_timeval
        elif opcode == "<=":
            return cubeval <= num_timeval
        elif opcode == ">":
            return cubeval > num_timeval
        elif opcode == "<":
            return cubeval < num_timeval
        elif opcode == "==":
            return cubeval == num_timeval
        elif opcode == "!=":
            return cubeval != num_timeval

        raise ValueError("invalid expr: %s %s %s" % (key, opcode, str(val)))

    def log_oper(self, expr1, opcode, expr2, ncdataset, cdftime):
        '''
        opcode : "and", "or"
        '''
        if not isinstance(expr1, (tuple, list)):
            raise ValueError("require list/tuple: %s %s" % (str(expr1), opcode))
        if not isinstance(expr1, (tuple, list)):
            raise ValueError("require list/tuple: %s %s" % (opcode, str(expr2),))

        ix1 = self.one_oper(expr1, ncdataset, cdftime)
        ix2 = self.one_oper(expr2, ncdataset, cdftime)

        if opcode == "and":
            return np.logical_and(ix1, ix2)
        elif opcode == "or":
            return np.logical_or(ix1, ix2)
        raise ValueError("invalid expr: %s %s %s" % (str(expr1), opcode, str(expr2)))

    def not_oper(self, opcode, val, ncdataset, cdftime):
        if isinstance(val, (tuple, list)):  # not 操作符, val 为表达式，需要重新迭代计算
            ix = self.one_oper(val, ncdataset, cdftime)
            return np.logical_not(ix)

        raise ValueError("require list/tuple: %s %s" % (opcode, str(val)))

    def one_oper(self, expr, ncdataset, cdftime):
        '''        
        [ ["date", "in", ['2010-08-08', '2010-09-25']] , "or" , ["not", [ ["date", ">=", '2010-09-01 12:00:00' ] , "or" , ["month", "==", 9 ] ] ] ] 
        
        not: len(expr) == 2
        in, not in, >=, <=, >, <, ==, !=, and, or : len(expr) == 3
        array                        : others 
        '''
        l = len(expr)
        while l == 1:
            expr = expr[0]
            l = len(expr)

        if l == 2:  # 如果长度为 2，则为 not 数组
            opcode, val = expr
            if opcode == "not":
                rst2 = self.not_oper(opcode, val, ncdataset, cdftime)
            else:
                raise ValueError(str(expr))

        elif l == 3:
            key, opcode, val = expr
            if opcode in ["and", "or"]:
                rst2 = self.log_oper(key, opcode, val, ncdataset, cdftime)

            elif opcode in ["in", "not in", ">=", "<=", ">", "<", "==", "!="]:

                if key not in ["date", "year", "month", 'day', "hour", "minute"]:
                    raise ValueError("bad key: %s %s %s" % (key, opcode, str(val)))

                rst2 = self.cmp_oper(key, opcode, val, ncdataset, cdftime)
            else:
                raise ValueError(str(expr))

        else:
            if l % 2 == 0:
                raise ValueError(str(expr))

            for pos in range(1, l, 2):
                opcode = expr[pos]
                if opcode not in ["and", "or"]:
                    raise ValueError(str(expr))

            expr0 = expr[0]
            opcode = expr[1]
            expr1 = expr[2]

            rst2 = self.log_oper(expr0, opcode, expr1, ncdataset, cdftime)

            for pos in range(3, l, 2):
                opcode = expr[pos]
                expr1 = expr[pos + 1]

                if not isinstance(expr1, (tuple, list)):
                    raise ValueError("require list/tuple: %s %s" % (str(expr1), opcode))

                ix1 = self.one_oper(expr1, ncdataset, cdftime)

                if opcode == "and":
                    rst2 = np.logical_and(ix1, rst2)
                elif opcode == "or":
                    rst2 = np.logical_or(ix1, rst2)
        return rst2

    def get_slices(self, ncdataset, cdftime):
        if not self.timeval:
            return []

        #         if isinstance(self.timeval, [ bytes ]):
        #             self.timeval = self.timeval.encode("utf-8")

        if not isinstance(self.timeval, (tuple, list)):
            self.timeval = CondParser(self.timeval).parse()

        ix = self.one_oper(self.timeval, ncdataset, cdftime)

        return ix

    ####################LRUCache#######################


_CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])


class _HashedSeq(list):
    """ This class guarantees that hash() will be called no more than once
        per element.  This is important because the lru_cache() will hash
        the key multiple times on a cache miss.

    """

    __slots__ = 'hashvalue'

    def __init__(self, tup):
        self[:] = tup
        self.hashvalue = hash(tup)

    def __hash__(self):
        return self.hashvalue


def _make_key(args, kwds, typed,
              kwd_mark=(object(),),
              fasttypes={int, str, frozenset, type(None)}):
    """Make a cache key from optionally typed positional and keyword arguments

    The key is constructed in a way that is flat as possible rather than
    as a nested structure that would take more memory.

    If there is only a single argument and its data type is known to cache
    its hash value, then that argument is returned without a wrapper.  This
    saves space and improves lookup speed.

    """
    key = args
    if kwds:
        sorted_items = sorted(kwds.items())
        key += kwd_mark
        for item in sorted_items:
            key += item
    if typed:
        key += tuple(type(v) for v in args)
        if kwds:
            key += tuple(type(v) for k, v in sorted_items)
    elif len(key) == 1 and type(key[0]) in fasttypes:
        return key[0]
    return _HashedSeq(key)

def lru_cache(int maxsize=100, int timeout=600, bool typed=False, int args_base=0):
    def _cache_controller(viewfunc):

        cache = dict()
        stats = [0, 0]  # make statistics updateable non-locally
        HITS, MISSES = 0, 1  # names for the stats fields
        cache_get = cache.get  # bound method to lookup key or return None
        _len = len  # localize the global len() function
        lock = RLock()  # because linkedlist updates aren't threadsafe
        root = []  # root of the circular doubly linked list
        root[:] = [root, root, None, None]  # initialize by pointing to self
        nonlocal_root = [root]  # make updateable non-locally
        PREV, NEXT, KEY, RESULT = 0, 1, 2, 3  # names for the link fields

        if maxsize == 0:
            def wrapper(*args, **kwds):
                # no caching, just do a statistics update after a successful call
                result = viewfunc(*args, **kwds)
                stats[MISSES] += 1
                return result

        elif maxsize is None:
            def wrapper(*args, **kwds):
                t_args = args if args_base == 0 else args[1:]
                # simple caching without ordering or size limit
                key = _make_key(t_args, kwds, typed)

                result = cache_get(key, root)  # root used here as a unique not-found sentinel

                if result is not root:
                    old_time = result[1]

                    if timeout is not None:
                        if (int(time.time()) - old_time) <= timeout:
                            stats[HITS] += 1
                            return result[0]
                    else:
                        stats[HITS] += 1
                        return result[0]

                cache[key] = result = viewfunc(*args, **kwds), int(time.time())
                stats[MISSES] += 1
                return result[0]

        else:
            def wrapper(*args, **kwds):
                t_args = args if args_base == 0 else args[1:]
                # size limited caching that tracks accesses by recency
                key = _make_key(t_args, kwds, typed) if kwds or typed else t_args

                with lock:
                    link = cache_get(key)
                    if link is not None:
                        # record recent use of the key by moving it to the front of the list
                        root, = nonlocal_root
                        link_prev, link_next, key, result = link

                        old_time = result[1]

                        if timeout is not None:
                            if (int(time.time()) - old_time) <= timeout:
                                link_prev[NEXT] = link_next
                                link_next[PREV] = link_prev
                                last = root[PREV]
                                last[NEXT] = root[PREV] = link
                                link[PREV] = last
                                link[NEXT] = root
                                stats[HITS] += 1

                                return result[0]
                        else:
                            link_prev[NEXT] = link_next
                            link_next[PREV] = link_prev
                            last = root[PREV]
                            last[NEXT] = root[PREV] = link
                            link[PREV] = last
                            link[NEXT] = root
                            stats[HITS] += 1

                            return result[0]

                result = viewfunc(*args, **kwds), int(time.time())

                with lock:
                    root, = nonlocal_root
                    stats[MISSES] += 1
                    #                     if key in cache:
                    #                         pass
                    if _len(cache) >= maxsize:
                        # use the old root to store the new key and result
                        oldroot = root
                        oldroot[KEY] = key

                        oldroot[RESULT] = result

                        # empty the oldest link and make it the new root
                        root = nonlocal_root[0] = oldroot[NEXT]
                        oldkey = root[KEY]
                        oldvalue = root[RESULT]
                        root[KEY] = root[RESULT] = None
                        # now update the cache dictionary for the new links
                        if oldkey in cache:
                            del cache[oldkey]
                        cache[key] = oldroot
                    else:
                        # put result in a new link at the front of the list
                        last = root[PREV]
                        link = [last, root, key, result]
                        last[NEXT] = root[PREV] = cache[key] = link

                return result[0]

        def cache_info():
            """Report cache statistics"""
            with lock:
                return _CacheInfo(stats[HITS], stats[MISSES], maxsize, len(cache))

        def cache_clear():
            """Clear the cache and cache statistics"""
            with lock:
                cache.clear()
                root = nonlocal_root[0]
                root[:] = [root, root, None, None]
                stats[:] = [0, 0]

        wrapper.__wrapped__ = viewfunc
        #         wrapper.cache_info = cache_info
        #         wrapper.cache_clear = cache_clear
        return update_wrapper(wrapper, viewfunc)

    return _cache_controller


#######################netCDF4#######################

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

##########################DataBoxQuery##############################


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
            mask_geom)  #  self._get_geom_4326(mask_geom, EPSG_4326)
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
