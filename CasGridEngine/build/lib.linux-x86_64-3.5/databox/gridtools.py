import math

import ogr, osr

import numpy as np
import numpy.ma as ma


def get_grid_by_xy(x, y, grid_size):
    # x = ( -180 ~ 180 ),  
    grid_x = int(((x + 180) % 360) / grid_size)  # 0 - 360 / grid_size, 0 base from west to east

    # y = ( -90 ~ 90 )        
    grid_y = int(((y + 90) % 360) / grid_size)  # 0 - 180 / grid_size,  0 base from south to north   

    return grid_x, grid_y


def get_grids_by_bbox(minx0, miny0, maxx0, maxy0, grid_size):
    minx1 = minx0
    miny1 = miny0
    maxx1 = maxx0
    maxy1 = maxy0

    grids = []
    while miny0 < maxy1:
        while minx0 < maxx1:
            grids.append(get_grid_by_xy(minx0, miny0, grid_size))
            minx0 = minx0 + grid_size

        miny0 = miny0 + grid_size
        minx0 = minx1
    return grids


def get_grid_bbox(grid_x, grid_y, grid_size):
    minx = grid_x * grid_size - 180
    miny = grid_y * grid_size - 90
    maxx = minx + grid_size
    maxy = miny + grid_size
    return float(minx), float(miny), float(maxx), float(maxy)


def intersect_bbox(this_bbox, other_bbox):
    xmin0, ymin0, xmax0, ymax0 = this_bbox
    xmin1, ymin1, xmax1, ymax1 = other_bbox

    xmin = max(xmin0, xmin1)  # bboxs[:, 0].max()
    ymin = max(ymin0, ymin1)  # bboxs[:, 1].max()
    xmax = min(xmax0, xmax1)  # bboxs[:, 2].min()
    ymax = min(ymax0, ymax1)  # bboxs[:, 3].min()

    return xmin, ymin, xmax, ymax


def union_bbox(this_bbox, other_bbox):  # minx, miny, maxx, maxy
    xmin0, ymin0, xmax0, ymax0 = this_bbox
    xmin1, ymin1, xmax1, ymax1 = other_bbox

    xmin = min(xmin0, xmin1)  # bboxs[:, 0].max()
    ymin = min(ymin0, ymin1)  # bboxs[:, 1].max()
    xmax = max(xmax0, xmax1)  # bboxs[:, 2].min()
    ymax = max(ymax0, ymax1)  # bboxs[:, 3].min()

    return xmin, ymin, xmax, ymax


def adjust_bbox(bbox, res, ref_bbox=None):
    minx, miny, maxx, maxy = bbox
    xres, yres = res

    if ref_bbox is None:
        minx = round(minx / 10) * 10
        miny = round(miny / 10) * 10
    else:
        minx0, miny0, _, _ = ref_bbox

        xoff = int(round((minx - minx0) / xres))
        minx = minx0 + xoff * xres

        yoff = int(round((miny - miny0) / yres))
        miny = miny0 + yoff * yres

    xsize = int(round((maxx - minx) / xres))
    ysize = int(round((maxy - miny) / yres))

    maxx = minx + xsize * xres
    maxy = miny + ysize * yres

    return (minx, miny, maxx, maxy), (xsize, ysize)


def bbox_polygon(minx, miny, maxx, maxy):
    wkt = 'POLYGON(({minx} {miny}, {minx} {maxy}, {maxx} {maxy}, {maxx} {miny}, {minx} {miny}))'.format(minx=minx,
                                                                                                        miny=miny,
                                                                                                        maxx=maxx,
                                                                                                        maxy=maxy)
    return wkt


def map_bbox_win(dst_bbox, src_bbox, res):  # minx, miny, maxx, maxy
    '''
    将 src_bbox 映射到 dst_bbox 上，返回格网偏移范围，相对于左下角坐标
    '''
    xres, yres = res
    xmin0, ymin0, xmax0, ymax0 = dst_bbox
    xmin1, ymin1, xmax1, ymax1 = src_bbox

    xoff_f = (xmin1 - xmin0) / xres
    yoff_f = (ymin1 - ymin0) / yres

    xmin = int(math.floor(xoff_f))  # 往左对齐
    ymin = int(math.ceil(yoff_f))  # 往上对齐

    xsize_f = (xmax1 - xmin1) / xres
    ysize_f = (ymax1 - ymin1) / yres

    xmax = int(math.floor(xsize_f + xoff_f))  # 往左边对齐
    ymax = int(math.ceil(ysize_f + yoff_f))  # 往上对齐

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

    ndim0 = 1
    src_shape = src_array.shape
    if len(src_shape) == 3:
        ndim0 = src_shape[0]

    ndim1 = 1
    if dst_array is not None:
        dst_shape = dst_array.shape  # z, y, x
        if len(dst_shape) == 3:
            ndim1 = dst_shape[0]

    ndim = max(ndim0, ndim1)

    res_array = np.ndarray([ndim, max_size[1], max_size[0], ])
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
        ymin1 = max_size[1] - ymax0
        ymax1 = max_size[1] - ymin0
        res_array[dim, ymin1:ymax1, xmin0:xmax0] = t_array  # z, y, x

    return res_array, max_bbox, max_size


def crs_to_proj4(crs):
    sr = osr.SpatialReference()
    sr.SetFromUserInput(crs)
    return sr.ExportToProj4()


def is_geojson(geometry):
    if geometry.find("{") >= 0:
        return True
    return False


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
        if is_geojson(geometry):
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


if __name__ == "__main__":
    #     gt = GeomTrans("EPSG:4326", "EPSG:3857")
    #     print(gt.transform_points([ [ 123.456, 32.567  ]]))
    print(union_bbox([123, 23, 145, 45], [134, 23, 143, 34], [145, 32, 146, 33]))
