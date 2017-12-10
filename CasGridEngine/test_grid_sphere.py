'''
Created on Aug 22, 2017

@author: root
'''

import os
import ogr, osr


def create_sphere_grid(shapefile, gsize):
    dr = ogr.GetDriverByName("ESRI Shapefile")
    ds = dr.CreateDataSource(shapefile)
    sr = osr.SpatialReference()
    sr.SetFromUserInput("EPSG:4326")

    lyr = ds.CreateLayer("polygon", sr, ogr.wkbPolygon)

    xoff = -180
    yoff = 90

    idx = 0

    while yoff > -90:
        while xoff < 180:
            ffd = ogr.FeatureDefn()

            fgd = ogr.GeomFieldDefn()
            fgd.name = "the_geom"
            fgd.type = ogr.wkbPolygon

            ffd.AddGeomFieldDefn(fgd)

            feat = ogr.Feature(ffd)

            geom = ogr.Geometry(ogr.wkbLinearRing)
            geom.AddPoint(xoff, yoff)
            geom.AddPoint(xoff + gsize, yoff)
            geom.AddPoint(xoff + gsize, yoff - gsize)
            geom.AddPoint(xoff, yoff - gsize)
            geom.AddPoint(xoff, yoff)

            geom_p = ogr.Geometry(ogr.wkbPolygon)
            geom_p.AddGeometry(geom)

            feat.SetGeometry(geom_p)

            lyr.CreateFeature(feat)

            idx += 1
            xoff += gsize

        yoff -= gsize
        xoff = -180


if __name__ == '__main__':
    create_sphere_grid(os.path.join("/tmp/b.shp"), 1)
