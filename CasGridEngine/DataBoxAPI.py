# coding:utf-8
'''
Created on Sep 25, 2017

@author: root
'''
import sys
import json
from urllib.parse import urlencode
from urllib3.connectionpool import HTTPConnectionPool

try:
    import cPickle as pickle
except:
    import pickle

url_prefix = "/databox"


class DataBoxConnector(object):
    def __init__(self, host, port, user="", passwd=""):
        self.c = HTTPConnectionPool(host, port)
        url = url_prefix + "/metadata/authenticate"
        headers = {
            "content-type": "application/x-www-form-urlencoded",
        }

        params = {
            "user": user,
            "passwd": passwd
        }
        resp = self.c.urlopen("POST", url, body=urlencode(params), headers=headers)

        jdata = json.loads(resp.data.decode("utf-8"))
        self.auth_code = jdata["auth_code"]

    def _load_resp(self, resp):
        ctype = resp.headers["Content-Type"]

        #         print( resp.headers )

        code = resp.headers.get("Error-Code", None)
        reason = resp.headers.get("Error-Message", None)

        if ctype.startswith("application/json") == True:
            jdata = json.loads(resp.data.decode("utf-8"))
        elif ctype.startswith("application/pickle-bytes") == True:
            jdata = pickle.loads(resp.data)
        else:
            jdata = None

        return jdata, resp.status, code, reason

    def _info_with(self, product, ctype, crs, params):
        url = url_prefix + "/info_with/{product}/{ctype}".format(product=product, ctype=ctype)

        params["crs"] = "" if crs is None else crs
        params["format"] = "bytes"

        headers = {
            "content-type": "application/x-www-form-urlencoded",
            "auth_code": self.auth_code
        }

        #         print(params)

        resp = self.c.urlopen("POST", url, body=urlencode(params), headers=headers)
        return self._load_resp(resp)

    def info_by_point(self, product, ctype, x, y, crs=None, timeslice=None):
        '''
        杩斿洖 x,y 鍧愭爣鐐规墍鍦� cube 淇℃伅锛岃繑鍥烇細bands, crs, bbox, res, size锛寈y, nctimes
        ctype锛氭暟鎹骇鍝佸悕绉�
        x, y锛氬潗鏍�
        crs锛氬潗鏍囧搴旀姇褰变俊鎭紝濡備负锛歂one锛岄粯璁わ細EPSG:4326
        '''
        return self._info_with(product, ctype, crs, {
            "x": x,
            "y": y,
            "times": "" if timeslice is None else timeslice
        })

    def info_by_bbox(self, product, ctype, bbox, crs=None, timeslice=None):
        return self._info_with(product, ctype, crs, {
            "bbox": ",".join(map(lambda a: str(a), bbox)),
            "times": "" if timeslice is None else timeslice
        })

    def info_by_geom(self, product, ctype, geom, crs=None, timeslice=None):
        return self._info_with(product, ctype, crs, {
            "geom": geom,
            "times": "" if timeslice is None else timeslice
        })

    def query_by_point(self, product, ctype, bandid, x, y, crs=None, timeslice=None):
        url = url_prefix + "/query_point/{product}/{ctype}".format(product=product, ctype=ctype)

        params = {"bandid": bandid, "x": x, "y": y}
        params["crs"] = "" if crs is None else crs
        params["times"] = "" if timeslice is None else timeslice
        params["format"] = "bytes"

        headers = {
            "content-type": "application/x-www-form-urlencoded",
            "auth_code": self.auth_code
        }

        #         print(params)

        resp = self.c.urlopen("POST", url, body=urlencode(params), headers=headers)
        return self._load_resp(resp)

    def query_by_geom(self, product, ctype, bandid, geom_info, timeslice=None):
        url = url_prefix + "/query_geom/{product}/{ctype}".format(product=product, ctype=ctype)

        params = {"bandid": bandid, }
        #         params[ "geom_info" ] = json.dumps(geom_info, ensure_ascii=False)
        params["mask_geom"] = geom_info["geometry"]
        params["grid_x"], params["grid_y"] = geom_info["xy"]
        params["times"] = "" if timeslice is None else timeslice
        params["format"] = "bytes"

        headers = {
            "content-type": "application/x-www-form-urlencoded",
            "auth_code": self.auth_code
        }

        #         print(params)

        resp = self.c.urlopen("POST", url, body=urlencode(params), headers=headers)
        return self._load_resp(resp)


if __name__ == '__main__':

    c = DataBoxConnector("www.gscloud.cn", 80)

    import time

    t = time.time()
    n = 1

    for _ in range(n):
        timeslice = "year in [2010, 2011]"

        info, status, code, reason = c.info_by_point("LANDSAT", "L45TM", 116.056832678626, 39.5991427612245,
                                                     "EPSG:4326", timeslice=timeslice)
        print(info, status, code, reason)

        info, status, code, reason = c.query_by_point("LANDSAT", "L45TM", "B10", 116.456832678626, 39.5991427612245,
                                                      "EPSG:4326", timeslice=timeslice)
        print(info, status, code, reason)

        info, status, code, reason = c.info_by_bbox("LANDSAT", "L45TM", [117.513, 40.013, 118.5243, 41.023, ],
                                                    "EPSG:4326", timeslice=timeslice)
        print(info, status, code, reason)

        gjson = """{ "type": "Polygon", "coordinates": [ [ [ 116.056832678625995, 39.599142761224542 ], [ 116.063241002197074, 39.599753077755125 ], [ 116.063241002197074, 39.599753077755125 ], [ 116.064156476992935, 39.594565387245204 ], [ 116.056222362095411, 39.590598329796435 ], [ 116.05225530464665, 39.589377696735284 ], [ 116.046762455871431, 39.593649912449337 ], [ 116.056832678625995, 39.599142761224542 ] ] ] }"""
        info, status, code, reason = c.info_by_geom("LANDSAT", "L45TM", gjson, "EPSG:4326", timeslice=timeslice)
        print(info, status, code, reason)

        if len(info) > 0:
            info, status, code, reason = c.query_by_geom("LANDSAT", "L45TM", "B10", info[0], timeslice=timeslice)
            print(info, status, code, reason)

    t = time.time() - t
    print(t / n)
