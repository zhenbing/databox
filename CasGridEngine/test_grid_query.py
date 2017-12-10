import os
import yaml, time
import json

from databox.query import DataBoxQuery
from databox.timeslice import TimeSlice

if __name__ == "__main__":
    #     a = test_lru("Test", 1)
    #     print(a)

    cubefile = os.path.join(os.path.dirname(__file__), "databox/databox.yaml")
    with open(cubefile, "r") as f:
        TypeMaps = yaml.load(f)

    typeMap = TypeMaps["LANDSAT"]

    root = typeMap["ROOT"]
    gsize = typeMap["GSIZE"]

    cgq = DataBoxQuery(root, gsize)

    gjson = 'POLYGON((117.689 40.092, 118.595 40.170, 118.814 40.639, 117.502 40.65, 117.689 40.092))';
    gjson = """
    { "type": "Polygon", "coordinates": [ [ [ 114.921106371870636, 40.330037646000882 ], [ 115.628263571851761, 40.637314670664011 ], [ 117.033646534871295, 40.678691351090215 ], [ 118.986860179743331, 40.303170649319824 ], [ 118.851241451523563, 39.531519336491897 ], [ 118.256578878228808, 40.2137058181995 ], [ 117.719190166695881, 40.054484439751114 ], [ 117.216343313811109, 40.284423081962139 ], [ 116.507851176878944, 40.520972649510263 ], [ 115.387540328211969, 40.280510433742386 ], [ 114.746055569050327, 40.184335271789045 ], [ 114.782271571552585, 40.590932826305718 ], [ 114.921106371870636, 40.330037646000882 ] ] ] } 
    """
    gjson = """
    { "type": "Polygon", "coordinates": [ [ [ 116.121267604398, 39.940528822531668 ], [ 116.127675927969079, 39.941139139062251 ], [ 116.127675927969079, 39.941139139062251 ], [ 116.12859140276494, 39.935951448552331 ], [ 116.120657287867417, 39.931984391103562 ], [ 116.116690230418655, 39.93076375804241 ], [ 116.111197381643436, 39.935035973756463 ], [ 116.121267604398, 39.940528822531668 ] ] ] } 
    """

    gjson = """
    { "type": "Polygon", "coordinates": [ [ [ 116.056832678625995, 39.599142761224542 ], [ 116.063241002197074, 39.599753077755125 ], [ 116.063241002197074, 39.599753077755125 ], [ 116.064156476992935, 39.594565387245204 ], [ 116.056222362095411, 39.590598329796435 ], [ 116.05225530464665, 39.589377696735284 ], [ 116.046762455871431, 39.593649912449337 ], [ 116.056832678625995, 39.599142761224542 ] ] ] }
    """

    t = time.time()

    ctimes = TimeSlice()

    #     for _ in range(100):

    cinfo = cgq.info_at_point("L45TM", 116.056832678626, 39.5991427612245)
    print(cinfo)

    cinfo = cgq.info_at_bbox("L45TM", 117.513, 40.013, 118.5243, 41.023)
    print(len(cinfo), cinfo)

    #     cinfo = cgq.info_at_geom("L45TM", gjson)
    #     print(len(cinfo) , cinfo)

    #     cgq.query_by_point("L45TM", "B10", 117.51, 40.01)

    #     [ ['2010-08-08', '2010-09-25'] , "or" , [ [ ">=", '2010-09-01 12:00:00' ] , "or" , [ "==", "2005-09-01" ] ] ]

    #     time_range = [ ["date", "in", ['2010-08-08', '2010-09-25']] , "or" , ["not", [ ["date", ">=", '2010-09-01 12:00:00' ] , "or" , ["month", "==", 9 ] ] ] ]
    #     time_range = [ ["month" , "==", 8 ] , "or" , ["month", "==" , 5 ] , "or", [ "year", "==", "2011" ] ]

    #     ctimes = TimeSlice(time_range)
    #     cvalue = cgq.query_by_point("L45TM", "B10", 116.056832678626, 39.5991427612245, times=ctimes)
    #     print(cvalue)

    cvalue = cgq.query_by_geom("L45TM", "B10", cinfo[0], ctimes)
    svalue = json.dumps(cvalue)
    print(len(svalue))
    #
    ctimes = TimeSlice(timeval='date in ["2010-09-25 00:00:00", "2011-07-10 00:00:00"]')
    cvalue = cgq.query_by_point("L45TM", "B10", 118.0023326, 40.8635083, times=ctimes)
    print(cvalue)

    #     ctimes = TimeSlice(timemin=[ '2011-07-10 00:00:00', trange ])
    cvalue = cgq.query_by_point("L45TM", "B10", 117.3323, 40.6164, times=ctimes)
    print(cvalue)
    #
    #     ctimes = TimeSlice(timeval=['2010-09-25 00:00:00', '2011-07-10 00:00:00'])
    #     cvalue = cgq.query_by_point("L45TM", "B10", 117.3323, 40.6164, times=ctimes)
    #     print(cvalue)

    t = time.time() - t
    print(t / 100)

    # python 0.0032956790924072265 0.0041442131996154786
