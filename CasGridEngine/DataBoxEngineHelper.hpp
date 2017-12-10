#ifndef _ENGINEHELPER_HPP_
#define _ENGINEHELPER_HPP_

#include <string>
#include <gdal/gdal.h>

#include <gdal/ogr_api.h>
#include <gdal/cpl_conv.h>

#include <gdal/ogr_spatialref.h>

#include <math.h>
#include <sstream>

namespace GDAL {

std::string _bbox_polygon(float minx, float miny, float maxx, float maxy) {
	std::stringstream ss;
	ss << "POLYGON((";
	ss << minx << " " << miny << ",";
	ss << minx << " " << maxy << ",";
	ss << maxx << " " << maxy << ",";
	ss << maxx << " " << miny << ",";
	ss << minx << " " << miny;
	ss << "))";
	return ss.str();
}

std::string _crs_to_proj4(const std::string & crs) {

	OGRSpatialReference osr(NULL);
	osr.SetFromUserInput(crs.c_str());

	char *ptr = NULL;
	if (osr.exportToProj4(&ptr) == OGRERR_NONE) {
		if (ptr) {
			std::string proj4 = ptr;
			CPLFree(ptr);
			return proj4;
		}
	}
	return "";
}

//def crs_to_proj4(crs ):
//    sr = osr.SpatialReference()
//    sr.SetFromUserInput(crs)
//    return sr.ExportToProj4()

}
;

#endif
