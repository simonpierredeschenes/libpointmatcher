#pragma once

#include "PointMatcher.h"

#include <vector>

//! Surface curvature estimation. Find the curvature for every point using neighbour points
template<typename T>
struct SurfaceCurvatureDataPointsFilter: public PointMatcher<T>::DataPointsFilter
{
	typedef PointMatcher<T> PM;
	typedef PointMatcherSupport::Parametrizable Parametrizable;
	typedef Parametrizable::ParametersDoc ParametersDoc;
	typedef typename PM::DataPoints DataPoints;
	typedef Parametrizable::Parameters Parameters;
	typedef typename PM::DataPoints::InvalidField InvalidField;
	typedef Parametrizable::InvalidParameter InvalidParameter;
	
	inline static const std::string description()
	{
		return "This filter extracts the surface curvature to each point.\n\n"
			   "Required descriptors: eigValues(if estimation method no. 0), normals (if estimation method no. 1).\n"
			   "Produced descritors:  curvatures.\n"
			   "Altered descriptors:  none.\n"
			   "Altered features:     none.";
	}
	
	inline static const ParametersDoc availableParameters()
	{
		return {
				{ "knn",              "Number of nearest neighbors to consider, including the point itself",                  "5",   "3", "2147483647",
																																				 &Parametrizable::Comp <
																																				 unsigned > },
				{ "maxDist",          "Maximum distance to consider for neighbors",                                           "inf", "0", "inf", &Parametrizable::Comp <
																																				 T > },
				{ "epsilon",          "Approximation to use for the nearest-neighbor search",                                 "0",   "0", "inf", &Parametrizable::Comp <
																																				 T > },
				{ "estimationMethod", "Method used for curvature estimation. 0=Eigen values ratio, 1=Normals rate of change", "0",   "0", "1",   &Parametrizable::Comp <
																																				 unsigned > },
		};
	}
	
	const unsigned knn;
	const T maxDist;
	const T epsilon;
	const unsigned estimationMethod;
	
	SurfaceCurvatureDataPointsFilter(const Parameters& params = Parameters());
	
	virtual ~SurfaceCurvatureDataPointsFilter()
	{
	};
	
	virtual DataPoints filter(const DataPoints& input);
	
	virtual void inPlaceFilter(DataPoints& cloud);
};
