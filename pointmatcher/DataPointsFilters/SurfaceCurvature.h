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
		return "This filter extracts the surface curvature to each point using its nearest neighbors.\n\n"
			   "Required descriptors: normals.\n"
			   "Produced descritors:  curvatures.\n"
			   "Altered descriptors:  none.\n"
			   "Altered features:     none.";
	}
	
	inline static const ParametersDoc availableParameters()
	{
		return {
				{ "knn",     "Number of nearest neighbors to consider, including the point itself", "5",   "3", "2147483647",
																													   &Parametrizable::Comp < unsigned > },
				{ "maxDist", "Maximum distance to consider for neighbors",                          "inf", "0", "inf", &Parametrizable::Comp < T > },
				{ "epsilon", "Approximation to use for the nearest-neighbor search",                "0",   "0", "inf", &Parametrizable::Comp < T > },
		};
	}
	
	const unsigned knn;
	const T maxDist;
	const T epsilon;
	
	SurfaceCurvatureDataPointsFilter(const Parameters& params = Parameters());
	
	virtual ~SurfaceCurvatureDataPointsFilter()
	{
	};
	
	virtual DataPoints filter(const DataPoints& input);
	
	virtual void inPlaceFilter(DataPoints& cloud);
};
