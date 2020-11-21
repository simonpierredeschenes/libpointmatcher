#pragma once

#include "PointMatcher.h"

template<typename T>
struct NoiseSkewDataPointsFilter: public PointMatcher<T>::DataPointsFilter
{
	typedef PointMatcher<T> PM;
	typedef PointMatcherSupport::Parametrizable Parametrizable;
	typedef Parametrizable::ParametersDoc ParametersDoc;
	typedef typename PM::DataPoints DataPoints;
	typedef Parametrizable::Parameters Parameters;
	typedef typename PM::DataPoints::InvalidField InvalidField;
	typedef Parametrizable::InvalidParameter InvalidParameter;
	typedef Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> Array;
	
	inline static const std::string description()
	{
		return "Adds a 1D descriptor named <skewWeight> that represents the weight of each point in the minimization process, based on the skew caused by noise on speed and acceleration.\n\n"
			   "Required descriptors: simpleSensorNoise (for skew model no. 1), normals (for skew model no. 2) curvatures (for skew model no. 3), rings (for 3D point clouds).\n"
			   "Required times: stamps.\n"
			   "Produced descriptors:  skewWeight.\n"
			   "Sensor assumed to be at the origin: yes.\n"
			   "Altered descriptors:  none.\n"
			   "Altered features:     none.";
	}
	
	inline static const ParametersDoc availableParameters()
	{
		return {
				{ "skewModel",                "Skew model used for weighting. Choices: 0=Model based on time only, 1=Model based on speed and acceleration noises, 2=Model based on speed and acceleration noises and on incidence angle, 3=Model based on \\cite{Al-Nuaimi2016}",
																										   "0",    "0",    "3",
																																  &Parametrizable::Comp <
																																  unsigned > },
				{ "rangePrecision",           "Precision of range measurements",                           "0.02", "-inf", "inf", &Parametrizable::Comp < T > },
				{ "linearSpeedNoise",         "Noise on linear speed",                                     "0",    "-inf", "inf", &Parametrizable::Comp < T > },
				{ "linearAccelerationNoise",  "Noise on linear acceleration",                              "0",    "-inf", "inf", &Parametrizable::Comp < T > },
				{ "angularSpeedNoise",        "Noise on angular speed",                                    "0",    "-inf", "inf", &Parametrizable::Comp < T > },
				{ "angularAccelerationNoise", "Noise on angular acceleration",                             "0",    "-inf", "inf", &Parametrizable::Comp < T > },
				{ "cornerPointWeight",        "Weight to give to points at junction of multiple surfaces", "1",    "-inf", "inf", &Parametrizable::Comp < T > },
				{ "weightQuantile",           "Quantile under which weights are set to 0",                 "0",    "-inf", "inf", &Parametrizable::Comp < T > },
		};
	}
	
	NoiseSkewDataPointsFilter(const Parameters& params = Parameters());
	
	virtual DataPoints filter(const DataPoints& input);
	
	virtual void inPlaceFilter(DataPoints& value);
	
	const unsigned skewModel;
	const T rangePrecision;
	const T linearSpeedNoise;
	const T linearAccelerationNoise;
	const T angularSpeedNoise;
	const T angularAccelerationNoise;
	const T cornerPointWeight;
	const T weightQuantile;

private:
	template<typename U>
	std::vector<int> computeOrdering(const Eigen::Matrix<U, 1, Eigen::Dynamic>& elements);
	
	void applyOrdering(const std::vector<int>& ordering, Eigen::Array<int, 1, Eigen::Dynamic>& idTable, DataPoints& dataPoints);
	
	const T REFERENCE_CURVATURE = 40.0;
};
