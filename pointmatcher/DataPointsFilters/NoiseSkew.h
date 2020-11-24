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
				{ "skewModel",                 "Skew model used for weighting. Choices: 0=Model based on time only, 1=Model based on speed and acceleration noises, 2=Model based on speed and acceleration noises and on incidence angle, 3=Model based on \\cite{Al-Nuaimi2016}",
																											"0",    "0",    "3",
																																   &Parametrizable::Comp <
																																   unsigned > },
				{ "rangePrecision",            "Precision of range measurements",                           "0.02", "-inf", "inf", &Parametrizable::Comp <
																																   T > },
				{ "linearSpeedNoiseX",         "Noise on linear speed along the X axis",                    "0",    "-inf", "inf", &Parametrizable::Comp <
																																   T > },
				{ "linearSpeedNoiseY",         "Noise on linear speed along the Y axis",                    "0",    "-inf", "inf", &Parametrizable::Comp <
																																   T > },
				{ "linearSpeedNoiseZ",         "Noise on linear speed along the Z axis",                    "0",    "-inf", "inf", &Parametrizable::Comp <
																																   T > },
				{ "linearAccelerationNoiseX",  "Noise on linear acceleration along the X axis",             "0",    "-inf", "inf", &Parametrizable::Comp <
																																   T > },
				{ "linearAccelerationNoiseY",  "Noise on linear acceleration along the Y axis",             "0",    "-inf", "inf", &Parametrizable::Comp <
																																   T > },
				{ "linearAccelerationNoiseZ",  "Noise on linear acceleration along the Z axis",             "0",    "-inf", "inf", &Parametrizable::Comp <
																																   T > },
				{ "angularSpeedNoiseX",        "Noise on angular speed along the X axis",                   "0",    "-inf", "inf", &Parametrizable::Comp <
																																   T > },
				{ "angularSpeedNoiseY",        "Noise on angular speed along the Y axis",                   "0",    "-inf", "inf", &Parametrizable::Comp <
																																   T > },
				{ "angularSpeedNoiseZ",        "Noise on angular speed along the Z axis",                   "0",    "-inf", "inf", &Parametrizable::Comp <
																																   T > },
				{ "angularAccelerationNoiseX", "Noise on angular acceleration along the X axis",            "0",    "-inf", "inf", &Parametrizable::Comp <
																																   T > },
				{ "angularAccelerationNoiseY", "Noise on angular acceleration along the Y axis",            "0",    "-inf", "inf", &Parametrizable::Comp <
																																   T > },
				{ "angularAccelerationNoiseZ", "Noise on angular acceleration along the Z axis",            "0",    "-inf", "inf", &Parametrizable::Comp <
																																   T > },
				{ "cornerPointWeight",         "Weight to give to points at junction of multiple surfaces", "1",    "-inf", "inf", &Parametrizable::Comp <
																																   T > },
				{ "weightQuantile",            "Quantile under which weights are set to 0",                 "0",    "-inf", "inf", &Parametrizable::Comp <
																																   T > },
		};
	}
	
	NoiseSkewDataPointsFilter(const Parameters& params = Parameters());
	
	virtual DataPoints filter(const DataPoints& input);
	
	virtual void inPlaceFilter(DataPoints& value);
	
	const unsigned skewModel;
	const T rangePrecision;
	const T linearSpeedNoiseX;
	const T linearSpeedNoiseY;
	const T linearSpeedNoiseZ;
	const T linearAccelerationNoiseX;
	const T linearAccelerationNoiseY;
	const T linearAccelerationNoiseZ;
	const T angularSpeedNoiseX;
	const T angularSpeedNoiseY;
	const T angularSpeedNoiseZ;
	const T angularAccelerationNoiseX;
	const T angularAccelerationNoiseY;
	const T angularAccelerationNoiseZ;
	const T cornerPointWeight;
	const T weightQuantile;

private:
	template<typename U>
	std::vector<int> computeOrdering(const Eigen::Matrix<U, 1, Eigen::Dynamic>& elements);
	
	void applyOrdering(const std::vector<int>& ordering, Eigen::Array<int, 1, Eigen::Dynamic>& idTable, DataPoints& dataPoints);
	
	const T REFERENCE_CURVATURE = 40.0;
};
