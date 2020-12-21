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
		return "Adds a 1D descriptor named <skewUncertainty> that represents the uncertainty of each point, based on the skew caused by noise on speed and acceleration.\n\n"
			   "Required descriptors: normals (for skew model no. 2) curvatures (for skew models no. 3 and 5), rings (for skew model no. 2 with 3D point clouds).\n"
			   "Required times: stamps (for skew models no. 0, 1, 2, 3 and 4).\n"
			   "Produced descriptors:  skewUncertainty.\n"
			   "Sensor assumed to be at the origin: yes.\n"
			   "Altered descriptors:  none.\n"
			   "Altered features:     none.";
	}

	inline static const ParametersDoc availableParameters()
	{
		return {
				{ "skewModel",                  "Skew model used for computing uncertainty. Choices: 0=Model based on time only, 1=Model based on speed and acceleration noises, 2=Model based on speed and acceleration noises and on incidence angle, 3=Model based on \\cite{Al-Nuaimi2016}, 4=Model based on scanning angle, 5=Model based on curvature", "0",    "0",    "5", &Parametrizable::Comp < unsigned > },
				{ "linearSpeedsX",              "Comma-separated linear speeds along the X axis during the scan",         "0" },
				{ "linearSpeedsY",              "Comma-separated linear speeds along the Y axis during the scan",         "0" },
				{ "linearSpeedsZ",              "Comma-separated linear speeds along the Z axis during the scan",         "0" },
				{ "linearAccelerationsX",       "Comma-separated linear accelerations along the X axis during the scan",  "0" },
				{ "linearAccelerationsY",       "Comma-separated linear accelerations along the Y axis during the scan",  "0" },
				{ "linearAccelerationsZ",       "Comma-separated linear accelerations along the Z axis during the scan",  "0" },
				{ "angularSpeedsX",             "Comma-separated angular speeds along the X axis during the scan",        "0" },
				{ "angularSpeedsY",             "Comma-separated angular speeds along the Y axis during the scan",        "0" },
				{ "angularSpeedsZ",             "Comma-separated angular speeds along the Z axis during the scan",        "0" },
				{ "angularAccelerationsX",      "Comma-separated angular accelerations along the X axis during the scan", "0" },
				{ "angularAccelerationsY",      "Comma-separated angular accelerations along the Y axis during the scan", "0" },
				{ "angularAccelerationsZ",      "Comma-separated angular accelerations along the Z axis during the scan", "0" },
				{ "measureTimes",               "Times at which inertial measurements were acquired",                     "0" },
				{ "cornerPointUncertainty",     "Uncertainty to add to points at junction of multiple surfaces",          "0",    "-inf", "inf", &Parametrizable::Comp < T > },
				{ "uncertaintyThreshold",       "Threshold of uncertainty over which uncertainty is set to infinity",     "1000", "-inf", "inf", &Parametrizable::Comp < T > },
				{ "uncertaintyQuantile",        "Quantile of uncertainty over which uncertainty is set to infinity",      "1",    "-inf", "inf", &Parametrizable::Comp < T > },
				{ "afterDeskewing",             "1 if this filter is applied after point cloud de-skewing, 0 otherwise.", "1", 	  "0",	  "1",	 &Parametrizable::Comp< bool >},
		};
	}

	NoiseSkewDataPointsFilter(const Parameters& params = Parameters());

	virtual DataPoints filter(const DataPoints& input);

	virtual void inPlaceFilter(DataPoints& value);

	const unsigned skewModel;
	const Array linearSpeedNoisesX;
	const Array linearSpeedNoisesY;
	const Array linearSpeedNoisesZ;
	const Array linearAccelerationNoisesX;
	const Array linearAccelerationNoisesY;
	const Array linearAccelerationNoisesZ;
	const Array angularSpeedNoisesX;
	const Array angularSpeedNoisesY;
	const Array angularSpeedNoisesZ;
	const Array angularAccelerationNoisesX;
	const Array angularAccelerationNoisesY;
	const Array angularAccelerationNoisesZ;
	const Array measureTimes;
	const T cornerPointUncertainty;
	const T uncertaintyThreshold;
	const T uncertaintyQuantile;

private:
	Array castToLinearSpeedNoises(const std::string& values, bool afterDeskewing);
	Array castToLinearAccelerationNoises(const std::string& values, bool afterDeskewing);
	Array castToAngularSpeedNoises(const std::string& values, bool afterDeskewing);
	Array castToAngularAccelerationNoises(const std::string& values, bool afterDeskewing);
	Array castToArray(const std::string& values);

	template<typename U>
	std::vector<int> computeOrdering(const Eigen::Matrix<U, 1, Eigen::Dynamic>& elements);

	void applyOrdering(const std::vector<int>& ordering, Eigen::Array<int, 1, Eigen::Dynamic>& idTable, DataPoints& dataPoints);

	Array computeTranslations(const Array& linearSpeeds, const Array& linearAccelerations, const Array& times, const Array& firingDelays);

	Array computeRotations(const Array& angularSpeeds, const Array& angularAccelerations, const Array& times, const Array& firingDelays);

	const T REFERENCE_CURVATURE = 40.0;
};
