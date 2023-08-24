#pragma once

#include "PointMatcher.h"
#include "DataPointsFilters/utils/utils_lie.hpp"

template<typename T>
struct DeskewingUncertaintyDataPointsFilter : public PointMatcher<T>::DataPointsFilter
{
	typedef PointMatcher<T> PM;
	typedef PointMatcherSupport::Parametrizable Parametrizable;
	typedef Parametrizable::ParametersDoc ParametersDoc;
	typedef typename PM::DataPoints DataPoints;
	typedef Parametrizable::Parameters Parameters;
	typedef typename PM::DataPoints::InvalidField InvalidField;
	typedef Parametrizable::InvalidParameter InvalidParameter;
	typedef typename PM::Matrix Matrix;
	typedef typename PM::Vector Vector;

	inline static const std::string description()
	{
		return "Adds a 9D descriptor named <covariance> that represents the covariance of each point, based on the de-skewing uncertainty.\n\n"
			   "Required descriptors: none.\n"
			   "Required times: stamps.\n"
			   "Produced descriptors:  covariance.\n"
			   "Sensor assumed to be at the origin: yes.\n"
			   "Altered descriptors:  none.\n"
			   "Altered features:     none.";
	}

	inline static const ParametersDoc availableParameters()
	{
		return {
				{"linearSpeedsX",  "Comma-separated linear speeds along the X axis during the scan",  "0"},
				{"linearSpeedsY",  "Comma-separated linear speeds along the Y axis during the scan",  "0"},
				{"linearSpeedsZ",  "Comma-separated linear speeds along the Z axis during the scan",  "0"},
				{"linearSpeedVariancesX",  "Comma-separated linear speed variances along the X axis during the scan",  "1"},
				{"linearSpeedCovariancesXY",  "Comma-separated linear speed covariances of the X and Y axes during the scan",  "0"},
				{"linearSpeedCovariancesXZ",  "Comma-separated linear speed covariances of the X and Z axes during the scan",  "0"},
                {"linearSpeedVariancesY",  "Comma-separated linear speed variances along the Y axis during the scan",  "1"},
                {"linearSpeedCovariancesYZ",  "Comma-separated linear speed covariances of the Y and Z axes during the scan",  "0"},
                {"linearSpeedVariancesZ",  "Comma-separated linear speed variances along the Z axis during the scan",  "1"},
				{"angularSpeedsX", "Comma-separated angular speeds along the X axis during the scan", "0"},
				{"angularSpeedsY", "Comma-separated angular speeds along the Y axis during the scan", "0"},
				{"angularSpeedsZ", "Comma-separated angular speeds along the Z axis during the scan", "0"},
                {"angularSpeedVariancesX",  "Comma-separated angular speed variances along the X axis during the scan",  "1"},
                {"angularSpeedCovariancesXY",  "Comma-separated angular speed variances of the X and Y axes during the scan",  "0"},
                {"angularSpeedCovariancesXZ",  "Comma-separated angular speed variances of the X and Z axes during the scan",  "0"},
                {"angularSpeedVariancesY",  "Comma-separated angular speed variances along the Y axis during the scan",  "1"},
                {"angularSpeedCovariancesYZ",  "Comma-separated angular speed variances of the Y and Z axes during the scan",  "0"},
                {"angularSpeedVariancesZ",  "Comma-separated angular speed variances along the Z axis during the scan",  "1"},
				{"measureTimes",   "Times at which inertial measurements were acquired",              "0"},
		};
	}

	DeskewingUncertaintyDataPointsFilter(const Parameters& params = Parameters());

	virtual DataPoints filter(const DataPoints& input);

	virtual void inPlaceFilter(DataPoints& value);

	std::vector<Gaussian<T>> motionGaussians;
	const std::vector<T> measureTimes;

private:
	std::vector<T> castToScalarVector(const std::string& values);
	std::vector<Vector> castToVectorVector(const std::string& xValues, const std::string& yValues, const std::string& zValues);
	std::vector<Matrix> castToMatrixVector(const std::string& xValues, const std::string& xyValues, const std::string& xzValues, const std::string& yValues,
                                           const std::string& yzValues, const std::string& zValues);
	template<typename U>
	std::vector<int> computeOrdering(const Eigen::Matrix<U, 1, Eigen::Dynamic>& elements);
	void applyOrdering(const std::vector<int>& ordering, Eigen::Matrix<int, 1, Eigen::Dynamic>& idTable, DataPoints& dataPoints);
};
