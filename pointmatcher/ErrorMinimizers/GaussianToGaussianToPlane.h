#pragma once

#include "PointMatcher.h"

template<typename T>
struct GaussianToGaussianToPlaneErrorMinimizer : public PointMatcher<T>::ErrorMinimizer
{
	typedef PointMatcherSupport::Parametrizable Parametrizable;
	typedef PointMatcherSupport::Parametrizable P;
	typedef Parametrizable::Parameters Parameters;
	typedef Parametrizable::ParameterDoc ParameterDoc;
	typedef Parametrizable::ParametersDoc ParametersDoc;

	typedef typename PointMatcher<T>::DataPoints DataPoints;
	typedef typename PointMatcher<T>::Matches Matches;
	typedef typename PointMatcher<T>::OutlierWeights OutlierWeights;
	typedef typename PointMatcher<T>::ErrorMinimizer ErrorMinimizer;
	typedef typename PointMatcher<T>::ErrorMinimizer::ErrorElements ErrorElements;
	typedef typename PointMatcher<T>::TransformationParameters TransformationParameters;
	typedef typename PointMatcher<T>::Vector Vector;
	typedef typename PointMatcher<T>::Matrix Matrix;

	virtual inline const std::string name()
	{
		return "GaussianToGaussianToPlaneErrorMinimizer";
	}

	inline static const std::string description()
	{
		return "Gaussian-to-Gaussian-to-plane error.";
	}

	inline static const ParametersDoc availableParameters()
	{
		return {
				{"scaleFactor", "Minimum point variance", "0", "0", "inf", &P::Comp < T > }
		};
	}

	const T scaleFactor;

	GaussianToGaussianToPlaneErrorMinimizer(const Parameters& params = Parameters());
	GaussianToGaussianToPlaneErrorMinimizer(const ParametersDoc paramsDoc, const Parameters& params);
	virtual TransformationParameters compute(const ErrorElements& mPts);
};