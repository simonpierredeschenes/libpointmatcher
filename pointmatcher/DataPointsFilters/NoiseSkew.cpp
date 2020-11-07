#include "NoiseSkew.h"
#include <numeric>

template<typename Func>
struct lambda_as_visitor_wrapper: Func
{
	lambda_as_visitor_wrapper(const Func& f):
			Func(f)
	{
	}
	
	template<typename S, typename I>
	void init(const S& v, I i, I j)
	{
		return Func::operator()(v, i, j);
	}
};

template<typename Mat, typename Func>
void visit_lambda(const Mat& m, const Func& f)
{
	lambda_as_visitor_wrapper<Func> visitor(f);
	m.visit(visitor);
}

// https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
template<typename T>
template<typename U>
std::vector<int> NoiseSkewDataPointsFilter<T>::computeOrdering(const Eigen::Matrix<U, 1, Eigen::Dynamic>& elements)
{
	std::vector<int> indices(elements.cols());
	std::iota(indices.begin(), indices.end(), 0);
	std::stable_sort(indices.begin(), indices.end(), [&elements](int index1, int index2){ return elements(0, index1) < elements(0, index2); });
	return indices;
}

template<typename T>
void NoiseSkewDataPointsFilter<T>::applyOrdering(const std::vector<int>& ordering, Eigen::Array<int, 1, Eigen::Dynamic>& idTable, DataPoints& dataPoints)
{
	std::vector<int> newIndices(ordering.size());
	std::iota(newIndices.begin(), newIndices.end(), 0);
	for(int i = 0; i < ordering.size(); i++)
	{
		int indexToSwap = newIndices[ordering[i]];
		idTable.col(i).swap(idTable.col(indexToSwap));
		dataPoints.swapCols(i, indexToSwap);
		int tempIndex = newIndices[i];
		newIndices[newIndices[i]] = indexToSwap;
		newIndices[indexToSwap] = tempIndex;
	}
}

template<typename T>
NoiseSkewDataPointsFilter<T>::NoiseSkewDataPointsFilter(const Parameters& params):
		PointMatcher<T>::DataPointsFilter("NoiseSkewDataPointsFilter", NoiseSkewDataPointsFilter::availableParameters(), params),
		skewModel(Parametrizable::get<unsigned>("skewModel")),
		linearSpeedNoise(Parametrizable::get<T>("linearSpeedNoise")),
		linearAccelerationNoise(Parametrizable::get<T>("linearAccelerationNoise")),
		angularSpeedNoise(Parametrizable::get<T>("angularSpeedNoise")),
		angularAccelerationNoise(Parametrizable::get<T>("angularAccelerationNoise")),
		cornerPointWeight(Parametrizable::get<T>("cornerPointWeight")),
		weightQuantile(Parametrizable::get<T>("weightQuantile"))
{
}

template<typename T>
typename PointMatcher<T>::DataPoints NoiseSkewDataPointsFilter<T>::filter(const DataPoints& input)
{
	DataPoints output(input);
	inPlaceFilter(output);
	return output;
}

#include <iostream>

template<typename T>
void NoiseSkewDataPointsFilter<T>::inPlaceFilter(DataPoints& cloud)
{
	if(!cloud.descriptorExists("normals"))
	{
		throw InvalidField("NoiseSkewDataPointsFilter: Error, cannot find normals in descriptors.");
	}
	if(!cloud.descriptorExists("observationDirections"))
	{
		throw InvalidField("NoiseSkewDataPointsFilter: Error, cannot find observation directions in descriptors.");
	}
	if(!cloud.timeExists("stamps"))
	{
		throw InvalidField("NoiseSkewDataPointsFilter: Error, cannot find stamps in times.");
	}
	
	// TODO: remove observationDirections dependency
	
	std::vector<int> ringIds;
	std::map<int, Eigen::Array<int, 1, Eigen::Dynamic>> idTables;
	std::map<int, typename PM::DataPoints> ringDataPoints;
	if(cloud.getEuclideanDim() == 2)
	{
		ringIds.push_back(0);
		idTables[0] = Eigen::Array<int, 1, Eigen::Dynamic>::LinSpaced(cloud.getNbPoints(), 0, cloud.getNbPoints() - 1);
		ringDataPoints[0] = cloud;
	}
	else
	{
		if(!cloud.descriptorExists("ring"))
		{
			throw InvalidField("NoiseSkewDataPointsFilter: Error, cannot find ring in descriptors.");
		}
		
		std::map<int, int> pointCounts;
		const auto& rings = cloud.getDescriptorViewByName("ring");
		for(int i = 0; i < cloud.getNbPoints(); i++)
		{
			int ringId = rings(0, i);
			if(idTables[ringId].cols() == 0)
			{
				ringIds.push_back(ringId);
				idTables[ringId] = Eigen::Array<int, 1, Eigen::Dynamic>::Zero(1, cloud.getNbPoints());
				ringDataPoints[ringId] = typename PM::DataPoints(cloud.featureLabels, cloud.descriptorLabels, cloud.timeLabels, cloud.getNbPoints());
			}
			idTables[ringId](0, pointCounts[ringId]) = i;
			ringDataPoints[ringId].setColFrom(pointCounts[ringId], cloud, i);
			pointCounts[ringId]++;
		}
		for(const int& ringId : ringIds)
		{
			idTables[ringId].conservativeResize(1, pointCounts[ringId]);
			ringDataPoints[ringId].conservativeResize(pointCounts[ringId]);
		}
	}
	
	for(const int& ringId: ringIds) // TODO: add this to main loop
	{
		const auto& stamps = ringDataPoints[ringId].getTimeViewByName("stamps");
		std::vector<int> ordering = computeOrdering<std::int64_t>(stamps);
		applyOrdering(ordering, idTables[ringId], ringDataPoints[ringId]);
	}
	
	Array weights = Array::Zero(1, cloud.getNbPoints());
	for(const int& ringId: ringIds)
	{
		Eigen::Array<int, 1, Eigen::Dynamic>& idTable = idTables[ringId];
		typename PM::DataPoints& ring = ringDataPoints[ringId];
		
		const auto& normals = ring.getDescriptorViewByName("normals");
		const auto& observationDirections = ring.getDescriptorViewByName("observationDirections");
		const auto& stamps = ring.getTimeViewByName("stamps");
		
		Array laserDirections = (-observationDirections.array()).rowwise() / observationDirections.array().pow(2).colwise().sum().sqrt();
		Array firingDelays = (stamps.colwise() - stamps.col(0)).template cast<T>() / 1e9;
		Array points = ring.features.topRows(ring.getEuclideanDim());
		
		Array linearVelocities = Array::Zero(points.rows(), points.cols());
		linearVelocities.row(0) = Array::Constant(1, points.cols(), linearSpeedNoise);
		
		Array linearAccelerations = Array::Constant(points.rows(), points.cols(), linearAccelerationNoise);
		Array backwardRightPositions = (-linearVelocities).rowwise() * firingDelays.row(0) - linearAccelerations.rowwise() * (0.5 * firingDelays.pow(2)).row(0);
		Array forwardRightPositions = linearVelocities.rowwise() * firingDelays.row(0) + linearAccelerations.rowwise() * (0.5 * firingDelays.pow(2)).row(0);
		
		Array backwardRightLaserDirections = Array::Zero(points.rows(), points.cols());
		Array forwardRightLaserDirections = Array::Zero(points.rows(), points.cols());
		Array backwardRotatedPoints = Array::Zero(points.rows(), points.cols());
		Array forwardRotatedPoints = Array::Zero(points.rows(), points.cols());
		if(ring.getEuclideanDim() == 2)
		{
			Array angularVelocities = Array::Constant(1, points.cols(), angularSpeedNoise);
			Array angularAccelerations = Array::Constant(1, points.cols(), angularAccelerationNoise);
			Array rightOrientations = angularVelocities * firingDelays + angularAccelerations * 0.5 * firingDelays.pow(2);
			Array rotationMatrix11 = rightOrientations.cos();
			Array rotationMatrix12 = -rightOrientations.sin();
			Array rotationMatrix21 = rightOrientations.sin();
			Array rotationMatrix22 = rightOrientations.cos();
			backwardRightLaserDirections.row(0) = rotationMatrix11 * laserDirections.row(0) + rotationMatrix21 * laserDirections.row(1);
			backwardRightLaserDirections.row(1) = rotationMatrix12 * laserDirections.row(0) + rotationMatrix22 * laserDirections.row(1);
			forwardRightLaserDirections.row(0) = rotationMatrix11 * laserDirections.row(0) + rotationMatrix12 * laserDirections.row(1);
			forwardRightLaserDirections.row(1) = rotationMatrix21 * laserDirections.row(0) + rotationMatrix22 * laserDirections.row(1);
			backwardRotatedPoints.row(0) = rotationMatrix11 * points.row(0) + rotationMatrix21 * points.row(1);
			backwardRotatedPoints.row(1) = rotationMatrix12 * points.row(0) + rotationMatrix22 * points.row(1);
			forwardRotatedPoints.row(0) = rotationMatrix11 * points.row(0) + rotationMatrix12 * points.row(1);
			forwardRotatedPoints.row(1) = rotationMatrix21 * points.row(0) + rotationMatrix22 * points.row(1);
		}
		else
		{
			Array angularVelocities = Array::Constant(3, points.cols(), angularSpeedNoise); // TODO only speed in Z
			Array angularAccelerations = Array::Constant(3, points.cols(), angularAccelerationNoise);
			Array rightOrientations = angularVelocities.rowwise() * firingDelays.row(0) + angularAccelerations.rowwise() * (0.5 * firingDelays.pow(2)).row(0);
			// https://math.stackexchange.com/questions/1874898/simultaneous-action-of-two-quaternions
			Array angles = rightOrientations.pow(2).colwise().sum().sqrt();
			Array axes = rightOrientations.rowwise() / angles.row(0);
			axes.row(0) = axes.row(0).unaryExpr([](T value){ return std::isfinite(value) ? value : T(1); });
			axes.row(1) = axes.row(1).unaryExpr([](T value){ return std::isfinite(value) ? value : T(0); });
			axes.row(2) = axes.row(2).unaryExpr([](T value){ return std::isfinite(value) ? value : T(0); });
			Array axesCrossLaserDirections = Array::Zero(axes.rows(), axes.cols());
			axesCrossLaserDirections.row(0) = axes.row(1) * laserDirections.row(2) - axes.row(2) * laserDirections.row(1);
			axesCrossLaserDirections.row(1) = axes.row(2) * laserDirections.row(0) - axes.row(0) * laserDirections.row(2);
			axesCrossLaserDirections.row(2) = axes.row(0) * laserDirections.row(1) - axes.row(1) * laserDirections.row(0);
			Array axesAxesTLaserDirections = Array::Zero(axes.rows(), axes.cols());
			axesAxesTLaserDirections.row(0) = axes.row(0) * axes.row(0) * laserDirections.row(0) +
											  axes.row(0) * axes.row(1) * laserDirections.row(1) +
											  axes.row(0) * axes.row(2) * laserDirections.row(2);
			axesAxesTLaserDirections.row(1) = axes.row(1) * axes.row(0) * laserDirections.row(0) +
											  axes.row(1) * axes.row(1) * laserDirections.row(1) +
											  axes.row(1) * axes.row(2) * laserDirections.row(2);
			axesAxesTLaserDirections.row(2) = axes.row(2) * axes.row(0) * laserDirections.row(0) +
											  axes.row(2) * axes.row(1) * laserDirections.row(1) +
											  axes.row(2) * axes.row(2) * laserDirections.row(2);
			backwardRightLaserDirections = laserDirections.rowwise() * (-(angles.row(0))).cos() + axesCrossLaserDirections.rowwise() * (-(angles.row(0))).sin()
										   + axesAxesTLaserDirections.rowwise() * (1.0 - (-(angles.row(0))).cos());
			forwardRightLaserDirections = laserDirections.rowwise() * angles.row(0).cos() + axesCrossLaserDirections.rowwise() * angles.row(0).sin()
										  + axesAxesTLaserDirections.rowwise() * (1.0 - angles.row(0).cos());
			Array axesCrossPoints = Array::Zero(axes.rows(), axes.cols());
			axesCrossPoints.row(0) = axes.row(1) * points.row(2) - axes.row(2) * points.row(1);
			axesCrossPoints.row(1) = axes.row(2) * points.row(0) - axes.row(0) * points.row(2);
			axesCrossPoints.row(2) = axes.row(0) * points.row(1) - axes.row(1) * points.row(0);
			Array axesAxesTPoints = Array::Zero(axes.rows(), axes.cols());
			axesAxesTPoints.row(0) = axes.row(0) * axes.row(0) * points.row(0) +
									 axes.row(0) * axes.row(1) * points.row(1) +
									 axes.row(0) * axes.row(2) * points.row(2);
			axesAxesTPoints.row(1) = axes.row(1) * axes.row(0) * points.row(0) +
									 axes.row(1) * axes.row(1) * points.row(1) +
									 axes.row(1) * axes.row(2) * points.row(2);
			axesAxesTPoints.row(2) = axes.row(2) * axes.row(0) * points.row(0) +
									 axes.row(2) * axes.row(1) * points.row(1) +
									 axes.row(2) * axes.row(2) * points.row(2);
			backwardRotatedPoints = points.rowwise() * (-(angles.row(0))).cos() + axesCrossPoints.rowwise() * (-(angles.row(0))).sin()
									+ axesAxesTPoints.rowwise() * (1.0 - (-(angles.row(0))).cos());
			forwardRotatedPoints = points.rowwise() * angles.row(0).cos() + axesCrossPoints.rowwise() * angles.row(0).sin()
								   + axesAxesTPoints.rowwise() * (1.0 - angles.row(0).cos());
		}
		
		Array backwardRightDistances = ((points - backwardRightPositions) * normals.array()).colwise().sum() / //TODO: need to check backwardRightPositions with
									   (backwardRightLaserDirections * normals.array()).colwise().sum();      //TODO: forwardRightLaserDirections and vice-versa
		Array forwardRightDistances = ((points - forwardRightPositions) * normals.array()).colwise().sum() /
									  (forwardRightLaserDirections * normals.array()).colwise().sum();
		Array backwardCorrectedPoints = backwardRotatedPoints - backwardRightPositions;
		Array forwardCorrectedPoints = forwardRotatedPoints - forwardRightPositions;
		Array backwardCorrectedPointDirections = backwardCorrectedPoints.rowwise() / backwardCorrectedPoints.pow(2).colwise().sum().sqrt();
		Array forwardCorrectedPointDirections = forwardCorrectedPoints.rowwise() / forwardCorrectedPoints.pow(2).colwise().sum().sqrt();
		
		Array estimatedErrors = (forwardRightDistances - backwardRightDistances).abs();
		Array ringWeights = 1.0 / (estimatedErrors.pow(2) + 0.01);
		
		Array cornerness = Array::Zero(estimatedErrors.rows(), estimatedErrors.cols());
		cornerness.block(0, 1, 1, cornerness.cols() - 1) = (estimatedErrors.block(0, 1, 1, estimatedErrors.cols() - 1) -
															estimatedErrors.block(0, 0, 1, estimatedErrors.cols() - 1)).abs();
		Array sortedCornerness(cornerness);
		std::sort(sortedCornerness.data(), sortedCornerness.data() + sortedCornerness.size());
		T lowerQuartile = sortedCornerness(0, sortedCornerness.cols() / 4);
		T upperQuartile = sortedCornerness(0, 3 * sortedCornerness.cols() / 4);
		T IQR = upperQuartile - lowerQuartile;
		T threshold = upperQuartile + (15 * IQR);
		std::vector<int> cornerIds;
		visit_lambda(cornerness, [&cornerIds, threshold](double value, int i, int j){
			if(value >= threshold)
			{
				cornerIds.push_back(j);
			}
		});
		
		Array pointHorizontalAngles;
		Array backwardCorrectedPointHorizontalAngles;
		Array forwardCorrectedPointHorizontalAngles;
		if(ring.getEuclideanDim() == 2)
		{
			pointHorizontalAngles = laserDirections.row(1)
					.binaryExpr(laserDirections.row(0), [](double a, double b){ return std::atan2(a, b); }).template cast<T>();
			backwardCorrectedPointHorizontalAngles = backwardCorrectedPointDirections.row(1)
					.binaryExpr(backwardCorrectedPointDirections.row(0), [](double a, double b){ return std::atan2(a, b); }).template cast<T>();
			forwardCorrectedPointHorizontalAngles = forwardCorrectedPointDirections.row(1)
					.binaryExpr(forwardCorrectedPointDirections.row(0), [](double a, double b){ return std::atan2(a, b); }).template cast<T>();
		}
		else
		{
			Array pointVerticalAngles = laserDirections.row(2)
					.unaryExpr([](double a){ return std::asin(a); }).template cast<T>();
			pointHorizontalAngles = (laserDirections.row(1) / pointVerticalAngles.cos())
					.binaryExpr(laserDirections.row(0) / pointVerticalAngles.cos(),
								[](double a, double b){ return std::atan2(a, b); }).template cast<T>();
			Array backwardCorrectedPointVerticalAngles = backwardCorrectedPointDirections.row(2)
					.unaryExpr([](double a){ return std::asin(a); }).template cast<T>();
			backwardCorrectedPointHorizontalAngles = (backwardCorrectedPointDirections.row(1) / backwardCorrectedPointVerticalAngles.cos())
					.binaryExpr(backwardCorrectedPointDirections.row(0) / backwardCorrectedPointVerticalAngles.cos(),
								[](double a, double b){ return std::atan2(a, b); }).template cast<T>();
			Array forwardCorrectedPointVerticalAngles = forwardCorrectedPointDirections.row(2)
					.unaryExpr([](double a){ return std::asin(a); }).template cast<T>();
			forwardCorrectedPointHorizontalAngles = (forwardCorrectedPointDirections.row(1) / forwardCorrectedPointVerticalAngles.cos())
					.binaryExpr(forwardCorrectedPointDirections.row(0) / forwardCorrectedPointVerticalAngles.cos(),
								[](double a, double b){ return std::atan2(a, b); }).template cast<T>();
		}
		
		Array cornerWeights = Array::Ones(1, points.cols());
		for(const auto& cornerId: cornerIds)
		{
			T lowerHorizontalAngle = std::min(backwardCorrectedPointHorizontalAngles(0, cornerId), forwardCorrectedPointHorizontalAngles(0, cornerId));
			T higherHorizontalAngle = std::max(backwardCorrectedPointHorizontalAngles(0, cornerId), forwardCorrectedPointHorizontalAngles(0, cornerId));
			visit_lambda(pointHorizontalAngles, [this, &lowerHorizontalAngle, &higherHorizontalAngle, &cornerWeights](double value, int i, int j){
				if(value >= lowerHorizontalAngle && value <= higherHorizontalAngle)
				{
					cornerWeights(i, j) = cornerPointWeight;
				}
			});
		}
		ringWeights *= cornerWeights;
		
		std::vector<int> ordering = computeOrdering<T>(ringWeights);
		int startIndex = weightQuantile * (ordering.size() - 1);
		for(int i = startIndex; i < ordering.size(); i++)
		{
			weights(0, idTable(0, ordering[i])) = ringWeights(0, ordering[i]);
		}
	}
	cloud.addDescriptor("skewWeight", weights);
}

template struct NoiseSkewDataPointsFilter<float>;
template struct NoiseSkewDataPointsFilter<double>;
