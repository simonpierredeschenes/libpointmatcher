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
void NoiseSkewDataPointsFilter<T>::applyOrdering(const std::vector<int>& ordering, Eigen::Array<int, 1, Eigen::Dynamic>& idTable,
												 DataPoints& dataPoints)
{
	std::vector<int> newIndices(ordering.size());
	std::iota(newIndices.begin(), newIndices.end(), 0);
	for(size_t i = 0; i < ordering.size(); i++)
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
		rangePrecision(Parametrizable::get<T>("rangePrecision")),
		linearSpeedNoiseX(Parametrizable::get<T>("linearSpeedNoiseX")),
		linearSpeedNoiseY(Parametrizable::get<T>("linearSpeedNoiseY")),
		linearSpeedNoiseZ(Parametrizable::get<T>("linearSpeedNoiseZ")),
		linearAccelerationNoiseX(Parametrizable::get<T>("linearAccelerationNoiseX")),
		linearAccelerationNoiseY(Parametrizable::get<T>("linearAccelerationNoiseY")),
		linearAccelerationNoiseZ(Parametrizable::get<T>("linearAccelerationNoiseZ")),
		angularSpeedNoiseX(Parametrizable::get<T>("angularSpeedNoiseX")),
		angularSpeedNoiseY(Parametrizable::get<T>("angularSpeedNoiseY")),
		angularSpeedNoiseZ(Parametrizable::get<T>("angularSpeedNoiseZ")),
		angularAccelerationNoiseX(Parametrizable::get<T>("angularAccelerationNoiseX")),
		angularAccelerationNoiseY(Parametrizable::get<T>("angularAccelerationNoiseY")),
		angularAccelerationNoiseZ(Parametrizable::get<T>("angularAccelerationNoiseZ")),
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

template<typename T>
void NoiseSkewDataPointsFilter<T>::inPlaceFilter(DataPoints& cloud)
{
	if(!cloud.timeExists("stamps"))
	{
		throw InvalidField("NoiseSkewDataPointsFilter: Error, cannot find stamps in times.");
	}
	
	Array weights = Array::Zero(1, cloud.getNbPoints());
	switch(skewModel)
	{
		case 0:
		{
			const auto& stamps = cloud.getTimeViewByName("stamps");
			Array firingDelays = (stamps.array() - stamps.minCoeff()).template cast<T>() / 1e9;
			weights = firingDelays.maxCoeff() - firingDelays;
			break;
		}
		case 1:
		{
			if(!cloud.descriptorExists("simpleSensorNoise"))
			{
				throw InvalidField("NoiseSkewDataPointsFilter: Error, cannot find simple sensor noise in descriptors.");
			}
			
			const auto& stamps = cloud.getTimeViewByName("stamps");
			const auto& simpleSensorNoise = cloud.getDescriptorViewByName("simpleSensorNoise");
			
			Array points = cloud.features.topRows(cloud.getEuclideanDim());
			Array firingDelays = (stamps.array() - stamps.minCoeff()).template cast<T>() / 1e9;
			
			Array linearVelocities = Array::Zero(points.rows(), points.cols());
			linearVelocities.row(0) = Array::Constant(1, points.cols(), linearSpeedNoiseX);
			linearVelocities.row(1) = Array::Constant(1, points.cols(), linearSpeedNoiseY);
			if(cloud.getEuclideanDim() == 3)
			{
				linearVelocities.row(2) = Array::Constant(1, points.cols(), linearSpeedNoiseZ);
			}
			Array linearAccelerations = Array::Zero(points.rows(), points.cols());
			linearAccelerations.row(0) = Array::Constant(1, points.cols(), linearAccelerationNoiseX);
			linearAccelerations.row(1) = Array::Constant(1, points.cols(), linearAccelerationNoiseY);
			if(cloud.getEuclideanDim() == 3)
			{
				linearAccelerations.row(2) = Array::Constant(1, points.cols(), linearAccelerationNoiseZ);
			}
			Array backwardTranslations = (-linearVelocities).rowwise() * firingDelays.row(0) -
										 linearAccelerations.rowwise() * (0.5 * firingDelays.pow(2)).row(0);
			Array forwardTranslations = linearVelocities.rowwise() * firingDelays.row(0) +
										linearAccelerations.rowwise() * (0.5 * firingDelays.pow(2)).row(0);
			
			Array backwardRotatedPoints = Array::Zero(points.rows(), points.cols());
			Array forwardRotatedPoints = Array::Zero(points.rows(), points.cols());
			if(cloud.getEuclideanDim() == 2)
			{
				Array angularVelocities = Array::Constant(1, points.cols(), angularSpeedNoiseZ);
				Array angularAccelerations = Array::Constant(1, points.cols(), angularAccelerationNoiseZ);
				Array rotations = angularVelocities * firingDelays + angularAccelerations * 0.5 * firingDelays.pow(2);
				Array rotationMatrix11 = rotations.cos();
				Array rotationMatrix12 = -rotations.sin();
				Array rotationMatrix21 = rotations.sin();
				Array rotationMatrix22 = rotations.cos();
				backwardRotatedPoints.row(0) = rotationMatrix11 * points.row(0) + rotationMatrix21 * points.row(1);
				backwardRotatedPoints.row(1) = rotationMatrix12 * points.row(0) + rotationMatrix22 * points.row(1);
				forwardRotatedPoints.row(0) = rotationMatrix11 * points.row(0) + rotationMatrix12 * points.row(1);
				forwardRotatedPoints.row(1) = rotationMatrix21 * points.row(0) + rotationMatrix22 * points.row(1);
			}
			else
			{
				Array angularVelocities = Array::Zero(3, points.cols());
				angularVelocities.row(0) = Array::Constant(1, points.cols(), angularSpeedNoiseX);
				angularVelocities.row(1) = Array::Constant(1, points.cols(), angularSpeedNoiseY);
				angularVelocities.row(2) = Array::Constant(1, points.cols(), angularSpeedNoiseZ);
				Array angularAccelerations = Array::Zero(3, points.cols());
				angularAccelerations.row(0) = Array::Constant(1, points.cols(), angularAccelerationNoiseX);
				angularAccelerations.row(1) = Array::Constant(1, points.cols(), angularAccelerationNoiseY);
				angularAccelerations.row(2) = Array::Constant(1, points.cols(), angularAccelerationNoiseZ);
				Array rotations = angularVelocities.rowwise() * firingDelays.row(0) +
								  angularAccelerations.rowwise() * (0.5 * firingDelays.pow(2)).row(0);
				// https://math.stackexchange.com/questions/1874898/simultaneous-action-of-two-quaternions
				Array angles = rotations.pow(2).colwise().sum().sqrt();
				Array axes = rotations.rowwise() / angles.row(0);
				axes.row(0) = axes.row(0).unaryExpr([](T value){ return std::isfinite(value) ? value : T(1); });
				axes.row(1) = axes.row(1).unaryExpr([](T value){ return std::isfinite(value) ? value : T(0); });
				axes.row(2) = axes.row(2).unaryExpr([](T value){ return std::isfinite(value) ? value : T(0); });
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
			
			Array btbrCorrectedPoints = backwardRotatedPoints + backwardTranslations;
			Array btfrCorrectedPoints = forwardRotatedPoints + backwardTranslations;
			Array ftbrCorrectedPoints = backwardRotatedPoints + forwardTranslations;
			Array ftfrCorrectedPoints = forwardRotatedPoints + forwardTranslations;
			
			Array allPossibleErrors = Array::Zero(6, points.cols());
			allPossibleErrors.row(0) = (btbrCorrectedPoints - btfrCorrectedPoints).pow(2).colwise().sum().sqrt() / 2.0;
			allPossibleErrors.row(1) = (btbrCorrectedPoints - ftbrCorrectedPoints).pow(2).colwise().sum().sqrt() / 2.0;
			allPossibleErrors.row(2) = (btbrCorrectedPoints - ftfrCorrectedPoints).pow(2).colwise().sum().sqrt() / 2.0;
			allPossibleErrors.row(3) = (btfrCorrectedPoints - ftbrCorrectedPoints).pow(2).colwise().sum().sqrt() / 2.0;
			allPossibleErrors.row(4) = (btfrCorrectedPoints - ftfrCorrectedPoints).pow(2).colwise().sum().sqrt() / 2.0;
			allPossibleErrors.row(5) = (ftbrCorrectedPoints - ftfrCorrectedPoints).pow(2).colwise().sum().sqrt() / 2.0;
			
			Array estimatedErrors = allPossibleErrors.colwise().maxCoeff();
			weights = 1.0 / (estimatedErrors + simpleSensorNoise.array()).pow(2);
			break;
		}
		case 2:
		{
			if(!cloud.descriptorExists("normals"))
			{
				throw InvalidField("NoiseSkewDataPointsFilter: Error, cannot find normals in descriptors.");
			}
			
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
				if(!cloud.descriptorExists("rings"))
				{
					throw InvalidField("NoiseSkewDataPointsFilter: Error, cannot find rings in descriptors.");
				}
				
				std::map<int, int> pointCounts;
				const auto& rings = cloud.getDescriptorViewByName("rings");
				for(unsigned int i = 0; i < cloud.getNbPoints(); i++)
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
			
			for(const int& ringId: ringIds)
			{
				Eigen::Array<int, 1, Eigen::Dynamic>& idTable = idTables[ringId];
				typename PM::DataPoints& ring = ringDataPoints[ringId];
				
				const auto& stamps = ring.getTimeViewByName("stamps");
				const auto& normals = ring.getDescriptorViewByName("normals");
				
				std::vector<int> stampOrdering = computeOrdering<std::int64_t>(stamps);
				applyOrdering(stampOrdering, idTable, ring);
				
				Array points = ring.features.topRows(ring.getEuclideanDim());
				Array laserDirections = points.rowwise() / points.pow(2).colwise().sum().sqrt();
				Array firingDelays = (stamps.colwise() - stamps.col(0)).template cast<T>() / 1e9;
				
				Array linearVelocities = Array::Zero(points.rows(), points.cols());
				linearVelocities.row(0) = Array::Constant(1, points.cols(), linearSpeedNoiseX);
				linearVelocities.row(1) = Array::Constant(1, points.cols(), linearSpeedNoiseY);
				if(ring.getEuclideanDim() == 3)
				{
					linearVelocities.row(2) = Array::Constant(1, points.cols(), linearSpeedNoiseZ);
				}
				Array linearAccelerations = Array::Zero(points.rows(), points.cols());
				linearAccelerations.row(0) = Array::Constant(1, points.cols(), linearAccelerationNoiseX);
				linearAccelerations.row(1) = Array::Constant(1, points.cols(), linearAccelerationNoiseY);
				if(ring.getEuclideanDim() == 3)
				{
					linearAccelerations.row(2) = Array::Constant(1, points.cols(), linearAccelerationNoiseZ);
				}
				Array backwardTranslations = (-linearVelocities).rowwise() * firingDelays.row(0) -
											 linearAccelerations.rowwise() * (0.5 * firingDelays.pow(2)).row(0);
				Array forwardTranslations = linearVelocities.rowwise() * firingDelays.row(0) +
											linearAccelerations.rowwise() * (0.5 * firingDelays.pow(2)).row(0);
				
				Array btbrRightPositions = Array::Zero(points.rows(), points.cols());
				Array btfrRightPositions = Array::Zero(points.rows(), points.cols());
				Array ftbrRightPositions = Array::Zero(points.rows(), points.cols());
				Array ftfrRightPositions = Array::Zero(points.rows(), points.cols());
				Array backwardRightLaserDirections = Array::Zero(points.rows(), points.cols());
				Array forwardRightLaserDirections = Array::Zero(points.rows(), points.cols());
				Array backwardRotatedPoints = Array::Zero(points.rows(), points.cols());
				Array forwardRotatedPoints = Array::Zero(points.rows(), points.cols());
				if(ring.getEuclideanDim() == 2)
				{
					Array angularVelocities = Array::Constant(1, points.cols(), angularSpeedNoiseZ);
					Array angularAccelerations = Array::Constant(1, points.cols(), angularAccelerationNoiseZ);
					Array rotations = angularVelocities * firingDelays + angularAccelerations * 0.5 * firingDelays.pow(2);
					Array rotationMatrix11 = rotations.cos();
					Array rotationMatrix12 = -rotations.sin();
					Array rotationMatrix21 = rotations.sin();
					Array rotationMatrix22 = rotations.cos();
					btbrRightPositions.row(0) = rotationMatrix11 * backwardTranslations.row(0) + rotationMatrix21 * backwardTranslations.row(1);
					btbrRightPositions.row(1) = rotationMatrix12 * backwardTranslations.row(0) + rotationMatrix22 * backwardTranslations.row(1);
					btfrRightPositions.row(0) = rotationMatrix11 * backwardTranslations.row(0) + rotationMatrix12 * backwardTranslations.row(1);
					btfrRightPositions.row(1) = rotationMatrix21 * backwardTranslations.row(0) + rotationMatrix22 * backwardTranslations.row(1);
					ftbrRightPositions.row(0) = rotationMatrix11 * forwardTranslations.row(0) + rotationMatrix21 * forwardTranslations.row(1);
					ftbrRightPositions.row(1) = rotationMatrix12 * forwardTranslations.row(0) + rotationMatrix22 * forwardTranslations.row(1);
					ftfrRightPositions.row(0) = rotationMatrix11 * forwardTranslations.row(0) + rotationMatrix12 * forwardTranslations.row(1);
					ftfrRightPositions.row(1) = rotationMatrix21 * forwardTranslations.row(0) + rotationMatrix22 * forwardTranslations.row(1);
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
					Array angularVelocities = Array::Zero(3, points.cols());
					angularVelocities.row(0) = Array::Constant(1, points.cols(), angularSpeedNoiseX);
					angularVelocities.row(1) = Array::Constant(1, points.cols(), angularSpeedNoiseY);
					angularVelocities.row(2) = Array::Constant(1, points.cols(), angularSpeedNoiseZ);
					Array angularAccelerations = Array::Zero(3, points.cols());
					angularAccelerations.row(0) = Array::Constant(1, points.cols(), angularAccelerationNoiseX);
					angularAccelerations.row(1) = Array::Constant(1, points.cols(), angularAccelerationNoiseY);
					angularAccelerations.row(2) = Array::Constant(1, points.cols(), angularAccelerationNoiseZ);
					Array rotations = angularVelocities.rowwise() * firingDelays.row(0) +
									  angularAccelerations.rowwise() * (0.5 * firingDelays.pow(2)).row(0);
					// https://math.stackexchange.com/questions/1874898/simultaneous-action-of-two-quaternions
					Array angles = rotations.pow(2).colwise().sum().sqrt();
					Array axes = rotations.rowwise() / angles.row(0);
					axes.row(0) = axes.row(0).unaryExpr([](T value){ return std::isfinite(value) ? value : T(1); });
					axes.row(1) = axes.row(1).unaryExpr([](T value){ return std::isfinite(value) ? value : T(0); });
					axes.row(2) = axes.row(2).unaryExpr([](T value){ return std::isfinite(value) ? value : T(0); });
					Array axesCrossBackwardTranslations = Array::Zero(axes.rows(), axes.cols());
					axesCrossBackwardTranslations.row(0) = axes.row(1) * backwardTranslations.row(2) - axes.row(2) * backwardTranslations.row(1);
					axesCrossBackwardTranslations.row(1) = axes.row(2) * backwardTranslations.row(0) - axes.row(0) * backwardTranslations.row(2);
					axesCrossBackwardTranslations.row(2) = axes.row(0) * backwardTranslations.row(1) - axes.row(1) * backwardTranslations.row(0);
					Array axesAxesTBackwardTranslations = Array::Zero(axes.rows(), axes.cols());
					axesAxesTBackwardTranslations.row(0) = axes.row(0) * axes.row(0) * backwardTranslations.row(0) +
														   axes.row(0) * axes.row(1) * backwardTranslations.row(1) +
														   axes.row(0) * axes.row(2) * backwardTranslations.row(2);
					axesAxesTBackwardTranslations.row(1) = axes.row(1) * axes.row(0) * backwardTranslations.row(0) +
														   axes.row(1) * axes.row(1) * backwardTranslations.row(1) +
														   axes.row(1) * axes.row(2) * backwardTranslations.row(2);
					axesAxesTBackwardTranslations.row(2) = axes.row(2) * axes.row(0) * backwardTranslations.row(0) +
														   axes.row(2) * axes.row(1) * backwardTranslations.row(1) +
														   axes.row(2) * axes.row(2) * backwardTranslations.row(2);
					Array axesCrossForwardTranslations = Array::Zero(axes.rows(), axes.cols());
					axesCrossForwardTranslations.row(0) = axes.row(1) * forwardTranslations.row(2) - axes.row(2) * forwardTranslations.row(1);
					axesCrossForwardTranslations.row(1) = axes.row(2) * forwardTranslations.row(0) - axes.row(0) * forwardTranslations.row(2);
					axesCrossForwardTranslations.row(2) = axes.row(0) * forwardTranslations.row(1) - axes.row(1) * forwardTranslations.row(0);
					Array axesAxesTForwardTranslations = Array::Zero(axes.rows(), axes.cols());
					axesAxesTForwardTranslations.row(0) = axes.row(0) * axes.row(0) * forwardTranslations.row(0) +
														  axes.row(0) * axes.row(1) * forwardTranslations.row(1) +
														  axes.row(0) * axes.row(2) * forwardTranslations.row(2);
					axesAxesTForwardTranslations.row(1) = axes.row(1) * axes.row(0) * forwardTranslations.row(0) +
														  axes.row(1) * axes.row(1) * forwardTranslations.row(1) +
														  axes.row(1) * axes.row(2) * forwardTranslations.row(2);
					axesAxesTForwardTranslations.row(2) = axes.row(2) * axes.row(0) * forwardTranslations.row(0) +
														  axes.row(2) * axes.row(1) * forwardTranslations.row(1) +
														  axes.row(2) * axes.row(2) * forwardTranslations.row(2);
					btbrRightPositions = backwardTranslations.rowwise() * (-(angles.row(0))).cos()
										 + axesCrossBackwardTranslations.rowwise() * (-(angles.row(0))).sin()
										 + axesAxesTBackwardTranslations.rowwise() * (1.0 - (-(angles.row(0))).cos());
					btfrRightPositions = backwardTranslations.rowwise() * angles.row(0).cos()
										 + axesCrossBackwardTranslations.rowwise() * angles.row(0).sin()
										 + axesAxesTBackwardTranslations.rowwise() * (1.0 - angles.row(0).cos());
					ftbrRightPositions = forwardTranslations.rowwise() * (-(angles.row(0))).cos()
										 + axesCrossForwardTranslations.rowwise() * (-(angles.row(0))).sin()
										 + axesAxesTForwardTranslations.rowwise() * (1.0 - (-(angles.row(0))).cos());
					ftfrRightPositions = forwardTranslations.rowwise() * angles.row(0).cos() + axesCrossForwardTranslations.rowwise() * angles.row(0).sin()
										 + axesAxesTForwardTranslations.rowwise() * (1.0 - angles.row(0).cos());
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
					backwardRightLaserDirections = laserDirections.rowwise() * (-(angles.row(0))).cos()
												   + axesCrossLaserDirections.rowwise() * (-(angles.row(0))).sin()
												   + axesAxesTLaserDirections.rowwise() * (1.0 - (-(angles.row(0))).cos());
					forwardRightLaserDirections = laserDirections.rowwise() * angles.row(0).cos()
												  + axesCrossLaserDirections.rowwise() * angles.row(0).sin()
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
				
				Array btbrRightDistances = ((points - btbrRightPositions) * normals.array()).colwise().sum() /
										   (backwardRightLaserDirections * normals.array()).colwise().sum();
				Array btfrRightDistances = ((points - btfrRightPositions) * normals.array()).colwise().sum() /
										   (forwardRightLaserDirections * normals.array()).colwise().sum();
				Array ftbrRightDistances = ((points - ftbrRightPositions) * normals.array()).colwise().sum() /
										   (backwardRightLaserDirections * normals.array()).colwise().sum();
				Array ftfrRightDistances = ((points - ftfrRightPositions) * normals.array()).colwise().sum() /
										   (forwardRightLaserDirections * normals.array()).colwise().sum();
				Array btbrCorrectedPoints = backwardRotatedPoints + backwardTranslations;
				Array btfrCorrectedPoints = forwardRotatedPoints + backwardTranslations;
				Array ftbrCorrectedPoints = backwardRotatedPoints + forwardTranslations;
				Array ftfrCorrectedPoints = forwardRotatedPoints + forwardTranslations;
				Array btbrCorrectedPointDirections = btbrCorrectedPoints.rowwise() / btbrCorrectedPoints.pow(2).colwise().sum().sqrt();
				Array btfrCorrectedPointDirections = btfrCorrectedPoints.rowwise() / btfrCorrectedPoints.pow(2).colwise().sum().sqrt();
				Array ftbrCorrectedPointDirections = ftbrCorrectedPoints.rowwise() / ftbrCorrectedPoints.pow(2).colwise().sum().sqrt();
				Array ftfrCorrectedPointDirections = ftfrCorrectedPoints.rowwise() / ftfrCorrectedPoints.pow(2).colwise().sum().sqrt();
				
				Array allPossibleErrors = Array::Zero(6, points.cols());
				allPossibleErrors.row(0) = (btbrRightDistances - btfrRightDistances).abs() / 2.0;
				allPossibleErrors.row(1) = (btbrRightDistances - ftbrRightDistances).abs() / 2.0;
				allPossibleErrors.row(2) = (btbrRightDistances - ftfrRightDistances).abs() / 2.0;
				allPossibleErrors.row(3) = (btfrRightDistances - ftbrRightDistances).abs() / 2.0;
				allPossibleErrors.row(4) = (btfrRightDistances - ftfrRightDistances).abs() / 2.0;
				allPossibleErrors.row(5) = (ftbrRightDistances - ftfrRightDistances).abs() / 2.0;
				
				Array estimatedErrors = allPossibleErrors.colwise().maxCoeff();
				Array ringWeights = 1.0 / (estimatedErrors + rangePrecision).pow(2);
				
				Array cornerness = Array::Zero(estimatedErrors.rows(), estimatedErrors.cols());
				cornerness.block(0, 1, 1, cornerness.cols() - 1) = (estimatedErrors.block(0, 1, 1, estimatedErrors.cols() - 1) -
																	estimatedErrors.block(0, 0, 1, estimatedErrors.cols() - 1)).abs();
				Array sortedCornerness(cornerness);
				std::sort(sortedCornerness.data(), sortedCornerness.data() + sortedCornerness.size());
				T lowerQuartile = sortedCornerness(0, std::ceil(sortedCornerness.cols() / 4.0) - 1);
				T upperQuartile = sortedCornerness(0, std::ceil(3.0 * sortedCornerness.cols() / 4.0) - 1);
				T IQR = upperQuartile - lowerQuartile;
				T threshold = upperQuartile + (15 * IQR);
				std::vector<int> cornerIds;
				visit_lambda(cornerness, [&cornerIds, threshold](T value, int i, int j){
					if(value > threshold)
					{
						cornerIds.push_back(j);
					}
				});
				
				Array pointHorizontalAngles;
				Array btbrCorrectedPointHorizontalAngles;
				Array btfrCorrectedPointHorizontalAngles;
				Array ftbrCorrectedPointHorizontalAngles;
				Array ftfrCorrectedPointHorizontalAngles;
				if(ring.getEuclideanDim() == 2)
				{
					pointHorizontalAngles = laserDirections.row(1)
							.binaryExpr(laserDirections.row(0), [](T a, T b){ return std::atan2(a, b); });
					btbrCorrectedPointHorizontalAngles = btbrCorrectedPointDirections.row(1)
							.binaryExpr(btbrCorrectedPointDirections.row(0), [](T a, T b){ return std::atan2(a, b); });
					btfrCorrectedPointHorizontalAngles = btfrCorrectedPointDirections.row(1)
							.binaryExpr(btfrCorrectedPointDirections.row(0), [](T a, T b){ return std::atan2(a, b); });
					ftbrCorrectedPointHorizontalAngles = ftbrCorrectedPointDirections.row(1)
							.binaryExpr(ftbrCorrectedPointDirections.row(0), [](T a, T b){ return std::atan2(a, b); });
					ftfrCorrectedPointHorizontalAngles = ftfrCorrectedPointDirections.row(1)
							.binaryExpr(ftfrCorrectedPointDirections.row(0), [](T a, T b){ return std::atan2(a, b); });
				}
				else
				{
					Array pointVerticalAngles = laserDirections.row(2).unaryExpr([](T a){ return std::asin(a); });
					pointHorizontalAngles = (laserDirections.row(1) / pointVerticalAngles.cos())
							.binaryExpr(laserDirections.row(0) / pointVerticalAngles.cos(), [](T a, T b){ return std::atan2(a, b); });
					Array btbrCorrectedPointVerticalAngles = btbrCorrectedPointDirections.row(2).unaryExpr([](T a){ return std::asin(a); });
					btbrCorrectedPointHorizontalAngles = (btbrCorrectedPointDirections.row(1) / btbrCorrectedPointVerticalAngles.cos())
							.binaryExpr(btbrCorrectedPointDirections.row(0) / btbrCorrectedPointVerticalAngles.cos(), [](T a, T b){ return std::atan2(a, b); });
					Array btfrCorrectedPointVerticalAngles = btfrCorrectedPointDirections.row(2).unaryExpr([](T a){ return std::asin(a); });
					btfrCorrectedPointHorizontalAngles = (btfrCorrectedPointDirections.row(1) / btfrCorrectedPointVerticalAngles.cos())
							.binaryExpr(btfrCorrectedPointDirections.row(0) / btfrCorrectedPointVerticalAngles.cos(), [](T a, T b){ return std::atan2(a, b); });
					Array ftbrCorrectedPointVerticalAngles = ftbrCorrectedPointDirections.row(2).unaryExpr([](T a){ return std::asin(a); });
					ftbrCorrectedPointHorizontalAngles = (ftbrCorrectedPointDirections.row(1) / ftbrCorrectedPointVerticalAngles.cos())
							.binaryExpr(ftbrCorrectedPointDirections.row(0) / ftbrCorrectedPointVerticalAngles.cos(), [](T a, T b){ return std::atan2(a, b); });
					Array ftfrCorrectedPointVerticalAngles = ftfrCorrectedPointDirections.row(2).unaryExpr([](T a){ return std::asin(a); });
					ftfrCorrectedPointHorizontalAngles = (ftfrCorrectedPointDirections.row(1) / ftfrCorrectedPointVerticalAngles.cos())
							.binaryExpr(ftfrCorrectedPointDirections.row(0) / ftfrCorrectedPointVerticalAngles.cos(), [](T a, T b){ return std::atan2(a, b); });
				}
				
				Array cornerWeights = Array::Ones(1, points.cols());
				for(const auto& cornerId: cornerIds)
				{
					T lowerHorizontalAngle = std::min({ btbrCorrectedPointHorizontalAngles(0, cornerId), btfrCorrectedPointHorizontalAngles(0, cornerId),
														ftbrCorrectedPointHorizontalAngles(0, cornerId), ftfrCorrectedPointHorizontalAngles(0, cornerId) });
					T higherHorizontalAngle = std::max({ btbrCorrectedPointHorizontalAngles(0, cornerId), btfrCorrectedPointHorizontalAngles(0, cornerId),
														 ftbrCorrectedPointHorizontalAngles(0, cornerId), ftfrCorrectedPointHorizontalAngles(0, cornerId) });
					visit_lambda(pointHorizontalAngles, [this, &lowerHorizontalAngle, &higherHorizontalAngle, &cornerWeights](T value, int i, int j){
						if(value >= lowerHorizontalAngle && value <= higherHorizontalAngle)
						{
							cornerWeights(i, j) = cornerPointWeight;
						}
					});
				}
				ringWeights *= cornerWeights;
				
				visit_lambda(ringWeights, [&weights, &idTable](T value, int i, int j){ weights(0, idTable(i, j)) = value; });
			}
			break;
		}
		case 3:
		{
			if(!cloud.descriptorExists("curvatures"))
			{
				throw InvalidField("NoiseSkewDataPointsFilter: Error, cannot find curvatures in descriptors.");
			}
			
			const auto& stamps = cloud.getTimeViewByName("stamps");
			const auto& curvatures = cloud.getDescriptorViewByName("curvatures");
			
			Array firingDelays = (stamps.colwise() - stamps.col(0)).template cast<T>() / 1e9;
			Array scanningAngles = (firingDelays / firingDelays.maxCoeff()) * T(2 * M_PI);
			Array partialWeights = Array::Zero(2, cloud.getNbPoints());
			partialWeights.row(0) = (scanningAngles / T(4)).cos();
			partialWeights.row(1) = (curvatures.array() / REFERENCE_CURVATURE).unaryExpr([](T a){ return std::max(std::min(a, T(1)), T(0.25)); });
			weights = partialWeights.colwise().maxCoeff();
			break;
		}
		default:
			throw InvalidParameter("NoiseSkewDataPointsFilter: Error, skewModel id " + std::to_string(skewModel) + " does not exist.");
	}
	
	std::vector<int> weightOrdering = computeOrdering<T>(weights);
	int weightQuantileIndex = std::ceil(weightQuantile * (weightOrdering.size() - 1));
	T weightQuantileValue = weights(0, weightOrdering[weightQuantileIndex]);
	for(int i = 0; i < weights.cols(); i++)
	{
		if(weights(0, i) < weightQuantileValue)
		{
			weights(0, i) = T(0);
		}
	}
	
	cloud.addDescriptor("skewWeight", weights);
}

template struct NoiseSkewDataPointsFilter<float>;
template struct NoiseSkewDataPointsFilter<double>;
