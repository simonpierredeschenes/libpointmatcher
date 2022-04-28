#include "DeskewingUncertainty.h"
#include <numeric>
#include <boost/lexical_cast.hpp>
#include "utils/utils.h"

template<typename Func>
struct lambda_as_visitor_wrapper : Func
{
	lambda_as_visitor_wrapper(const Func& f) :
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

template<typename T>
std::vector<T> DeskewingUncertaintyDataPointsFilter<T>::castToScalarVector(const std::string& values)
{
	size_t lastPos = 0;
	size_t pos;
	std::vector<T> vector;
	while((pos = values.find(',', lastPos)) != std::string::npos)
	{
		vector.push_back(boost::lexical_cast<T>(values.substr(lastPos, pos - lastPos)));
		lastPos = pos + 1;
	}
	if(!values.empty())
	{
		vector.push_back(boost::lexical_cast<T>(values.substr(lastPos)));
	}
	return vector;
}

template<typename T>
std::vector<typename PointMatcher<T>::Vector>
DeskewingUncertaintyDataPointsFilter<T>::castToVectorVector(const std::string& xValues, const std::string& yValues, const std::string& zValues)
{
	std::vector<T> x = castToScalarVector(xValues);
	std::vector<T> y = castToScalarVector(yValues);
	std::vector<T> z = castToScalarVector(zValues);

	std::vector<Vector> vector;
	for(unsigned int i = 0; i < x.size(); ++i)
	{
		vector.push_back((Vector(3) << x[i], y[i], z[i]).finished());
	}
	return vector;
}

// https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
template<typename T>
template<typename U>
std::vector<int> DeskewingUncertaintyDataPointsFilter<T>::computeOrdering(const Eigen::Matrix<U, 1, Eigen::Dynamic>& elements)
{
	std::vector<int> indices(elements.cols());
	std::iota(indices.begin(), indices.end(), 0);
	std::stable_sort(indices.begin(), indices.end(), [&elements](int index1, int index2)
	{ return elements(0, index1) < elements(0, index2); });
	return indices;
}

template<typename T>
void DeskewingUncertaintyDataPointsFilter<T>::applyOrdering(const std::vector<int>& ordering, Eigen::Matrix<int, 1, Eigen::Dynamic>& idTable, DataPoints& dataPoints)
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
DeskewingUncertaintyDataPointsFilter<T>::DeskewingUncertaintyDataPointsFilter(const Parameters& params):
		PointMatcher<T>::DataPointsFilter("DeskewingUncertaintyDataPointsFilter", DeskewingUncertaintyDataPointsFilter::availableParameters(), params),
		linearVelocities(castToVectorVector(Parametrizable::getParamValueString("linearSpeedsX"),
											Parametrizable::getParamValueString("linearSpeedsY"),
											Parametrizable::getParamValueString("linearSpeedsZ"))),
		angularVelocities(castToVectorVector(Parametrizable::getParamValueString("angularSpeedsX"),
											 Parametrizable::getParamValueString("angularSpeedsY"),
											 Parametrizable::getParamValueString("angularSpeedsZ"))),
		measureTimes(castToScalarVector(Parametrizable::getParamValueString("measureTimes"))
		)
{
	for(unsigned int i = 0; i < linearVelocities.size(); ++i)
	{
		Gaussian<T> motionGaussian;
		motionGaussian.mean = Vector::Zero(6);
		motionGaussian.mean.head(3) = linearVelocities[i];
		motionGaussian.mean.tail(3) = angularVelocities[i];
		motionGaussian.covariance = Matrix::Zero(6, 6);
//		motionGaussian.covariance(0, 0) = linearVelocities[i](0) * linearVelocities[i](0);
//		motionGaussian.covariance(1, 1) = linearVelocities[i](1) * linearVelocities[i](1);
//		motionGaussian.covariance(2, 2) = linearVelocities[i](2) * linearVelocities[i](2);
//		motionGaussian.covariance(3, 3) = angularVelocities[i](0) * angularVelocities[i](0);
//		motionGaussian.covariance(4, 4) = angularVelocities[i](1) * angularVelocities[i](1);
//		motionGaussian.covariance(5, 5) = angularVelocities[i](2) * angularVelocities[i](2);
		motionGaussian.covariance(0, 0) = 0.004 * 0.004;
		motionGaussian.covariance(1, 1) = 0.004 * 0.004;
		motionGaussian.covariance(2, 2) = 0.004 * 0.004;
		motionGaussian.covariance(3, 3) = 0.025 * 0.025;
		motionGaussian.covariance(4, 4) = 0.025 * 0.025;
		motionGaussian.covariance(5, 5) = 0.025 * 0.025;
		motionGaussians.push_back(motionGaussian);
	}
}

template<typename T>
typename PointMatcher<T>::DataPoints DeskewingUncertaintyDataPointsFilter<T>::filter(const DataPoints& input)
{
	DataPoints output(input);
	inPlaceFilter(output);
	return output;
}

template<typename T>
void DeskewingUncertaintyDataPointsFilter<T>::inPlaceFilter(DataPoints& cloud)
{
	if(!cloud.timeExists("stamps"))
	{
		throw InvalidField("DeskewingUncertaintyDataPointsFilter: Error, cannot find stamps in times.");
	}

	typename PM::DataPoints orderedDataPoints = cloud;
	const auto& stamps = orderedDataPoints.getTimeViewByName("stamps");
	std::vector<int> stampOrdering = computeOrdering<std::int64_t>(stamps);
	Eigen::Matrix<int, 1, Eigen::Dynamic> idTable = Eigen::Matrix<int, 1, Eigen::Dynamic>::LinSpaced(orderedDataPoints.getNbPoints(), 0, orderedDataPoints.getNbPoints() - 1);
	applyOrdering(stampOrdering, idTable, orderedDataPoints);

	Matrix points = orderedDataPoints.features;
	Matrix firingDelays = (stamps.colwise() - stamps.col(0)).template cast<T>() / 1e9;

	int latestMeasureIndex = 0;
	Gaussian<T> latestPoseGaussian;
	latestPoseGaussian.mean = Vector::Zero(6);
	latestPoseGaussian.covariance = Matrix::Zero(6, 6);
	latestPoseGaussian.covariance.topLeftCorner(3, 3) = Matrix::Identity(3, 3) * 0.0001;
	Matrix pointCovariances = Matrix::Zero(9, cloud.getNbPoints());
	for(unsigned int i = 0; i < orderedDataPoints.getNbPoints(); ++i)
	{
		if(latestMeasureIndex + 1 < measureTimes.size() && firingDelays(0, i) > measureTimes[latestMeasureIndex + 1])
		{
			latestPoseGaussian = addMotion<T>(latestPoseGaussian, motionGaussians[latestMeasureIndex], measureTimes[latestMeasureIndex + 1] - measureTimes[latestMeasureIndex]);
			++latestMeasureIndex;
		}
		Gaussian<T> currentPoseGaussian = addMotion<T>(latestPoseGaussian, motionGaussians[latestMeasureIndex], firingDelays(0, i) - measureTimes[latestMeasureIndex]);
		Gaussian<T> pointGaussian = propagateUncertainty<T>(currentPoseGaussian, points.col(i));
		pointCovariances.col(idTable(0, i)) = PointMatcherSupport::serializeEigVec<T>(pointGaussian.covariance);
	}
	cloud.addDescriptor("covariance", pointCovariances);

	// ========================= DEBUG =========================
//	Matrix covXScale = Matrix::Zero(1, cloud.getNbPoints());
//	Matrix covYScale = Matrix::Zero(1, cloud.getNbPoints());
//	Matrix covZScale = Matrix::Zero(1, cloud.getNbPoints());
//	Matrix covX = Matrix::Zero(3, cloud.getNbPoints());
//	Matrix covY = Matrix::Zero(3, cloud.getNbPoints());
//	Matrix covZ = Matrix::Zero(3, cloud.getNbPoints());
//	for(unsigned int i = 0; i < cloud.getNbPoints(); ++i)
//	{
//		Matrix C = PointMatcherSupport::deserializeEigVec<T>(pointCovariances.col(i));
//		if(C.fullPivHouseholderQr().rank() >= 2)
//		{
//			const Eigen::EigenSolver<Matrix> solver(C);
//			Vector eigenVa = solver.eigenvalues().real();
//			Matrix eigenVe = solver.eigenvectors().real();
//			covXScale(0, i) = std::sqrt(eigenVa(0));
//			covYScale(0, i) = std::sqrt(eigenVa(1));
//			covZScale(0, i) = std::sqrt(eigenVa(2));
//			covX.col(i) = eigenVe.col(0);
//			covY.col(i) = eigenVe.col(1);
//			covZ.col(i) = eigenVe.col(2);
//		}
//	}
//	cloud.addDescriptor("covXScale", covXScale);
//	cloud.addDescriptor("covYScale", covYScale);
//	cloud.addDescriptor("covZScale", covZScale);
//	cloud.addDescriptor("covX", covX);
//	cloud.addDescriptor("covY", covY);
//	cloud.addDescriptor("covZ", covZ);
	// =========================================================
}

template
struct DeskewingUncertaintyDataPointsFilter<float>;
template
struct DeskewingUncertaintyDataPointsFilter<double>;
