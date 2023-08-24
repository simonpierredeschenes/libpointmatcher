#include "DeskewingUncertainty.h"
#include <numeric>
#include <boost/lexical_cast.hpp>
#include "utils/utils.h"
#include <fstream>

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

template<typename T>
std::vector<typename PointMatcher<T>::Matrix>
DeskewingUncertaintyDataPointsFilter<T>::castToMatrixVector(const std::string& xValues, const std::string& xyValues, const std::string& xzValues, const std::string& yValues,
                                                            const std::string& yzValues, const std::string& zValues)
{
    std::vector<T> x = castToScalarVector(xValues);
    std::vector<T> xy = castToScalarVector(xyValues);
    std::vector<T> xz = castToScalarVector(xzValues);
    std::vector<T> y = castToScalarVector(yValues);
    std::vector<T> yz = castToScalarVector(yzValues);
    std::vector<T> z = castToScalarVector(zValues);

    std::vector<Matrix> vector;
    for(unsigned int i = 0; i < x.size(); ++i)
    {
        Matrix matrix = Matrix::Zero(3, 3);
        matrix(0, 0) = x[i];
        matrix(0, 1) = xy[i];
        matrix(0, 2) = xz[i];
        matrix(1, 0) = xy[i];
        matrix(1, 1) = y[i];
        matrix(1, 2) = yz[i];
        matrix(2, 0) = xz[i];
        matrix(2, 1) = yz[i];
        matrix(2, 2) = z[i];
        vector.push_back(matrix);
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
        measureTimes(castToScalarVector(Parametrizable::getParamValueString("measureTimes")))
{
    std::vector<Vector> linearVelocities = castToVectorVector(Parametrizable::getParamValueString("linearSpeedsX"), Parametrizable::getParamValueString("linearSpeedsY"), Parametrizable::getParamValueString("linearSpeedsZ"));
    std::vector<Matrix> linearVelocityCovariances = castToMatrixVector(Parametrizable::getParamValueString("linearSpeedVariancesX"), Parametrizable::getParamValueString("linearSpeedCovariancesXY"),
                                                                        Parametrizable::getParamValueString("linearSpeedCovariancesXZ"), Parametrizable::getParamValueString("linearSpeedVariancesY"),
                                                                        Parametrizable::getParamValueString("linearSpeedCovariancesYZ"), Parametrizable::getParamValueString("linearSpeedVariancesZ"));
    std::vector<Vector> angularVelocities = castToVectorVector(Parametrizable::getParamValueString("angularSpeedsX"), Parametrizable::getParamValueString("angularSpeedsY"), Parametrizable::getParamValueString("angularSpeedsZ"));
    std::vector<Matrix> angularVelocityCovariances = castToMatrixVector(Parametrizable::getParamValueString("angularSpeedVariancesX"), Parametrizable::getParamValueString("angularSpeedCovariancesXY"),
                                                                        Parametrizable::getParamValueString("angularSpeedCovariancesXZ"), Parametrizable::getParamValueString("angularSpeedVariancesY"),
                                                                        Parametrizable::getParamValueString("angularSpeedCovariancesYZ"), Parametrizable::getParamValueString("angularSpeedVariancesZ"));

	for(unsigned int i = 0; i < linearVelocities.size(); ++i)
	{
		Gaussian<T> motionGaussian;
		motionGaussian.mean = Vector::Zero(6);
		motionGaussian.covariance = Matrix::Zero(6, 6);
		motionGaussian.covariance.topLeftCorner(3, 3) = linearVelocityCovariances[i];
        motionGaussian.covariance.bottomRightCorner(3, 3) = angularVelocityCovariances[i];
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
	latestPoseGaussian.covariance = Matrix::Identity(6, 6) * 1e-12;
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
//	Matrix covScale = Matrix::Zero(1, cloud.getNbPoints());
//	Matrix covX = Matrix::Zero(3, cloud.getNbPoints());
//	Matrix covY = Matrix::Zero(3, cloud.getNbPoints());
//	Matrix covZ = Matrix::Zero(3, cloud.getNbPoints());
//	for(unsigned int i = 0; i < cloud.getNbPoints(); ++i)
//	{
//		Matrix C = PointMatcherSupport::deserializeEigVec<T>(pointCovariances.col(i));
//		const Eigen::EigenSolver<Matrix> solver(C);
//		Vector eigenVa = solver.eigenvalues().real();
//		Matrix eigenVe = solver.eigenvectors().real();
//
//		Vector tmp_eigenVa = eigenVa;
//		Matrix tmp_eigenVe = eigenVe;
//		if(tmp_eigenVa(0) > tmp_eigenVa(1))
//		{
//			if(tmp_eigenVa(1) > tmp_eigenVa(2))
//			{
//				eigenVa(0) = tmp_eigenVa(0);
//				eigenVe.col(0) = tmp_eigenVe.col(0);
//				eigenVa(1) = tmp_eigenVa(1);
//				eigenVe.col(1) = tmp_eigenVe.col(1);
//				eigenVa(2) = tmp_eigenVa(2);
//				eigenVe.col(2) = tmp_eigenVe.col(2);
//			}
//			else
//			{
//				if(tmp_eigenVa(0) > tmp_eigenVa(2))
//				{
//					eigenVa(0) = tmp_eigenVa(0);
//					eigenVe.col(0) = tmp_eigenVe.col(0);
//					eigenVa(1) = tmp_eigenVa(2);
//					eigenVe.col(1) = tmp_eigenVe.col(2);
//					eigenVa(2) = tmp_eigenVa(1);
//					eigenVe.col(2) = tmp_eigenVe.col(1);
//				}
//				else
//				{
//					eigenVa(0) = tmp_eigenVa(2);
//					eigenVe.col(0) = tmp_eigenVe.col(2);
//					eigenVa(1) = tmp_eigenVa(0);
//					eigenVe.col(1) = tmp_eigenVe.col(0);
//					eigenVa(2) = tmp_eigenVa(1);
//					eigenVe.col(2) = tmp_eigenVe.col(1);
//				}
//			}
//		}
//		else
//		{
//			if(tmp_eigenVa(0) > tmp_eigenVa(2))
//			{
//				eigenVa(0) = tmp_eigenVa(1);
//				eigenVe.col(0) = tmp_eigenVe.col(1);
//				eigenVa(1) = tmp_eigenVa(0);
//				eigenVe.col(1) = tmp_eigenVe.col(0);
//				eigenVa(2) = tmp_eigenVa(2);
//				eigenVe.col(2) = tmp_eigenVe.col(2);
//			}
//			else
//			{
//				if(tmp_eigenVa(1) > tmp_eigenVa(2))
//				{
//					eigenVa(0) = tmp_eigenVa(1);
//					eigenVe.col(0) = tmp_eigenVe.col(1);
//					eigenVa(1) = tmp_eigenVa(2);
//					eigenVe.col(1) = tmp_eigenVe.col(2);
//					eigenVa(2) = tmp_eigenVa(0);
//					eigenVe.col(2) = tmp_eigenVe.col(0);
//				}
//				else
//				{
//					eigenVa(0) = tmp_eigenVa(2);
//					eigenVe.col(0) = tmp_eigenVe.col(2);
//					eigenVa(1) = tmp_eigenVa(1);
//					eigenVe.col(1) = tmp_eigenVe.col(1);
//					eigenVa(2) = tmp_eigenVa(0);
//					eigenVe.col(2) = tmp_eigenVe.col(0);
//				}
//			}
//		}
//
//		covXScale(0, i) = std::sqrt(eigenVa(0));
//		covYScale(0, i) = std::sqrt(eigenVa(1));
//		covZScale(0, i) = std::sqrt(eigenVa(2));
//		covScale(0, i) = std::sqrt(eigenVa(0) + eigenVa(1) + eigenVa(2));
//		covX.col(i) = eigenVe.col(0);
//		covY.col(i) = eigenVe.col(1);
//		covZ.col(i) = eigenVe.col(2);
//	}
//	cloud.addDescriptor("covXScale", covXScale);
//	cloud.addDescriptor("covYScale", covYScale);
//	cloud.addDescriptor("covZScale", covZScale);
//	cloud.addDescriptor("covScale", covScale);
//	cloud.addDescriptor("covX", covX);
//	cloud.addDescriptor("covY", covY);
//	cloud.addDescriptor("covZ", covZ);
	// =========================================================
}

template
struct DeskewingUncertaintyDataPointsFilter<float>;
template
struct DeskewingUncertaintyDataPointsFilter<double>;
