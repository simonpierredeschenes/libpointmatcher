#include "SurfaceCurvature.h"
#include "MatchersImpl.h"

template<typename T>
SurfaceCurvatureDataPointsFilter<T>::SurfaceCurvatureDataPointsFilter(const Parameters& params):
		PointMatcher<T>::DataPointsFilter("SurfaceCurvatureDataPointsFilter", SurfaceCurvatureDataPointsFilter::availableParameters(), params),
		knn(Parametrizable::get<int>("knn")),
		maxDist(Parametrizable::get<T>("maxDist")),
		epsilon(Parametrizable::get<T>("epsilon"))
{
}

template<typename T>
typename PointMatcher<T>::DataPoints
SurfaceCurvatureDataPointsFilter<T>::filter(const DataPoints& input)
{
	DataPoints output(input);
	inPlaceFilter(output);
	return output;
}

template<typename T>
void SurfaceCurvatureDataPointsFilter<T>::inPlaceFilter(DataPoints& cloud)
{
	if(!cloud.descriptorExists("normals"))
	{
		throw InvalidField("SurfaceCurvatureDataPointsFilter: Error, cannot find normals in descriptors.");
	}
	
	typedef typename MatchersImpl<T>::KDTreeMatcher KDTreeMatcher;
	typedef typename PointMatcher<T>::Matches Matches;
	
	const auto& normals = cloud.getDescriptorViewByName("normals");
	const auto& points = cloud.features.topRows(cloud.getEuclideanDim());
	
	Parametrizable::Parameters matcherParams;
	matcherParams["knn"] = PointMatcherSupport::toParam(knn);
	matcherParams["epsilon"] = PointMatcherSupport::toParam(epsilon);
	matcherParams["maxDist"] = PointMatcherSupport::toParam(maxDist);
	KDTreeMatcher matcher(matcherParams);
	matcher.init(cloud);
	
	Matches matches(typename Matches::Dists(knn, cloud.getNbPoints()), typename Matches::Ids(knn, cloud.getNbPoints()));
	matches = matcher.findClosests(cloud);
	
	typename PM::Matrix curvatures = PM::Matrix::Zero(1, cloud.getNbPoints());
	for(int i = 0; i < points.cols(); ++i)
	{
		int realKnn = 0;
		T curvatureSum = T(0);
		typename PM::Matrix neighbors(points.rows(), knn);
		for(unsigned int j = 0; j < knn; ++j)
		{
			if(matches.ids(j, i) != i && matches.dists(j, i) != Matches::InvalidDist)
			{
				int neighborIndex = matches.ids(j, i);
				T angleBetweenNormals = std::acos(std::min(T(1), std::fabs(normals.col(i).normalized().dot(normals.col(neighborIndex).normalized()))));
				T distanceBetweenPoints = (points.col(i) - points.col(neighborIndex)).norm();
				curvatureSum += angleBetweenNormals / distanceBetweenPoints;
				++realKnn;
			}
		}
		
		if(realKnn > 0)
		{
			curvatures(0, i) = curvatureSum / T(realKnn);
		}
	}
	
	cloud.addDescriptor("curvatures", curvatures);
}

template struct SurfaceCurvatureDataPointsFilter<float>;
template struct SurfaceCurvatureDataPointsFilter<double>;

