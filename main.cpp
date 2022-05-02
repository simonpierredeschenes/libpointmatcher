#include <pointmatcher/PointMatcher.h>
#include "pointmatcher/DataPointsFilters/utils/utils.h"

typedef PointMatcher<float> PM;

typename PM::DataPoints generateReading()
{
	PM::Matrix features = PM::Matrix::Zero(4, 3);
	for(unsigned int i = 0; i < 3; ++i)
	{
		double angle = -M_PI / 4 + (i * M_PI / 4);
		if(i == 1)
		{
			angle += M_PI / 8;
		}
		features.col(i) = (PM::Vector(4) << std::cos(angle), std::sin(angle), 0.f, 1).finished();
	}
	PM::DataPoints::Labels featureLabels;
	featureLabels.push_back(PM::DataPoints::Label("x", 1));
	featureLabels.push_back(PM::DataPoints::Label("y", 1));
	featureLabels.push_back(PM::DataPoints::Label("z", 1));
	featureLabels.push_back(PM::DataPoints::Label("pad", 1));

	PM::Matrix descriptors = PM::Matrix::Zero(1, features.cols());
	for(unsigned int i = 0; i < features.cols(); ++i)
	{
		descriptors(0, i) = i;
	}
	PM::DataPoints::Labels descriptorLabels;
	descriptorLabels.push_back(PM::DataPoints::Label("firingDelays", 1));

	PM::Int64Matrix times = PM::Int64Matrix::Zero(1, features.cols());
	for(unsigned int i = 0; i < features.cols(); ++i)
	{
		times(0, i) = descriptors(0, i) * 1e7;
	}
	PM::DataPoints::Labels timeLabels;
	timeLabels.push_back(PM::DataPoints::Label("stamps", 1));

	PM::DataPoints cloud(features, featureLabels, descriptors, descriptorLabels, times, timeLabels);

//	PM::Matrix covariances = PM::Matrix::Zero(9, 3);
//	covariances.col(0) = PointMatcherSupport::serializeEigVec<float>(PM::Matrix::Identity(3, 3) * 1);
//	covariances.col(1) = PointMatcherSupport::serializeEigVec<float>(PM::Matrix::Identity(3, 3) * 0.1);
//	covariances.col(2) = PointMatcherSupport::serializeEigVec<float>(PM::Matrix::Identity(3, 3) * 1);
//	cloud.addDescriptor("covariance", covariances);
//
//  return cloud;

	PM::Parameters params;
	params["linearSpeedsX"] = "0";
	params["linearSpeedsY"] = "0";
	params["linearSpeedsZ"] = "0";
	params["angularSpeedsX"] = "0";
	params["angularSpeedsY"] = "0";
	params["angularSpeedsZ"] = "1";
	params["measureTimes"] = "0";
	std::shared_ptr<PM::DataPointsFilter> deskewingFilter = PM::get().DataPointsFilterRegistrar.create("DeskewingUncertaintyDataPointsFilter", params);
	return deskewingFilter->filter(cloud);
}

typename PM::DataPoints generateReference()
{
	PM::Matrix features = PM::Matrix::Zero(4, 3);
	for(unsigned int i = 0; i < 3; ++i)
	{
		double angle = -M_PI / 4 + (i * M_PI / 4);
		features.col(i) = (PM::Vector(4) << std::cos(angle), std::sin(angle), 0.f, 1).finished();
	}
	PM::DataPoints::Labels featureLabels;
	featureLabels.push_back(PM::DataPoints::Label("x", 1));
	featureLabels.push_back(PM::DataPoints::Label("y", 1));
	featureLabels.push_back(PM::DataPoints::Label("z", 1));
	featureLabels.push_back(PM::DataPoints::Label("pad", 1));
	return {features, featureLabels};
}

int main(int argc, char** argv)
{
	PM::DataPoints reading = generateReading();
	PM::DataPoints reference = generateReference();

	PM::ICP icp;
	std::shared_ptr<PM::Transformation> transformation = PM::get().TransformationRegistrar.create("RigidTransformation");
	icp.transformations.push_back(transformation);
	PM::Parameters params;
	params["maxDist"] = "inf";
	params["knn"] = "1";
	std::shared_ptr<PM::Matcher> matcher = PM::get().MatcherRegistrar.create("KDTreeMatcher", params);
	matcher->init(reference);
	icp.matcher = matcher;
	std::shared_ptr<PM::ErrorMinimizer> gaussianToPointMinimizer = PM::get().ErrorMinimizerRegistrar.create("GaussianToPointErrorMinimizer");
	icp.errorMinimizer = gaussianToPointMinimizer;
	params.clear();
	params["minDiffTransErr"] = "0.01";
	params["minDiffRotErr"] = "0.001";
	params["smoothLength"] = "2";
	std::shared_ptr<PM::TransformationChecker> diffChecker = PM::get().TransformationCheckerRegistrar.create("DifferentialTransformationChecker", params);
	icp.transformationCheckers.push_back(diffChecker);
	params.clear();
	params["maxIterationCount"] = "40";
	std::shared_ptr<PM::TransformationChecker> counterChecker = PM::get().TransformationCheckerRegistrar.create("CounterTransformationChecker", params);
	icp.transformationCheckers.push_back(counterChecker);
	params.clear();
	params["baseFileName"] = "/home/norlab/Desktop/inspector_output/test";
	params["dumpDataLinks"] = "1";
	params["dumpReading"] = "1";
	params["dumpReference"] = "1";
	std::shared_ptr<PM::Inspector> inspector = PM::get().InspectorRegistrar.create("VTKFileInspector", params);
	icp.inspector = inspector;

	PM::TransformationParameters optimalTransform = icp(reading, reference);
	icp.transformations.apply(reading, optimalTransform);

	return 0;
}
