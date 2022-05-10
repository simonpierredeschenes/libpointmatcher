#include <pointmatcher/PointMatcher.h>

typedef PointMatcher<float> PM;

typename PM::DataPoints generateReading()
{
	PM::DataPoints cloud(PM::DataPoints::load("/home/norlab/repos/libpointmatcher/examples/data/cloud.00000.vtk"));

	PM::Int64Matrix stamps = PM::Int64Matrix::Zero(1, cloud.getNbPoints());
	for(unsigned int i = 0; i < cloud.getNbPoints(); ++i)
	{
		stamps(0, i) = (cloud.getNbPoints() - 1) * 1e8 / (cloud.getNbPoints() - 1);
	}
	stamps(0, 0) = 0;
	cloud.addTime("stamps", stamps);

	PM::Parameters params;
	params["linearSpeedsX"] = "1";
	params["linearSpeedsY"] = "1";
	params["linearSpeedsZ"] = "1";
	params["angularSpeedsX"] = "0";
	params["angularSpeedsY"] = "0";
	params["angularSpeedsZ"] = "0";
	params["measureTimes"] = "0";
	std::shared_ptr<PM::DataPointsFilter> deskewingFilter = PM::get().DataPointsFilterRegistrar.create("DeskewingUncertaintyDataPointsFilter", params);
	return deskewingFilter->filter(cloud);
}

typename PM::DataPoints generateReference()
{
	PM::DataPoints cloud(PM::DataPoints::load("/home/norlab/repos/libpointmatcher/examples/data/cloud.00001.vtk"));
	PM::Parameters params;
	params["knn"] = "20";
	std::shared_ptr<PM::DataPointsFilter> normalFilter = PM::get().DataPointsFilterRegistrar.create("SurfaceNormalDataPointsFilter", params);
	return normalFilter->filter(cloud);;
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
	std::shared_ptr<PM::ErrorMinimizer> gaussianToPlaneMinimizer = PM::get().ErrorMinimizerRegistrar.create("GaussianToPlaneErrorMinimizer");
	icp.errorMinimizer = gaussianToPlaneMinimizer;
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
//	params.clear();
//	params["baseFileName"] = "/home/norlab/Desktop/inspector_output/test";
//	params["dumpDataLinks"] = "1";
//	params["dumpReading"] = "1";
//	params["dumpReference"] = "1";
//	std::shared_ptr<PM::Inspector> inspector = PM::get().InspectorRegistrar.create("VTKFileInspector", params);
//	icp.inspector = inspector;
	std::shared_ptr<PM::Inspector> inspector = PM::get().InspectorRegistrar.create("NullInspector");
	icp.inspector = inspector;

	PM::TransformationParameters optimalTransform = icp(reading, reference);
	icp.transformations.apply(reading, optimalTransform);

	reading.save("/home/norlab/Desktop/reading.vtk");
	reference.save("/home/norlab/Desktop/reference.vtk");

	return 0;
}
