#include <iostream>
#include <pointmatcher/PointMatcher.h>

typedef PointMatcher<float> PM;

typename PM::DataPoints generateReading()
{
    PM::Matrix features = PM::Matrix::Zero(4, 10);
    for(unsigned int i = 0; i < features.cols(); ++i)
    {
        double angle = (2.0 * M_PI * i / (features.cols() - 1)) - M_PI/8;
        features.col(i) = (PM::Vector(4) << std::cos(angle) * 3, std::sin(angle) * 3, i / 10, 1).finished();
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

    PM::Parameters params;
    params["skewModel"] = "1";
    params["linearSpeedsX"] = "10";
    params["linearSpeedsY"] = "10";
    params["linearSpeedsZ"] = "10";
    params["angularSpeedsX"] = "0";
    params["angularSpeedsY"] = "0";
    params["angularSpeedsZ"] = "0";
    params["measureTimes"] = "0";
    std::shared_ptr<PM::DataPointsFilter> deskewingFilter = PM::get().DataPointsFilterRegistrar.create("DeskewingUncertaintyDataPointsFilter", params);
    std::shared_ptr<PM::DataPointsFilter> surfaceNormalFilter = PM::get().DataPointsFilterRegistrar.create("SurfaceNormalDataPointsFilter");
    surfaceNormalFilter->inPlaceFilter(cloud);
    return deskewingFilter->filter(cloud);

}

typename PM::DataPoints generateReference()
{
    PM::Matrix features = PM::Matrix::Zero(4, 10);
    for(unsigned int i = 0; i < features.cols(); ++i)
    {
        double angle = 2.0 * M_PI * i / (features.cols() - 1);
        features.col(i) = (PM::Vector(4) << std::cos(angle) * 2, std::sin(angle) * 2, i / 10, 1).finished();
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
    std::shared_ptr<PM::ErrorMinimizer> pointToGaussianMinimizer = PM::get().ErrorMinimizerRegistrar.create("PointToGaussianErrorMinimizer");
    icp.errorMinimizer = pointToGaussianMinimizer;


//    std::shared_ptr<PM::Parametrizable> matcher_params;
    PM::Parameters matcher_params;
    matcher_params["maxDist"] = "100.0";
    matcher_params["knn"] = "10";
    std::shared_ptr<PM::Matcher> matcher = PM::get().MatcherRegistrar.create("KDTreeMatcher", matcher_params);
    matcher->init(reference);
    icp.matcher = matcher;
    std::shared_ptr<PM::Inspector> nullInspector = PM::get().InspectorRegistrar.create("NullInspector");
    icp.inspector = nullInspector;
    std::shared_ptr<PM::TransformationChecker> checker = PM::get().TransformationCheckerRegistrar.create("CounterTransformationChecker");
    icp.transformationCheckers.push_back(checker);
    std::shared_ptr<PM::Transformation> transformation = PM::get().TransformationRegistrar.create("RigidTransformation");
    icp.transformations.push_back(transformation);

    std::shared_ptr<PM::DataPointsFilter> surfaceNormalFilter = PM::get().DataPointsFilterRegistrar.create("SurfaceNormalDataPointsFilter");
    surfaceNormalFilter->filter(reference);

    PM::TransformationParameters optimalTransform = icp(reading, reference);
    icp.transformations.apply(reading, optimalTransform);

    reading.save("/home/dominic/repos/tests_point_to_gaussian/toy_reading.vtk");
    reference.save("/home/dominic/repos/tests_point_to_gaussian/toy_reference.vtk");

	return 0;
}
