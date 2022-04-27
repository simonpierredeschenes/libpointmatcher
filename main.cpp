#include <iostream>
#include <pointmatcher/PointMatcher.h>

typedef PointMatcher<float> PM;

int main(int argc, char** argv)
{
	PM::Matrix features = PM::Matrix::Zero(4, 10);
	for(unsigned int i = 0; i < features.cols(); ++i)
	{
		features.col(i) = (PM::Vector(4) << i, 0, 0, 1).finished();
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
	params["linearSpeedsX"] = "10,10,10,10,10,10,10,10,10,10";
	params["linearSpeedsY"] = "0,0,0,0,0,0,0,0,0,0";
	params["linearSpeedsZ"] = "0,0,0,0,0,0,0,0,0,0";
	params["angularSpeedsX"] = "0,0,0,0,0,0,0,0,0,0";
	params["angularSpeedsY"] = "0,0,0,0,0,0,0,0,0,0";
	params["angularSpeedsZ"] = "0,0,0,0,0,0,0,0,0,0";
	params["measureTimes"] = "0,1,2,3,4,5,6,7,8,9";
	std::shared_ptr<PM::DataPointsFilter> deskewingFilter = PM::get().DataPointsFilterRegistrar.create("DeskewingUncertaintyDataPointsFilter", params);
	deskewingFilter->inPlaceFilter(cloud);
	cloud.save("/home/norlab/Desktop/test_cloud.vtk");

	return 0;
}
