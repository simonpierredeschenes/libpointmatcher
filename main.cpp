#include "PointMatcher.h"

typedef PointMatcher<float> PM;

int main(int argc, char** argv)
{
//	PM::DataPoints reference(PM::DataPoints::load("/home/simon/Desktop/reference.vtk"));
//	reference.removeFeature("z");
//	PM::DataPoints reading(PM::DataPoints::load("/home/simon/Desktop/reading.vtk"));
//	reading.removeFeature("z");
	
	PM::DataPoints reading(PM::DataPoints::load("/home/simon/Desktop/2020-10-21-18-03-27.vtk"));
	reading.addTime("stamps", reading.getDescriptorViewByName("t").cast<std::int64_t>());
	
	PM::Parameters boundingBoxFilterParams;
	boundingBoxFilterParams["xMin"] = "-0.1";
	boundingBoxFilterParams["xMax"] = "0.1";
	boundingBoxFilterParams["yMin"] = "-0.1";
	boundingBoxFilterParams["yMax"] = "0.1";
	boundingBoxFilterParams["zMin"] = "-0.1";
	boundingBoxFilterParams["zMax"] = "0.1";
	boundingBoxFilterParams["removeInside"] = "1";
	std::shared_ptr<PM::DataPointsFilter>
			boundingBoxFilter = PM::get().DataPointsFilterRegistrar.create("BoundingBoxDataPointsFilter", boundingBoxFilterParams);
	reading = boundingBoxFilter->filter(reading);
	
	PM::Parameters normalFilterParams;
	normalFilterParams["knn"] = "12";
	std::shared_ptr<PM::DataPointsFilter> normalFilter = PM::get().DataPointsFilterRegistrar.create("SurfaceNormalDataPointsFilter", normalFilterParams);
	reading = normalFilter->filter(reading);
	
	std::shared_ptr<PM::DataPointsFilter> observationDirectionFilter = PM::get().DataPointsFilterRegistrar.create("ObservationDirectionDataPointsFilter");
	reading = observationDirectionFilter->filter(reading);
	
	PM::Parameters noiseSkewFilterParams;
	noiseSkewFilterParams["skewModel"] = "0";
	noiseSkewFilterParams["linearSpeedNoise"] = "0";
	noiseSkewFilterParams["linearAccelerationNoise"] = "0";
	noiseSkewFilterParams["angularSpeedNoise"] = "6.28";
	noiseSkewFilterParams["angularAccelerationNoise"] = "0";
	noiseSkewFilterParams["cornerPointWeight"] = "0";
	noiseSkewFilterParams["weightQuantile"] = "0";
	std::shared_ptr<PM::DataPointsFilter> noiseSkewFilter = PM::get().DataPointsFilterRegistrar.create("NoiseSkewDataPointsFilter", noiseSkewFilterParams);
	reading = noiseSkewFilter->filter(reading);
	reading.save("/home/simon/Desktop/skew_weights_ang_velocity.vtk");
	
	return 0;
}
