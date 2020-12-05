#include "noise_skew.h"

#include "DataPointsFilters/NoiseSkew.h"

namespace pointmatcher
{
	void pybindNoiseSkew(py::module& p_module)
	{
		using NoiseSkewDataPointsFilter = NoiseSkewDataPointsFilter<ScalarType>;
		py::class_<NoiseSkewDataPointsFilter, std::shared_ptr<NoiseSkewDataPointsFilter>, DataPointsFilter>
				(p_module, "NoiseSkewDataPointsFilter")
				.def_static("description", &NoiseSkewDataPointsFilter::description)
				.def_static("availableParameters", &NoiseSkewDataPointsFilter::availableParameters)
				
				.def_readonly("skewModel", &NoiseSkewDataPointsFilter::skewModel)
				.def_readonly("rangePrecision", &NoiseSkewDataPointsFilter::rangePrecision)
				.def_readonly("linearSpeedNoisesX", &NoiseSkewDataPointsFilter::linearSpeedNoisesX)
				.def_readonly("linearSpeedNoisesY", &NoiseSkewDataPointsFilter::linearSpeedNoisesY)
				.def_readonly("linearSpeedNoiseZ", &NoiseSkewDataPointsFilter::linearSpeedNoisesZ)
				.def_readonly("linearAccelerationNoisesX", &NoiseSkewDataPointsFilter::linearAccelerationNoisesX)
				.def_readonly("linearAccelerationNoisesY", &NoiseSkewDataPointsFilter::linearAccelerationNoisesY)
				.def_readonly("linearAccelerationNoisesZ", &NoiseSkewDataPointsFilter::linearAccelerationNoisesZ)
				.def_readonly("angularSpeedNoisesX", &NoiseSkewDataPointsFilter::angularSpeedNoisesX)
				.def_readonly("angularSpeedNoisesY", &NoiseSkewDataPointsFilter::angularSpeedNoisesY)
				.def_readonly("angularSpeedNoisesZ", &NoiseSkewDataPointsFilter::angularSpeedNoisesZ)
				.def_readonly("angularAccelerationNoisesX", &NoiseSkewDataPointsFilter::angularAccelerationNoisesX)
				.def_readonly("angularAccelerationNoisesY", &NoiseSkewDataPointsFilter::angularAccelerationNoisesY)
				.def_readonly("angularAccelerationNoisesZ", &NoiseSkewDataPointsFilter::angularAccelerationNoisesZ)
				.def_readonly("measureTimes", &NoiseSkewDataPointsFilter::measureTimes)
				.def_readonly("cornerPointWeight", &NoiseSkewDataPointsFilter::cornerPointWeight)
				.def_readonly("weightQuantile", &NoiseSkewDataPointsFilter::weightQuantile)
				
				.def(py::init<const Parameters&>(), py::arg("params") = Parameters(), "Constructor, uses parameter interface")
				
				.def("filter", &NoiseSkewDataPointsFilter::filter, py::arg("input"))
				.def("inPlaceFilter", &NoiseSkewDataPointsFilter::inPlaceFilter, py::arg("cloud"));
	}
}
