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
				.def_readonly("linearSpeedNoise", &NoiseSkewDataPointsFilter::linearSpeedNoise)
				.def_readonly("linearAccelerationNoise", &NoiseSkewDataPointsFilter::linearAccelerationNoise)
				.def_readonly("angularSpeedNoise", &NoiseSkewDataPointsFilter::angularSpeedNoise)
				.def_readonly("angularAccelerationNoise", &NoiseSkewDataPointsFilter::angularAccelerationNoise)
				.def_readonly("cornerPointWeight", &NoiseSkewDataPointsFilter::cornerPointWeight)
				.def_readonly("weightQuantile", &NoiseSkewDataPointsFilter::weightQuantile)
				
				.def(py::init<const Parameters&>(), py::arg("params") = Parameters(), "Constructor, uses parameter interface")
				
				.def("filter", &NoiseSkewDataPointsFilter::filter, py::arg("input"))
				.def("inPlaceFilter", &NoiseSkewDataPointsFilter::inPlaceFilter, py::arg("cloud"));
	}
}
