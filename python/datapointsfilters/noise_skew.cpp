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
				.def_readonly("linearSpeedNoiseX", &NoiseSkewDataPointsFilter::linearSpeedNoiseX)
				.def_readonly("linearSpeedNoiseY", &NoiseSkewDataPointsFilter::linearSpeedNoiseY)
				.def_readonly("linearSpeedNoiseZ", &NoiseSkewDataPointsFilter::linearSpeedNoiseZ)
				.def_readonly("linearAccelerationNoiseX", &NoiseSkewDataPointsFilter::linearAccelerationNoiseX)
				.def_readonly("linearAccelerationNoiseY", &NoiseSkewDataPointsFilter::linearAccelerationNoiseY)
				.def_readonly("linearAccelerationNoiseZ", &NoiseSkewDataPointsFilter::linearAccelerationNoiseZ)
				.def_readonly("angularSpeedNoiseX", &NoiseSkewDataPointsFilter::angularSpeedNoiseX)
				.def_readonly("angularSpeedNoiseY", &NoiseSkewDataPointsFilter::angularSpeedNoiseY)
				.def_readonly("angularSpeedNoiseZ", &NoiseSkewDataPointsFilter::angularSpeedNoiseZ)
				.def_readonly("angularAccelerationNoiseX", &NoiseSkewDataPointsFilter::angularAccelerationNoiseX)
				.def_readonly("angularAccelerationNoiseY", &NoiseSkewDataPointsFilter::angularAccelerationNoiseY)
				.def_readonly("angularAccelerationNoiseZ", &NoiseSkewDataPointsFilter::angularAccelerationNoiseZ)
				.def_readonly("cornerPointWeight", &NoiseSkewDataPointsFilter::cornerPointWeight)
				.def_readonly("weightQuantile", &NoiseSkewDataPointsFilter::weightQuantile)
				
				.def(py::init<const Parameters&>(), py::arg("params") = Parameters(), "Constructor, uses parameter interface")
				
				.def("filter", &NoiseSkewDataPointsFilter::filter, py::arg("input"))
				.def("inPlaceFilter", &NoiseSkewDataPointsFilter::inPlaceFilter, py::arg("cloud"));
	}
}
