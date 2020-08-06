#include "fixstepsampling.h"

namespace pointmatcher
{
	void pybindFixStepSampling(py::module& p_module)
	{
		using FixStepSamplingDataPointsFilter = FixStepSamplingDataPointsFilter<double>;
		py::class_<FixStepSamplingDataPointsFilter, std::shared_ptr<FixStepSamplingDataPointsFilter>, DataPointsFilter>
			(p_module, "FixStepSamplingDataPointsFilter")
			.def_static("description", &FixStepSamplingDataPointsFilter::description)
			.def_static("availableParameters", &FixStepSamplingDataPointsFilter::availableParameters)

			.def_readonly("startStep", &FixStepSamplingDataPointsFilter::startStep)
			.def_readonly("endStep", &FixStepSamplingDataPointsFilter::endStep)
			.def_readonly("stepMult", &FixStepSamplingDataPointsFilter::stepMult)

			.def(py::init<const Parameters&>(), py::arg("params") = Parameters(), "Constructor, uses parameter interface")

			.def("init", &FixStepSamplingDataPointsFilter::init)
			.def("filter", &FixStepSamplingDataPointsFilter::filter, py::arg("input"))
			.def("inPlaceFilter", &FixStepSamplingDataPointsFilter::inPlaceFilter, py::arg("cloud"));
	}
}