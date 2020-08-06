#include "cutatdescriptorthreshold.h"

namespace pointmatcher
{
	void pybindCutAtDescriptorThreshold(py::module& p_module)
	{
		using CutAtDescriptorThresholdDataPointsFilter = CutAtDescriptorThresholdDataPointsFilter<double>;
		py::class_<CutAtDescriptorThresholdDataPointsFilter, std::shared_ptr<CutAtDescriptorThresholdDataPointsFilter>, DataPointsFilter>
			(p_module, "CutAtDescriptorThresholdDataPointsFilter")
			.def_static("description", &CutAtDescriptorThresholdDataPointsFilter::description)
			.def_static("availableParameters", &CutAtDescriptorThresholdDataPointsFilter::availableParameters)

			.def_readonly("descName", &CutAtDescriptorThresholdDataPointsFilter::descName)
			.def_readonly("useLargerThan", &CutAtDescriptorThresholdDataPointsFilter::useLargerThan)
			.def_readonly("threshold", &CutAtDescriptorThresholdDataPointsFilter::threshold)

			.def(py::init<const Parameters&>(), py::arg("params") = Parameters(), "Constructor, uses parameter interface")

			.def("filter", &CutAtDescriptorThresholdDataPointsFilter::filter, py::arg("input"))
			.def("inPlaceFilter", &CutAtDescriptorThresholdDataPointsFilter::inPlaceFilter, py::arg("cloud"));
	}
}