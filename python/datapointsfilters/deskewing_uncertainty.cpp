#include "deskewing_uncertainty.h"

#include "DataPointsFilters/DeskewingUncertainty.h"

namespace pointmatcher
{
	void pybindDeskewingUncertainty(py::module& p_module)
	{
		using DeskewingUncertaintyDataPointsFilter = DeskewingUncertaintyDataPointsFilter<ScalarType>;
		py::class_<DeskewingUncertaintyDataPointsFilter, std::shared_ptr<DeskewingUncertaintyDataPointsFilter>, DataPointsFilter>
				(p_module, "DeskewingUncertaintyDataPointsFilter")
				.def_static("description", &DeskewingUncertaintyDataPointsFilter::description)
				.def_static("availableParameters", &DeskewingUncertaintyDataPointsFilter::availableParameters)

				.def_readonly("motionGaussians", &DeskewingUncertaintyDataPointsFilter::motionGaussians)
				.def_readonly("measureTimes", &DeskewingUncertaintyDataPointsFilter::measureTimes)

				.def(py::init<const Parameters&>(), py::arg("params") = Parameters(), "Constructor, uses parameter interface")

				.def("filter", &DeskewingUncertaintyDataPointsFilter::filter, py::arg("input"))
				.def("inPlaceFilter", &DeskewingUncertaintyDataPointsFilter::inPlaceFilter, py::arg("cloud"));
	}
}
