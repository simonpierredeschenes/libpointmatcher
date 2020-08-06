#include "identity.h"

namespace pointmatcher
{
	void pybindIdentityDPF(py::module& p_module)
	{
		using IdentityDataPointsFilter = IdentityDataPointsFilter<double>;
		py::class_<IdentityDataPointsFilter, std::shared_ptr<IdentityDataPointsFilter>, DataPointsFilter>
			(p_module, "IdentityDataPointsFilter")
			.def_static("description", &IdentityDataPointsFilter::description)

			.def(py::init<>())

			.def("filter", &IdentityDataPointsFilter::filter, py::arg("input"))
			.def("inPlaceFilter", &IdentityDataPointsFilter::inPlaceFilter, py::arg("cloud"));
	}
}