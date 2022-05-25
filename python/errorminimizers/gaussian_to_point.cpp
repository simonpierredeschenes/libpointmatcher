#include "gaussian_to_point.h"

#include "pointmatcher/ErrorMinimizersImpl.h"

namespace pointmatcher
{
	void pybindGaussianToPoint(py::module& p_module)
	{
		using GaussianToPointErrorMinimizer = ErrorMinimizersImpl<ScalarType>::GaussianToPointErrorMinimizer;
		py::class_<GaussianToPointErrorMinimizer, std::shared_ptr<GaussianToPointErrorMinimizer>, ErrorMinimizer>(p_module, "GaussianToPointErrorMinimizer")
				.def(py::init<const Parameters&>(), py::arg("params") = Parameters())
				.def(py::init<const ParametersDoc, const Parameters&>(), py::arg("paramsDoc"), py::arg("params") = Parameters())

				.def_readonly("scaleFactor", &GaussianToPointErrorMinimizer::scaleFactor)

				.def_static("description", &GaussianToPointErrorMinimizer::description)
				.def_static("availableParameters", &GaussianToPointErrorMinimizer::availableParameters)

				.def("name", &GaussianToPointErrorMinimizer::name)
				.def("compute", &GaussianToPointErrorMinimizer::compute, py::arg("mPts"))
				.def("compute_in_place", &GaussianToPointErrorMinimizer::compute_in_place, py::arg("mPts"))
				.def("getResidualError", &GaussianToPointErrorMinimizer::getResidualError, py::arg("filteredReading"), py::arg("filteredReference"), py::arg("outlierWeights"), py::arg("matches"))
				.def("getOverlap", &GaussianToPointErrorMinimizer::getOverlap)
				.def("computeResidualError", &GaussianToPointErrorMinimizer::computeResidualError, py::arg("mPts"));
	}
}
