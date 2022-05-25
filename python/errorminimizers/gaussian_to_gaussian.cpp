#include "gaussian_to_gaussian.h"

#include "pointmatcher/ErrorMinimizersImpl.h"

namespace pointmatcher
{
	void pybindGaussianToGaussian(py::module& p_module)
	{
		using GaussianToGaussianErrorMinimizer = ErrorMinimizersImpl<ScalarType>::GaussianToGaussianErrorMinimizer;
		py::class_<GaussianToGaussianErrorMinimizer, std::shared_ptr<GaussianToGaussianErrorMinimizer>, ErrorMinimizer>(p_module, "GaussianToGaussianErrorMinimizer")
				.def(py::init<const Parameters&>(), py::arg("params") = Parameters())
				.def(py::init<const ParametersDoc, const Parameters&>(), py::arg("paramsDoc"), py::arg("params") = Parameters())

				.def_readonly("scaleFactor", &GaussianToGaussianErrorMinimizer::scaleFactor)

				.def_static("description", &GaussianToGaussianErrorMinimizer::description)
				.def_static("availableParameters", &GaussianToGaussianErrorMinimizer::availableParameters)

				.def("name", &GaussianToGaussianErrorMinimizer::name)
				.def("compute", &GaussianToGaussianErrorMinimizer::compute, py::arg("mPts"))
				.def("compute_in_place", &GaussianToGaussianErrorMinimizer::compute_in_place, py::arg("mPts"))
				.def("getResidualError", &GaussianToGaussianErrorMinimizer::getResidualError, py::arg("filteredReading"), py::arg("filteredReference"), py::arg("outlierWeights"), py::arg("matches"))
				.def("getOverlap", &GaussianToGaussianErrorMinimizer::getOverlap)
				.def("computeResidualError", &GaussianToGaussianErrorMinimizer::computeResidualError, py::arg("mPts"));
	}
}
