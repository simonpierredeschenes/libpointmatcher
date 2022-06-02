#include "gaussian_to_plane.h"

#include "pointmatcher/ErrorMinimizersImpl.h"

namespace pointmatcher
{
	void pybindGaussianToPlane(py::module& p_module)
	{
		using GaussianToPlaneErrorMinimizer = ErrorMinimizersImpl<ScalarType>::GaussianToPlaneErrorMinimizer;
		py::class_<GaussianToPlaneErrorMinimizer, std::shared_ptr<GaussianToPlaneErrorMinimizer>, ErrorMinimizer>(p_module, "GaussianToPlaneErrorMinimizer")
				.def(py::init<const Parameters&>(), py::arg("params") = Parameters())
				.def(py::init<const ParametersDoc, const Parameters&>(), py::arg("paramsDoc"), py::arg("params") = Parameters())

				.def_readonly("scaleFactor", &GaussianToPlaneErrorMinimizer::scaleFactor)

				.def_static("description", &GaussianToPlaneErrorMinimizer::description)
				.def_static("availableParameters", &GaussianToPlaneErrorMinimizer::availableParameters)

				.def("name", &GaussianToPlaneErrorMinimizer::name)
				.def("compute", &GaussianToPlaneErrorMinimizer::compute, py::arg("mPts"));
	}
}
