#include "gaussian_to_gaussian_to_plane.h"

#include "pointmatcher/ErrorMinimizersImpl.h"

namespace pointmatcher
{
	void pybindGaussianToGaussianToPlane(py::module& p_module)
	{
		using GaussianToGaussianToPlaneErrorMinimizer = ErrorMinimizersImpl<ScalarType>::GaussianToGaussianToPlaneErrorMinimizer;
		py::class_<GaussianToGaussianToPlaneErrorMinimizer, std::shared_ptr<GaussianToGaussianToPlaneErrorMinimizer>, ErrorMinimizer>(p_module, "GaussianToGaussianToPlaneErrorMinimizer")
				.def(py::init<const Parameters&>(), py::arg("params") = Parameters())
				.def(py::init<const ParametersDoc, const Parameters&>(), py::arg("paramsDoc"), py::arg("params") = Parameters())

				.def_readonly("scaleFactor", &GaussianToGaussianToPlaneErrorMinimizer::scaleFactor)

				.def_static("description", &GaussianToGaussianToPlaneErrorMinimizer::description)
				.def_static("availableParameters", &GaussianToGaussianToPlaneErrorMinimizer::availableParameters)

				.def("name", &GaussianToGaussianToPlaneErrorMinimizer::name)
				.def("compute", &GaussianToGaussianToPlaneErrorMinimizer::compute, py::arg("mPts"));
	}
}
