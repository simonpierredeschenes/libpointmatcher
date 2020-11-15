#include "surface_curvature.h"

#include "DataPointsFilters/SurfaceCurvature.h"

namespace pointmatcher
{
	void pybindSurfaceCurvature(py::module& p_module)
	{
		using SurfaceCurvatureDataPointsFilter = SurfaceCurvatureDataPointsFilter<ScalarType>;
		py::class_<SurfaceCurvatureDataPointsFilter, std::shared_ptr<SurfaceCurvatureDataPointsFilter>, DataPointsFilter>
				(p_module, "SurfaceCurvatureDataPointsFilter")
				.def_static("description", &SurfaceCurvatureDataPointsFilter::description)
				.def_static("availableParameters", &SurfaceCurvatureDataPointsFilter::availableParameters)
				
				.def_readonly("knn", &SurfaceCurvatureDataPointsFilter::knn)
				.def_readonly("maxDist", &SurfaceCurvatureDataPointsFilter::maxDist)
				.def_readonly("epsilon", &SurfaceCurvatureDataPointsFilter::epsilon)
				
				.def(py::init<const Parameters&>(), py::arg("params") = Parameters(), "Constructor, uses parameter interface")
				
				.def("filter", &SurfaceCurvatureDataPointsFilter::filter, py::arg("input"))
				.def("inPlaceFilter", &SurfaceCurvatureDataPointsFilter::inPlaceFilter, py::arg("cloud"));
	}
}
