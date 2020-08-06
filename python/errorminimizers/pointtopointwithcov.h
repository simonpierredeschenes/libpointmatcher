#ifndef PYTHON_ERRORMINIMIZERS_POINTTOPOINTWITHCOV_H
#define PYTHON_ERRORMINIMIZERS_POINTTOPOINTWITHCOV_H

#include "pointmatcher/ErrorMinimizersImpl.h"
#include "pypointmatcher_helper.h"

namespace pointmatcher
{
	void pybindPointToPointWithCov(py::module& p_module);
}

#endif //PYTHON_ERRORMINIMIZERS_POINTTOPOINTWITHCOV_H
