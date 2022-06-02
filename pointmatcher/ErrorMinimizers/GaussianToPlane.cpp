/*************************************************************
  Generalized-ICP Copyright (c) 2009 Aleksandr Segal.
  All rights reserved.
  Redistribution and use in source and binary forms, with
  or without modification, are permitted provided that the
  following conditions are met:
* Redistributions of source code must retain the above
  copyright notice, this list of conditions and the
  following disclaimer.
* Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the
  following disclaimer in the documentation and/or other
  materials provided with the distribution.
* The names of the contributors may not be used to endorse
  or promote products derived from this software
  without specific prior written permission.
  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
  PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
  OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
  DAMAGE.
*************************************************************/

#include <Eigen/QR>
#include <Eigen/Eigenvalues>

#include "ErrorMinimizersImpl.h"
#include "PointMatcherPrivate.h"
#include "Functions.h"
#include "../DataPointsFilters/utils/utils.h"

#include "GICP/transform.h"
#include "GICP/optimize.h"

using namespace Eigen;
using namespace std;

typedef PointMatcherSupport::Parametrizable Parametrizable;
typedef PointMatcherSupport::Parametrizable P;
typedef Parametrizable::Parameters Parameters;
typedef Parametrizable::ParameterDoc ParameterDoc;
typedef Parametrizable::ParametersDoc ParametersDoc;

template<typename T>
GaussianToPlaneErrorMinimizer<T>::GaussianToPlaneErrorMinimizer(const Parameters& params):
		ErrorMinimizer(name(), availableParameters(), params),
		scaleFactor(Parametrizable::get<T>("scaleFactor"))
{
}

template<typename T>
GaussianToPlaneErrorMinimizer<T>::GaussianToPlaneErrorMinimizer(const ParametersDoc paramsDoc, const Parameters& params):
		ErrorMinimizer(name(), paramsDoc, params),
		scaleFactor(Parametrizable::get<T>("scaleFactor"))
{
}

template<typename T>
void convertDescriptorToGSLMatrix(const typename PointMatcher<T>::Vector& descriptor, gsl_matrix* GSLMatrix)
{
	for(unsigned int i = 0; i < GSLMatrix->size1; ++i)
	{
		for(unsigned int j = 0; j < GSLMatrix->size2; ++j)
		{
			gsl_matrix_set(GSLMatrix, i, j, descriptor(j * GSLMatrix->size1 + i));
		}
	}
}

template<typename T>
typename PointMatcher<T>::TransformationParameters convertDGCTransformToTransformationParameters(const dgc_transform_t& DGCTransform, unsigned int nbRows, unsigned int nbCols)
{
	typename PointMatcher<T>::TransformationParameters transformationParameters(nbRows, nbCols);
	for(unsigned int i = 0; i < nbRows; ++i)
	{
		for(unsigned int j = 0; j < nbCols; ++j)
		{
			transformationParameters(i, j) = DGCTransform[i][j];
		}
	}
	return transformationParameters;
}

template<typename T>
typename PointMatcher<T>::TransformationParameters GaussianToPlaneErrorMinimizer<T>::compute(const ErrorElements& mPts_const)
{
	if(!mPts_const.reading.descriptorExists("covariance"))
	{
		throw typename DataPoints::InvalidField("GaussianToPlaneErrorMinimizer: Error, no covariance found in reading descriptors.");
	}
	if(!mPts_const.reference.descriptorExists("normals"))
	{
		throw typename DataPoints::InvalidField("GaussianToPlaneErrorMinimizer: Error, no normals found in reference descriptors.");
	}
	ErrorElements mPts = mPts_const;
	const auto& covariances = mPts.reading.getDescriptorViewByName("covariance");
	const auto& normals = mPts.reference.getDescriptorViewByName("normals");

	int n = mPts.reading.getNbPoints();
	dgc_transform_t t;
	dgc_transform_identity(t);
	dgc_transform_t identity;
	dgc_transform_identity(identity);

	dgc::gicp::gicp_mat_t* mahalanobis = new dgc::gicp::gicp_mat_t[n];

	gsl_matrix* gsl_temp = gsl_matrix_alloc(3, 3);
	gsl_matrix* C1 = gsl_matrix_alloc(3, 3);
	gsl_matrix* C_env = gsl_matrix_alloc(3, 3);
	gsl_matrix* gsl_R = gsl_matrix_alloc(3, 3);

	/* set up the optimization parameters */
	dgc::gicp::GICPOptData<T> opt_data;
	opt_data.nn_indecies = &mPts.matches.ids;
	opt_data.p1 = &mPts.reading;
	opt_data.p2 = &mPts.reference;
	opt_data.M = mahalanobis;
	opt_data.solve_rotation = true;
	opt_data.num_matches = n;
	dgc_transform_copy(opt_data.base_t, identity);

	dgc::gicp::GICPOptimizer<T> opt;
	opt.SetDebug(false);
	opt.SetMaxIterations(10);
	/* set up the mahalanobis matrices */
	/* these are identity for now to ease debugging */
	for(int i = 0; i < n; i++)
	{
		for(int k = 0; k < 3; k++)
		{
			for(int l = 0; l < 3; l++)
			{
				mahalanobis[i][k][l] = (k == l) ? 1 : 0.;
			}
		}
	}

	for(int i = 0; i < n; i++)
	{
//		 set up the updated mahalanobis matrix here
		convertDescriptorToGSLMatrix<T>(covariances.col(i), C1);
		gsl_matrix_view M = gsl_matrix_view_array(&mahalanobis[i][0][0], 3, 3);
		gsl_matrix_set_zero(&M.matrix);

		// R = [n1, n2, n3]
		gsl_matrix_set(gsl_R, 0, 0, normals(0, i));
		gsl_matrix_set(gsl_R, 1, 0, normals(1, i));
		gsl_matrix_set(gsl_R, 2, 0, normals(2, i));
		if(std::fabs(normals(0, i)) < 0.57735)
		{
			gsl_matrix_set(gsl_R, 0, 1, 0);
			gsl_matrix_set(gsl_R, 1, 1, -normals(2, i) / std::sqrt(normals(1, i) * normals(1, i) + normals(2, i) * normals(2, i)));
			gsl_matrix_set(gsl_R, 2, 1, normals(1, i) / std::sqrt(normals(1, i) * normals(1, i) + normals(2, i) * normals(2, i)));
		}
		else if(std::fabs(normals(1, i)) < 0.57735)
		{
			gsl_matrix_set(gsl_R, 0, 1, -normals(2, i) / std::sqrt(normals(0, i) * normals(0, i) + normals(2, i) * normals(2, i)));
			gsl_matrix_set(gsl_R, 1, 1, 0);
			gsl_matrix_set(gsl_R, 2, 1, normals(0, i) / std::sqrt(normals(0, i) * normals(0, i) + normals(2, i) * normals(2, i)));
		}
		else
		{
			gsl_matrix_set(gsl_R, 0, 1, -normals(1, i) / std::sqrt(normals(0, i) * normals(0, i) + normals(1, i) * normals(1, i)));
			gsl_matrix_set(gsl_R, 1, 1, normals(0, i) / std::sqrt(normals(0, i) * normals(0, i) + normals(1, i) * normals(1, i)));
			gsl_matrix_set(gsl_R, 2, 1, 0);
		}
		gsl_matrix_set(gsl_R, 0, 2, gsl_matrix_get(gsl_R, 1, 0) * gsl_matrix_get(gsl_R, 2, 1) - gsl_matrix_get(gsl_R, 2, 0) * gsl_matrix_get(gsl_R, 1, 1));
		gsl_matrix_set(gsl_R, 1, 2, gsl_matrix_get(gsl_R, 2, 0) * gsl_matrix_get(gsl_R, 0, 1) - gsl_matrix_get(gsl_R, 0, 0) * gsl_matrix_get(gsl_R, 2, 1));
		gsl_matrix_set(gsl_R, 2, 2, gsl_matrix_get(gsl_R, 0, 0) * gsl_matrix_get(gsl_R, 1, 1) - gsl_matrix_get(gsl_R, 1, 0) * gsl_matrix_get(gsl_R, 0, 1));

		// C_env = [[1,0,0],[0,inf,0],[0,0,inf]]
		gsl_matrix_set_identity(C_env);
		gsl_matrix_set(C_env, 1, 1, 1e30);
		gsl_matrix_set(C_env, 2, 2, 1e30);

		// C_env = R * C_env * R^T
		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, gsl_R, C_env, 0.0, gsl_temp);
		gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, gsl_temp, gsl_R, 0.0, C_env);

		// temp = I * scaleFactor
		gsl_matrix_set_identity(gsl_temp);
		gsl_matrix_scale(gsl_temp, scaleFactor);

		// temp += C1
		gsl_matrix_add(gsl_temp, C1);

		// temp += C_env
		gsl_matrix_add(gsl_temp, C_env);

		// now invert temp to get the mahalanobis distance metric
		// M = temp^-1
		gsl_matrix_set_identity(&M.matrix);
		gsl_error_handler_t* error_handler = gsl_set_error_handler_off();
		int status = gsl_linalg_cholesky_decomp(gsl_temp);
		if(status != GSL_EDOM)
		{
			for(int k = 0; k < 3; k++)
			{
				gsl_vector_view row_view = gsl_matrix_row(&M.matrix, k);
				gsl_linalg_cholesky_svx(gsl_temp, &row_view.vector);
			}
		}
		gsl_set_error_handler(error_handler);
	}

	/* optimize transformation using the current assignment and Mahalanobis metrics*/
	opt.Optimize(t, opt_data);

	if(mahalanobis != NULL)
	{
		delete[] mahalanobis;
	}
	if(gsl_R != NULL)
	{
		gsl_matrix_free(gsl_R);
	}
	if(gsl_temp != NULL)
	{
		gsl_matrix_free(gsl_temp);
	}
	if(C1 != NULL)
	{
		gsl_matrix_free(C1);
	}
	if(C_env != NULL)
	{
		gsl_matrix_free(C_env);
	}

	return convertDGCTransformToTransformationParameters<T>(t, 4, 4);
}

template
struct GaussianToPlaneErrorMinimizer<float>;
template
struct GaussianToPlaneErrorMinimizer<double>;