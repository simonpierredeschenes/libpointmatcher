#include <pointmatcher/PointMatcher.h>
#include <unsupported/Eigen/MatrixFunctions>

template<typename T>
typename PointMatcher<T>::Matrix expm(const typename PointMatcher<T>::Matrix& matrix)
{
	if(matrix.rows() != matrix.cols())
	{
		throw std::runtime_error("expm: matrix should be a square skew-symmetric matrix, actual size is " + std::to_string(matrix.rows()) + " x " + std::to_string(matrix.cols()));
	}
	return matrix.exp();
}

template<typename T>
typename PointMatcher<T>::Matrix sqrtm(const typename PointMatcher<T>::Matrix& matrix)
{
	if(matrix.rows() != matrix.cols())
	{
		throw std::runtime_error("sqrtm: matrix should be a square matrix, actual size is " + std::to_string(matrix.rows()) + " x " + std::to_string(matrix.cols()));
	}
	return matrix.sqrt();
}

template<typename T>
typename PointMatcher<T>::Matrix wedge_so3(const typename PointMatcher<T>::Vector& vector)
{
	if(vector.size() != 3)
	{
		throw std::runtime_error("wedge_so3: vector should be of size 3 and not " + std::to_string(vector.size()));
	}
	typename PointMatcher<T>::Matrix result = PointMatcher<T>::Matrix::Zero(3, 3);
	result(0, 1) = -vector(2);
	result(0, 2) = vector(1);
	result(1, 0) = vector(2);
	result(1, 2) = -vector(0);
	result(2, 0) = -vector(1);
	result(2, 1) = vector(0);
	return result;
}

template<typename T>
T cotan(const T& scalar)
{
	return T(1.0) / std::tan(scalar);
}

template<typename T>
typename PointMatcher<T>::Matrix xxT(const typename PointMatcher<T>::Vector& vector)
{
	return vector * vector.transpose();
}

template<typename T>
typename PointMatcher<T>::Matrix wedge_se3(const typename PointMatcher<T>::Vector& vector)
{
	if(vector.size() != 6)
	{
		throw std::runtime_error("wedge_se3: vector should be of size 6 and not " + std::to_string(vector.size()));
	}
	typename PointMatcher<T>::Matrix result = PointMatcher<T>::Matrix::Zero(4, 4);
	result.topLeftCorner(3, 3) = wedge_so3<T>(vector.tail(3));
	result.topRightCorner(3, 1) = vector.head(3);
	return result;
}

template<typename T>
typename PointMatcher<T>::Matrix curlyWedge_se3(const typename PointMatcher<T>::Vector& vector)
{
	if(vector.size() != 6)
	{
		throw std::runtime_error("curlyWedge_se3: vector should be of size 6 and not " + std::to_string(vector.size()));
	}
	typename PointMatcher<T>::Matrix result = PointMatcher<T>::Matrix::Zero(6, 6);
	result.topLeftCorner(3, 3) = wedge_so3<T>(vector.tail(3));
	result.topRightCorner(3, 3) = wedge_so3<T>(vector.head(3));
	result.bottomRightCorner(3, 3) = wedge_so3<T>(vector.tail(3));
	return result;
}

template<typename T>
typename PointMatcher<T>::Matrix jacobianInverse_so3(const typename PointMatcher<T>::Vector& vector)
{
	if(vector.size() != 3)
	{
		throw std::runtime_error("jacobianInverse_so3: vector should be of size 3 and not " + std::to_string(vector.size()));
	}
	if(vector.norm() < 1e-3)
	{
		return PointMatcher<T>::Matrix::Identity(3, 3);
	}
	typename PointMatcher<T>::Vector axis = vector.normalized();
	T angle = vector.norm();
	return angle / 2. * cotan<T>(angle / 2.f) * PointMatcher<T>::Matrix::Identity(3, 3) + (1 - angle / 2. * cotan<T>(angle / 2.f)) * xxT<T>(axis) - angle / 2.f * wedge_so3<T>(axis);
}

template<typename T>
typename PointMatcher<T>::Matrix jacobianInverse_se3(const typename PointMatcher<T>::Vector& vector)
{
	if(vector.size() != 6)
	{
		throw std::runtime_error("jacobianInverse_se3: vector should be of size 6 and not " + std::to_string(vector.size()));
	}
	return PointMatcher<T>::Matrix::Identity(6, 6) - 0.5 * curlyWedge_se3<T>(vector);
}


template<typename T>
struct Gaussian
{
	typename PointMatcher<T>::Vector mean;
	typename PointMatcher<T>::Matrix covariance;
};

template<typename T>
typename PointMatcher<T>::Vector propagateSigmaPoint(const typename PointMatcher<T>::Vector& point, const typename PointMatcher<T>::Vector& sigmaPoint)
{
	return expm<T>(wedge_se3<T>(sigmaPoint)) * point;
}

template<typename T>
Gaussian<T> propagateUncertainty(const Gaussian<T>& poseGaussian, const typename PointMatcher<T>::Vector& point, const T& kappa = 0)
{
	typename PointMatcher<T>::Vector omega = PointMatcher<T>::Vector::Zero(2 * 6 + 1);
	omega(0) = kappa / (6 + kappa);
	for(unsigned i = 1; i <= 6; ++i)
	{
		omega(i) = 1.0f / (2.0f * (6 + kappa));
		omega(i + 6) = 1.0f / (2.0f * (6 + kappa));
	}
	typename PointMatcher<T>::Matrix chi = PointMatcher<T>::Matrix::Zero(6, 2 * 6 + 1);
	chi.col(0) = poseGaussian.mean;
	for(unsigned int i = 1; i <= 6; ++i)
	{
		chi.col(i) = poseGaussian.mean + sqrtm<T>((6 + kappa) * poseGaussian.covariance).col(i - 1);
	}
	for(unsigned int i = 6 + 1; i <= 2 * 6; ++i)
	{
		chi.col(i) = poseGaussian.mean - sqrtm<T>((6 + kappa) * poseGaussian.covariance).col(i - 6 - 1);
	}
	typename PointMatcher<T>::Matrix gamma = PointMatcher<T>::Matrix::Zero(3, 2 * 6 + 1);
	for(unsigned int i = 0; i <= 2 * 6; ++i)
	{
		gamma.col(i) = propagateSigmaPoint<T>(point, chi.col(i)).head(3);
	}
	Gaussian<T> result;
	result.mean = PointMatcher<T>::Vector::Zero(3);
	for(unsigned int i = 0; i <= 2 * 6; ++i)
	{
		result.mean += omega(i) * gamma.col(i);
	}
	result.covariance = PointMatcher<T>::Matrix::Zero(3, 3);
	for(unsigned int i = 0; i <= 2 * 6; ++i)
	{
		result.covariance += omega(i) * xxT<T>(gamma.col(i) - result.mean);
	}
	return result;
}

template<typename T>
Gaussian<T> addMotion(const Gaussian<T>& previousPoseGaussian, const Gaussian<T>& motionGaussian, const T& deltaTime)
{
	typename PointMatcher<T>::Matrix previousOrientationInverseJacobian = jacobianInverse_so3<T>(previousPoseGaussian.mean.tail(3));
	typename PointMatcher<T>::Vector motionMean_se3 = motionGaussian.mean;
	motionMean_se3.head(3) = previousOrientationInverseJacobian * motionGaussian.mean.head(3);
	typename PointMatcher<T>::Matrix linearVelocityCovariance_se3 = previousOrientationInverseJacobian * motionGaussian.covariance.topLeftCorner(3, 3) * previousOrientationInverseJacobian.transpose();
	typename PointMatcher<T>::Matrix motionCovariance_se3 = motionGaussian.covariance;
	motionCovariance_se3.topLeftCorner(3, 3) = linearVelocityCovariance_se3;
	typename PointMatcher<T>::Matrix previousPoseInverseJacobian = jacobianInverse_se3<T>(previousPoseGaussian.mean);
	Gaussian<T> poseGaussian;
	poseGaussian.mean = previousPoseGaussian.mean + previousPoseInverseJacobian * motionMean_se3 * deltaTime;
	poseGaussian.covariance = previousPoseGaussian.covariance + previousPoseInverseJacobian * motionCovariance_se3 * previousPoseInverseJacobian.transpose() * deltaTime * deltaTime;
	return poseGaussian;
}
