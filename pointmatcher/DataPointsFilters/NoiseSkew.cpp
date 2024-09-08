#include "NoiseSkew.h"
#include <numeric>
#include <boost/lexical_cast.hpp>
#include <boost/math/distributions/lognormal.hpp>

template<typename Func>
struct lambda_as_visitor_wrapper: Func
{
    lambda_as_visitor_wrapper(const Func& f):
            Func(f)
    {
    }

    template<typename S, typename I>
    void init(const S& v, I i, I j)
    {
        return Func::operator()(v, i, j);
    }
};

template<typename Mat, typename Func>
void visit_lambda(const Mat& m, const Func& f)
{
    lambda_as_visitor_wrapper<Func> visitor(f);
    m.visit(visitor);
}

template<typename T>
typename NoiseSkewDataPointsFilter<T>::Array NoiseSkewDataPointsFilter<T>::castToLinearSpeedNoises(const std::string& values, bool afterDeskewing)
{
    Array linearSpeeds = castToArray(values).abs();
    Array linearSpeedNoises = Array::Zero(1, linearSpeeds.cols());
    if(afterDeskewing)
    {
        auto distribution = boost::math::lognormal_distribution<T>(0.0, 1.1);
        for(int i = 0; i < linearSpeeds.cols(); i++)
        {
            linearSpeedNoises(0, i) = boost::math::pdf(distribution, linearSpeeds(0, i) / 1.9) / (1.9 * 4.5);
        }
    }
    else
    {
        for(int i = 0; i < linearSpeeds.cols(); i++)
        {
            linearSpeedNoises(0, i) = 0.14 * linearSpeeds(0, i);
        }
    }
    return linearSpeedNoises;
}

template<typename T>
typename NoiseSkewDataPointsFilter<T>::Array NoiseSkewDataPointsFilter<T>::castToLinearAccelerationNoises(const std::string& values, bool afterDeskewing)
{
    return castToArray(values).abs() * 0.0;
}

template<typename T>
typename NoiseSkewDataPointsFilter<T>::Array NoiseSkewDataPointsFilter<T>::castToAngularSpeedNoises(const std::string& values, bool afterDeskewing)
{
    Array angularSpeeds = castToArray(values).abs();
    Array angularSpeedNoises = Array::Zero(1, angularSpeeds.cols());
    if(afterDeskewing)
    {
        for(int i = 0; i < angularSpeeds.cols(); i++)
        {
            angularSpeedNoises(0, i) = std::pow(angularSpeeds(0, i) / 16.0, 3);
        }
    }
    else
    {
        for(int i = 0; i < angularSpeeds.cols(); i++)
        {
            angularSpeedNoises(0, i) = 0.72 * angularSpeeds(0, i);
        }
    }
    return angularSpeedNoises;
}

template<typename T>
typename NoiseSkewDataPointsFilter<T>::Array NoiseSkewDataPointsFilter<T>::castToAngularAccelerationNoises(const std::string& values, bool afterDeskewing)
{
    return castToArray(values).abs() * 0.0;
}

template<typename T>
typename NoiseSkewDataPointsFilter<T>::Array NoiseSkewDataPointsFilter<T>::castToArray(const std::string& values)
{
    size_t lastPos = 0;
    size_t pos;
    std::vector<T> vector;
    while((pos = values.find(',', lastPos)) != std::string::npos)
    {
        vector.push_back(boost::lexical_cast<T>(values.substr(lastPos, pos - lastPos)));
        lastPos = pos + 1;
    }
    if(!values.empty())
    {
        vector.push_back(boost::lexical_cast<T>(values.substr(lastPos)));
    }

    Array array = Array::Zero(1, vector.size());
    for(int i = 0; i < vector.size(); i++)
    {
        array(0, i) = vector[i];
    }
    return array;
}

// https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
template<typename T>
template<typename U>
std::vector<int> NoiseSkewDataPointsFilter<T>::computeOrdering(const Eigen::Matrix<U, 1, Eigen::Dynamic>& elements)
{
    std::vector<int> indices(elements.cols());
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(), [&elements](int index1, int index2){ return elements(0, index1) < elements(0, index2); });
    return indices;
}

template<typename T>
void NoiseSkewDataPointsFilter<T>::applyOrdering(const std::vector<int>& ordering, Eigen::Array<int, 1, Eigen::Dynamic>& idTable,
                                                 DataPoints& dataPoints)
{
    std::vector<int> newIndices(ordering.size());
    std::iota(newIndices.begin(), newIndices.end(), 0);
    for(size_t i = 0; i < ordering.size(); i++)
    {
        int indexToSwap = newIndices[ordering[i]];
        idTable.col(i).swap(idTable.col(indexToSwap));
        dataPoints.swapCols(i, indexToSwap);
        int tempIndex = newIndices[i];
        newIndices[newIndices[i]] = indexToSwap;
        newIndices[indexToSwap] = tempIndex;
    }
}

template<typename T>
typename NoiseSkewDataPointsFilter<T>::Array NoiseSkewDataPointsFilter<T>::computeTranslations(const Array& linearSpeeds, const Array& linearAccelerations,
                                                                                               const Array& times, const Array& firingDelays)
{
    int timeIndex = 0;
    T currentLinearSpeed = linearSpeeds(0, 0);
    T currentLinearAcceleration = linearAccelerations(0, 0);
    Array positions = Array::Zero(1, firingDelays.cols());
    for(int i = 1; i < firingDelays.cols(); i++)
    {
        if(timeIndex + 1 < times.cols() && times(0, timeIndex + 1) <= firingDelays(0, i))
        {
            timeIndex++;

            T dt1 = times(0, timeIndex) - firingDelays(0, i - 1);
            positions(0, i) = positions(0, i - 1) + (currentLinearSpeed * dt1) + (0.5 * currentLinearAcceleration * std::pow(dt1, 2));

            currentLinearSpeed = linearSpeeds(0, timeIndex);
            currentLinearAcceleration = linearAccelerations(0, timeIndex);

            T dt2 = firingDelays(0, i) - times(0, timeIndex);
            positions(0, i) += (currentLinearSpeed * dt2) + (0.5 * currentLinearAcceleration * std::pow(dt2, 2));
            currentLinearSpeed += currentLinearAcceleration * dt2;
        }
        else
        {
            T dt = firingDelays(0, i) - firingDelays(0, i - 1);
            positions(0, i) = positions(0, i - 1) + (currentLinearSpeed * dt) + (0.5 * currentLinearAcceleration * std::pow(dt, 2));
            currentLinearSpeed += currentLinearAcceleration * dt;
        }
    }
    return positions;
}

template<typename T>
typename NoiseSkewDataPointsFilter<T>::Array NoiseSkewDataPointsFilter<T>::computeRotations(const Array& angularSpeeds, const Array& angularAccelerations,
                                                                                            const Array& times, const Array& firingDelays)
{
    int timeIndex = 0;
    T currentAngularSpeed = angularSpeeds(0, 0);
    T currentAngularAcceleration = angularAccelerations(0, 0);
    Array orientations = Array::Zero(1, firingDelays.cols());
    for(int i = 1; i < firingDelays.cols(); i++)
    {
        if(timeIndex + 1 < times.cols() && times(0, timeIndex + 1) <= firingDelays(0, i))
        {
            timeIndex++;

            T dt1 = times(0, timeIndex) - firingDelays(0, i - 1);
            orientations(0, i) = orientations(0, i - 1) + (currentAngularSpeed * dt1) + (0.5 * currentAngularAcceleration * std::pow(dt1, 2));

            currentAngularSpeed = angularSpeeds(0, timeIndex);
            currentAngularAcceleration = angularAccelerations(0, timeIndex);

            T dt2 = firingDelays(0, i) - times(0, timeIndex);
            orientations(0, i) += (currentAngularSpeed * dt2) + (0.5 * currentAngularAcceleration * std::pow(dt2, 2));
            currentAngularSpeed += currentAngularAcceleration * dt2;
        }
        else
        {
            T dt = firingDelays(0, i) - firingDelays(0, i - 1);
            orientations(0, i) = orientations(0, i - 1) + (currentAngularSpeed * dt) + (0.5 * currentAngularAcceleration * std::pow(dt, 2));
            currentAngularSpeed += currentAngularAcceleration * dt;
        }
    }
    return orientations;
}

template<typename T>
NoiseSkewDataPointsFilter<T>::NoiseSkewDataPointsFilter(const Parameters& params):
        PointMatcher<T>::DataPointsFilter("NoiseSkewDataPointsFilter", NoiseSkewDataPointsFilter::availableParameters(), params),
        skewModel(Parametrizable::get<unsigned>("skewModel")),
        linearSpeedNoisesX(castToLinearSpeedNoises(Parametrizable::getParamValueString("linearSpeedsX"), Parametrizable::get<bool>("afterDeskewing"))),
        linearSpeedNoisesY(castToLinearSpeedNoises(Parametrizable::getParamValueString("linearSpeedsY"), Parametrizable::get<bool>("afterDeskewing"))),
        linearSpeedNoisesZ(castToLinearSpeedNoises(Parametrizable::getParamValueString("linearSpeedsZ"), Parametrizable::get<bool>("afterDeskewing"))),
        linearAccelerationNoisesX(castToLinearAccelerationNoises(Parametrizable::getParamValueString("linearAccelerationsX"), Parametrizable::get<bool>("afterDeskewing"))),
        linearAccelerationNoisesY(castToLinearAccelerationNoises(Parametrizable::getParamValueString("linearAccelerationsY"), Parametrizable::get<bool>("afterDeskewing"))),
        linearAccelerationNoisesZ(castToLinearAccelerationNoises(Parametrizable::getParamValueString("linearAccelerationsZ"), Parametrizable::get<bool>("afterDeskewing"))),
        angularSpeedNoisesX(castToAngularSpeedNoises(Parametrizable::getParamValueString("angularSpeedsX"), Parametrizable::get<bool>("afterDeskewing"))),
        angularSpeedNoisesY(castToAngularSpeedNoises(Parametrizable::getParamValueString("angularSpeedsY"), Parametrizable::get<bool>("afterDeskewing"))),
        angularSpeedNoisesZ(castToAngularSpeedNoises(Parametrizable::getParamValueString("angularSpeedsZ"), Parametrizable::get<bool>("afterDeskewing"))),
        angularAccelerationNoisesX(castToAngularAccelerationNoises(Parametrizable::getParamValueString("angularAccelerationsX"), Parametrizable::get<bool>("afterDeskewing"))),
        angularAccelerationNoisesY(castToAngularAccelerationNoises(Parametrizable::getParamValueString("angularAccelerationsY"), Parametrizable::get<bool>("afterDeskewing"))),
        angularAccelerationNoisesZ(castToAngularAccelerationNoises(Parametrizable::getParamValueString("angularAccelerationsZ"), Parametrizable::get<bool>("afterDeskewing"))),
        measureTimes(castToArray(Parametrizable::getParamValueString("measureTimes"))),
        cornerPointUncertainty(Parametrizable::get<T>("cornerPointUncertainty")),
        uncertaintyThreshold(Parametrizable::get<T>("uncertaintyThreshold")),
        uncertaintyQuantile(Parametrizable::get<T>("uncertaintyQuantile"))
{
}

template<typename T>
typename PointMatcher<T>::DataPoints NoiseSkewDataPointsFilter<T>::filter(const DataPoints& input)
{
    DataPoints output(input);
    inPlaceFilter(output);
    return output;
}

template<typename T>
void NoiseSkewDataPointsFilter<T>::inPlaceFilter(DataPoints& cloud)
{
    if(!cloud.descriptorExists("t"))
    {
        throw InvalidField("NoiseSkewDataPointsFilter: Error, cannot find t in times.");
    }

    Array uncertainties = Array::Zero(1, cloud.getNbPoints());
    switch(skewModel)
    {
        case 0:
        {
            const auto& stamps = cloud.getDescriptorViewByName("t");
            Array firingDelays = (stamps.array() - stamps.minCoeff()) / 1e9;
            uncertainties = firingDelays * 0.5 * T(0.5);
            break;
        }
        default:
            throw InvalidParameter("NoiseSkewDataPointsFilter: Error, skewModel id " + std::to_string(skewModel) + " does not exist.");
    }

    std::vector<int> uncertaintyOrdering = computeOrdering<T>(uncertainties);
    int uncertaintyQuantileIndex = std::ceil(uncertaintyQuantile * (uncertaintyOrdering.size() - 1));
    T uncertaintyQuantileValue = uncertainties(0, uncertaintyOrdering[uncertaintyQuantileIndex]);
    T maxUncertainty = std::min(uncertaintyQuantileValue, uncertaintyThreshold);
    for(int i = 0; i < uncertainties.cols(); i++)
    {
        if(uncertainties(0, i) > maxUncertainty)
        {
            uncertainties(0, i) = std::numeric_limits<T>::infinity();
        }
    }

    cloud.addDescriptor("skewUncertainty", uncertainties);
}

template struct NoiseSkewDataPointsFilter<float>;
template struct NoiseSkewDataPointsFilter<double>;