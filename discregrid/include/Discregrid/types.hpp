#pragma once

#include <Eigen/Dense>

namespace Discregrid {

#ifdef DISCREGRID_USE_DOUBLE
using real = double;
#else
using real = float;
#endif

using Vector3r = Eigen::Matrix<real, 3, 1>;
using Matrix3r = Eigen::Matrix<real, 3, 3>;
using AlignedBox3r = Eigen::AlignedBox<real, 3>;

}
