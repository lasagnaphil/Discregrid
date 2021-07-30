#include "data/z_sort_table.hpp"
#include "cubic_lagrange_discrete_grid.hpp"
#include <utility/serialize.hpp>
#include "utility/spinlock.hpp"
#include "utility/timing.hpp"

#include <iostream>
#include <iomanip>
#include <atomic>
#include <numeric>
#include <set>
#include <chrono>
#include <future>

using namespace Eigen;

namespace Discregrid
{

namespace
{

double const abscissae[32][3] = {
	{-1.000000000000, -1.000000000000, -1.000000000000}, // 0
	{-1.000000000000, -1.000000000000, 1.000000000000},  // 1
	{-1.000000000000, 1.000000000000, -1.000000000000},  // 2
	{-1.000000000000, 1.000000000000, 1.000000000000},   // 3
	{1.000000000000, -1.000000000000, -1.000000000000},  // 4
	{1.000000000000, -1.000000000000, 1.000000000000},   // 5
	{1.000000000000, 1.000000000000, -1.000000000000},   // 6
	{1.000000000000, 1.000000000000, 1.000000000000},	// 7
	{-0.333333333333, -1.000000000000, -1.000000000000}, // 8
	{-0.333333333333, -1.000000000000, 1.000000000000},  // 9
	{-0.333333333333, 1.000000000000, -1.000000000000},  //10
	{-0.333333333333, 1.000000000000, 1.000000000000},   //11
	{0.333333333333, -1.000000000000, -1.000000000000},  //12
	{0.333333333333, -1.000000000000, 1.000000000000},   //13
	{0.333333333333, 1.000000000000, -1.000000000000},   //14
	{0.333333333333, 1.000000000000, 1.000000000000},	//15
	{-1.000000000000, -0.333333333333, -1.000000000000}, //16
	{-1.000000000000, -0.333333333333, 1.000000000000},  //17
	{-1.000000000000, 0.333333333333, -1.000000000000},  //18
	{-1.000000000000, 0.333333333333, 1.000000000000},   //19
	{1.000000000000, -0.333333333333, -1.000000000000},  //20
	{1.000000000000, -0.333333333333, 1.000000000000},   //21
	{1.000000000000, 0.333333333333, -1.000000000000},   //22
	{1.000000000000, 0.333333333333, 1.000000000000},	//23
	{-1.000000000000, -1.000000000000, -0.333333333333}, //24
	{-1.000000000000, -1.000000000000, 0.333333333333},  //25
	{-1.000000000000, 1.000000000000, -0.333333333333},  //26
	{-1.000000000000, 1.000000000000, 0.333333333333},   //27
	{1.000000000000, -1.000000000000, -0.333333333333},  //28
	{1.000000000000, -1.000000000000, 0.333333333333},   //29
	{1.000000000000, 1.000000000000, -0.333333333333},   //30
	{1.000000000000, 1.000000000000, 0.333333333333}	 //31
};

double const abscissae_[32][3] = {
	{-1.000000000000, -1.000000000000, -1.000000000000}, // 0 -->  0
	{1.000000000000, -1.000000000000, -1.000000000000},  // 4 -->  1
	{-1.000000000000, 1.000000000000, -1.000000000000},  // 2 -->  2
	{1.000000000000, 1.000000000000, -1.000000000000},   // 6 -->  3
	{-1.000000000000, -1.000000000000, 1.000000000000},  // 1 -->  4
	{1.000000000000, -1.000000000000, 1.000000000000},   // 5 -->  5
	{-1.000000000000, 1.000000000000, 1.000000000000},   // 3 -->  6
	{1.000000000000, 1.000000000000, 1.000000000000},	// 7 -->  7

	{-0.333333333333, -1.000000000000, -1.000000000000}, // 8 -->  8
	{0.333333333333, -1.000000000000, -1.000000000000},  //12 -->  9
	{-0.333333333333, -1.000000000000, 1.000000000000},  // 9 --> 10
	{0.333333333333, -1.000000000000, 1.000000000000},   //13 --> 11
	{-0.333333333333, 1.000000000000, -1.000000000000},  //10 --> 12
	{0.333333333333, 1.000000000000, -1.000000000000},   //14 --> 13
	{-0.333333333333, 1.000000000000, 1.000000000000},   //11 --> 14
	{0.333333333333, 1.000000000000, 1.000000000000},	//15 --> 15

	{-1.000000000000, -0.333333333333, -1.000000000000}, //16 --> 16
	{-1.000000000000, 0.333333333333, -1.000000000000},  //18 --> 17
	{1.000000000000, -0.333333333333, -1.000000000000},  //20 --> 18
	{1.000000000000, 0.333333333333, -1.000000000000},   //22 --> 19
	{-1.000000000000, -0.333333333333, 1.000000000000},  //17 --> 20
	{-1.000000000000, 0.333333333333, 1.000000000000},   //19 --> 21
	{1.000000000000, -0.333333333333, 1.000000000000},   //21 --> 22
	{1.000000000000, 0.333333333333, 1.000000000000},	//23 --> 23

	{-1.000000000000, -1.000000000000, -0.333333333333}, //24 --> 24
	{-1.000000000000, -1.000000000000, 0.333333333333},  //25 --> 25
	{-1.000000000000, 1.000000000000, -0.333333333333},  //26 --> 26
	{-1.000000000000, 1.000000000000, 0.333333333333},   //27 --> 27
	{1.000000000000, -1.000000000000, -0.333333333333},  //28 --> 28
	{1.000000000000, -1.000000000000, 0.333333333333},   //29 --> 29
	{1.000000000000, 1.000000000000, -0.333333333333},   //30 --> 30
	{1.000000000000, 1.000000000000, 0.333333333333}	 //31 --> 31
};

Matrix<double, 32, 1>
shape_function(Vector3d const &xi, Matrix<double, 32, 3> *gradient = nullptr)
{
	auto res = Matrix<double, 32, 1>{};

	auto x = xi[0];
	auto y = xi[1];
	auto z = xi[2];

	auto x2 = x * x;
	auto y2 = y * y;
	auto z2 = z * z;

	auto _1mx = 1.0 - x;
	auto _1my = 1.0 - y;
	auto _1mz = 1.0 - z;

	auto _1px = 1.0 + x;
	auto _1py = 1.0 + y;
	auto _1pz = 1.0 + z;

	auto _1m3x = 1.0 - 3.0 * x;
	auto _1m3y = 1.0 - 3.0 * y;
	auto _1m3z = 1.0 - 3.0 * z;

	auto _1p3x = 1.0 + 3.0 * x;
	auto _1p3y = 1.0 + 3.0 * y;
	auto _1p3z = 1.0 + 3.0 * z;

	auto _1mxt1my = _1mx * _1my;
	auto _1mxt1py = _1mx * _1py;
	auto _1pxt1my = _1px * _1my;
	auto _1pxt1py = _1px * _1py;

	auto _1mxt1mz = _1mx * _1mz;
	auto _1mxt1pz = _1mx * _1pz;
	auto _1pxt1mz = _1px * _1mz;
	auto _1pxt1pz = _1px * _1pz;

	auto _1myt1mz = _1my * _1mz;
	auto _1myt1pz = _1my * _1pz;
	auto _1pyt1mz = _1py * _1mz;
	auto _1pyt1pz = _1py * _1pz;

	auto _1mx2 = 1.0 - x2;
	auto _1my2 = 1.0 - y2;
	auto _1mz2 = 1.0 - z2;

	// Corner nodes.
	auto fac = 1.0 / 64.0 * (9.0 * (x2 + y2 + z2) - 19.0);
	res[0] = fac * _1mxt1my * _1mz;
	res[1] = fac * _1mxt1my * _1pz;
	res[2] = fac * _1mxt1py * _1mz;
	res[3] = fac * _1mxt1py * _1pz;
	res[4] = fac * _1pxt1my * _1mz;
	res[5] = fac * _1pxt1my * _1pz;
	res[6] = fac * _1pxt1py * _1mz;
	res[7] = fac * _1pxt1py * _1pz;

	// Edge nodes.

	fac = 9.0 / 64.0 * _1mx2;
	auto fact1m3x = fac * _1m3x;
	auto fact1p3x = fac * _1p3x;
	res[8] = fact1m3x * _1myt1mz;
	res[9] = fact1m3x * _1myt1pz;
	res[10] = fact1m3x * _1pyt1mz;
	res[11] = fact1m3x * _1pyt1pz;
	res[12] = fact1p3x * _1myt1mz;
	res[13] = fact1p3x * _1myt1pz;
	res[14] = fact1p3x * _1pyt1mz;
	res[15] = fact1p3x * _1pyt1pz;

	fac = 9.0 / 64.0 * _1my2;
	auto fact1m3y = fac * _1m3y;
	auto fact1p3y = fac * _1p3y;
	res[16] = fact1m3y * _1mxt1mz;
	res[17] = fact1m3y * _1mxt1pz;
	res[18] = fact1p3y * _1mxt1mz;
	res[19] = fact1p3y * _1mxt1pz;
	res[20] = fact1m3y * _1pxt1mz;
	res[21] = fact1m3y * _1pxt1pz;
	res[22] = fact1p3y * _1pxt1mz;
	res[23] = fact1p3y * _1pxt1pz;

	fac = 9.0 / 64.0 * _1mz2;
	auto fact1m3z = fac * _1m3z;
	auto fact1p3z = fac * _1p3z;
	res[24] = fact1m3z * _1mxt1my;
	res[25] = fact1p3z * _1mxt1my;
	res[26] = fact1m3z * _1mxt1py;
	res[27] = fact1p3z * _1mxt1py;
	res[28] = fact1m3z * _1pxt1my;
	res[29] = fact1p3z * _1pxt1my;
	res[30] = fact1m3z * _1pxt1py;
	res[31] = fact1p3z * _1pxt1py;

	if (gradient)
	{
		auto &dN = *gradient;

		auto _9t3x2py2pz2m19 = 9.0 * (3.0 * x2 + y2 + z2) - 19.0;
		auto _9tx2p3y2pz2m19 = 9.0 * (x2 + 3.0 * y2 + z2) - 19.0;
		auto _9tx2py2p3z2m19 = 9.0 * (x2 + y2 + 3.0 * z2) - 19.0;
		auto _18x = 18.0 * x;
		auto _18y = 18.0 * y;
		auto _18z = 18.0 * z;

		auto _3m9x2 = 3.0 - 9.0 * x2;
		auto _3m9y2 = 3.0 - 9.0 * y2;
		auto _3m9z2 = 3.0 - 9.0 * z2;

		auto _2x = 2.0 * x;
		auto _2y = 2.0 * y;
		auto _2z = 2.0 * z;

		auto _18xm9t3x2py2pz2m19 = _18x - _9t3x2py2pz2m19;
		auto _18xp9t3x2py2pz2m19 = _18x + _9t3x2py2pz2m19;
		auto _18ym9tx2p3y2pz2m19 = _18y - _9tx2p3y2pz2m19;
		auto _18yp9tx2p3y2pz2m19 = _18y + _9tx2p3y2pz2m19;
		auto _18zm9tx2py2p3z2m19 = _18z - _9tx2py2p3z2m19;
		auto _18zp9tx2py2p3z2m19 = _18z + _9tx2py2p3z2m19;

		dN(0, 0) = _18xm9t3x2py2pz2m19 * _1myt1mz;
		dN(0, 1) = _1mxt1mz * _18ym9tx2p3y2pz2m19;
		dN(0, 2) = _1mxt1my * _18zm9tx2py2p3z2m19;
		dN(1, 0) = _18xm9t3x2py2pz2m19 * _1myt1pz;
		dN(1, 1) = _1mxt1pz * _18ym9tx2p3y2pz2m19;
		dN(1, 2) = _1mxt1my * _18zp9tx2py2p3z2m19;
		dN(2, 0) = _18xm9t3x2py2pz2m19 * _1pyt1mz;
		dN(2, 1) = _1mxt1mz * _18yp9tx2p3y2pz2m19;
		dN(2, 2) = _1mxt1py * _18zm9tx2py2p3z2m19;
		dN(3, 0) = _18xm9t3x2py2pz2m19 * _1pyt1pz;
		dN(3, 1) = _1mxt1pz * _18yp9tx2p3y2pz2m19;
		dN(3, 2) = _1mxt1py * _18zp9tx2py2p3z2m19;
		dN(4, 0) = _18xp9t3x2py2pz2m19 * _1myt1mz;
		dN(4, 1) = _1pxt1mz * _18ym9tx2p3y2pz2m19;
		dN(4, 2) = _1pxt1my * _18zm9tx2py2p3z2m19;
		dN(5, 0) = _18xp9t3x2py2pz2m19 * _1myt1pz;
		dN(5, 1) = _1pxt1pz * _18ym9tx2p3y2pz2m19;
		dN(5, 2) = _1pxt1my * _18zp9tx2py2p3z2m19;
		dN(6, 0) = _18xp9t3x2py2pz2m19 * _1pyt1mz;
		dN(6, 1) = _1pxt1mz * _18yp9tx2p3y2pz2m19;
		dN(6, 2) = _1pxt1py * _18zm9tx2py2p3z2m19;
		dN(7, 0) = _18xp9t3x2py2pz2m19 * _1pyt1pz;
		dN(7, 1) = _1pxt1pz * _18yp9tx2p3y2pz2m19;
		dN(7, 2) = _1pxt1py * _18zp9tx2py2p3z2m19;

		dN.topRows(8) /= 64.0;

		auto _m3m9x2m2x = -_3m9x2 - _2x;
		auto _p3m9x2m2x = _3m9x2 - _2x;
		auto _1mx2t1m3x = _1mx2 * _1m3x;
		auto _1mx2t1p3x = _1mx2 * _1p3x;
		dN(8, 0) = _m3m9x2m2x * _1myt1mz,
			  dN(8, 1) = -_1mx2t1m3x * _1mz,
			  dN(8, 2) = -_1mx2t1m3x * _1my;
		dN(9, 0) = _m3m9x2m2x * _1myt1pz,
			  dN(9, 1) = -_1mx2t1m3x * _1pz,
			  dN(9, 2) = _1mx2t1m3x * _1my;
		dN(10, 0) = _m3m9x2m2x * _1pyt1mz,
			   dN(10, 1) = _1mx2t1m3x * _1mz,
			   dN(10, 2) = -_1mx2t1m3x * _1py;
		dN(11, 0) = _m3m9x2m2x * _1pyt1pz,
			   dN(11, 1) = _1mx2t1m3x * _1pz,
			   dN(11, 2) = _1mx2t1m3x * _1py;
		dN(12, 0) = _p3m9x2m2x * _1myt1mz,
			   dN(12, 1) = -_1mx2t1p3x * _1mz,
			   dN(12, 2) = -_1mx2t1p3x * _1my;
		dN(13, 0) = _p3m9x2m2x * _1myt1pz,
			   dN(13, 1) = -_1mx2t1p3x * _1pz,
			   dN(13, 2) = _1mx2t1p3x * _1my;
		dN(14, 0) = _p3m9x2m2x * _1pyt1mz,
			   dN(14, 1) = _1mx2t1p3x * _1mz,
			   dN(14, 2) = -_1mx2t1p3x * _1py;
		dN(15, 0) = _p3m9x2m2x * _1pyt1pz,
			   dN(15, 1) = _1mx2t1p3x * _1pz,
			   dN(15, 2) = _1mx2t1p3x * _1py;

		auto _m3m9y2m2y = -_3m9y2 - _2y;
		auto _p3m9y2m2y = _3m9y2 - _2y;
		auto _1my2t1m3y = _1my2 * _1m3y;
		auto _1my2t1p3y = _1my2 * _1p3y;
		dN(16, 0) = -_1my2t1m3y * _1mz,
			   dN(16, 1) = _m3m9y2m2y * _1mxt1mz,
			   dN(16, 2) = -_1my2t1m3y * _1mx;
		dN(17, 0) = -_1my2t1m3y * _1pz,
			   dN(17, 1) = _m3m9y2m2y * _1mxt1pz,
			   dN(17, 2) = _1my2t1m3y * _1mx;
		dN(18, 0) = -_1my2t1p3y * _1mz,
			   dN(18, 1) = _p3m9y2m2y * _1mxt1mz,
			   dN(18, 2) = -_1my2t1p3y * _1mx;
		dN(19, 0) = -_1my2t1p3y * _1pz,
			   dN(19, 1) = _p3m9y2m2y * _1mxt1pz,
			   dN(19, 2) = _1my2t1p3y * _1mx;
		dN(20, 0) = _1my2t1m3y * _1mz,
			   dN(20, 1) = _m3m9y2m2y * _1pxt1mz,
			   dN(20, 2) = -_1my2t1m3y * _1px;
		dN(21, 0) = _1my2t1m3y * _1pz,
			   dN(21, 1) = _m3m9y2m2y * _1pxt1pz,
			   dN(21, 2) = _1my2t1m3y * _1px;
		dN(22, 0) = _1my2t1p3y * _1mz,
			   dN(22, 1) = _p3m9y2m2y * _1pxt1mz,
			   dN(22, 2) = -_1my2t1p3y * _1px;
		dN(23, 0) = _1my2t1p3y * _1pz,
			   dN(23, 1) = _p3m9y2m2y * _1pxt1pz,
			   dN(23, 2) = _1my2t1p3y * _1px;

		auto _m3m9z2m2z = -_3m9z2 - _2z;
		auto _p3m9z2m2z = _3m9z2 - _2z;
		auto _1mz2t1m3z = _1mz2 * _1m3z;
		auto _1mz2t1p3z = _1mz2 * _1p3z;
		dN(24, 0) = -_1mz2t1m3z * _1my,
			   dN(24, 1) = -_1mz2t1m3z * _1mx,
			   dN(24, 2) = _m3m9z2m2z * _1mxt1my;
		dN(25, 0) = -_1mz2t1p3z * _1my,
			   dN(25, 1) = -_1mz2t1p3z * _1mx,
			   dN(25, 2) = _p3m9z2m2z * _1mxt1my;
		dN(26, 0) = -_1mz2t1m3z * _1py,
			   dN(26, 1) = _1mz2t1m3z * _1mx,
			   dN(26, 2) = _m3m9z2m2z * _1mxt1py;
		dN(27, 0) = -_1mz2t1p3z * _1py,
			   dN(27, 1) = _1mz2t1p3z * _1mx,
			   dN(27, 2) = _p3m9z2m2z * _1mxt1py;
		dN(28, 0) = _1mz2t1m3z * _1my,
			   dN(28, 1) = -_1mz2t1m3z * _1px,
			   dN(28, 2) = _m3m9z2m2z * _1pxt1my;
		dN(29, 0) = _1mz2t1p3z * _1my,
			   dN(29, 1) = -_1mz2t1p3z * _1px,
			   dN(29, 2) = _p3m9z2m2z * _1pxt1my;
		dN(30, 0) = _1mz2t1m3z * _1py,
			   dN(30, 1) = _1mz2t1m3z * _1px,
			   dN(30, 2) = _m3m9z2m2z * _1pxt1py;
		dN(31, 0) = _1mz2t1p3z * _1py,
			   dN(31, 1) = _1mz2t1p3z * _1px,
			   dN(31, 2) = _p3m9z2m2z * _1pxt1py;

		dN.bottomRows(32u - 8u) *= 9.0 / 64.0;
	}

	return res;
}

Matrix<double, 32, 1>
shape_function_(Vector3d const &xi, Matrix<double, 32, 3> *gradient = nullptr)
{
	auto res = Matrix<double, 32, 1>{};

	auto x = xi[0];
	auto y = xi[1];
	auto z = xi[2];

	auto x2 = x * x;
	auto y2 = y * y;
	auto z2 = z * z;

	auto _1mx = 1.0 - x;
	auto _1my = 1.0 - y;
	auto _1mz = 1.0 - z;

	auto _1px = 1.0 + x;
	auto _1py = 1.0 + y;
	auto _1pz = 1.0 + z;

	auto _1m3x = 1.0 - 3.0 * x;
	auto _1m3y = 1.0 - 3.0 * y;
	auto _1m3z = 1.0 - 3.0 * z;

	auto _1p3x = 1.0 + 3.0 * x;
	auto _1p3y = 1.0 + 3.0 * y;
	auto _1p3z = 1.0 + 3.0 * z;

	auto _1mxt1my = _1mx * _1my;
	auto _1mxt1py = _1mx * _1py;
	auto _1pxt1my = _1px * _1my;
	auto _1pxt1py = _1px * _1py;

	auto _1mxt1mz = _1mx * _1mz;
	auto _1mxt1pz = _1mx * _1pz;
	auto _1pxt1mz = _1px * _1mz;
	auto _1pxt1pz = _1px * _1pz;

	auto _1myt1mz = _1my * _1mz;
	auto _1myt1pz = _1my * _1pz;
	auto _1pyt1mz = _1py * _1mz;
	auto _1pyt1pz = _1py * _1pz;

	auto _1mx2 = 1.0 - x2;
	auto _1my2 = 1.0 - y2;
	auto _1mz2 = 1.0 - z2;

	// Corner nodes.
	auto fac = 1.0 / 64.0 * (9.0 * (x2 + y2 + z2) - 19.0);
	res[0] = fac * _1mxt1my * _1mz;
	res[1] = fac * _1pxt1my * _1mz;
	res[2] = fac * _1mxt1py * _1mz;
	res[3] = fac * _1pxt1py * _1mz;
	res[4] = fac * _1mxt1my * _1pz;
	res[5] = fac * _1pxt1my * _1pz;
	res[6] = fac * _1mxt1py * _1pz;
	res[7] = fac * _1pxt1py * _1pz;

	// Edge nodes.

	fac = 9.0 / 64.0 * _1mx2;
	auto fact1m3x = fac * _1m3x;
	auto fact1p3x = fac * _1p3x;
	res[8] = fact1m3x * _1myt1mz;
	res[9] = fact1p3x * _1myt1mz;
	res[10] = fact1m3x * _1myt1pz;
	res[11] = fact1p3x * _1myt1pz;
	res[12] = fact1m3x * _1pyt1mz;
	res[13] = fact1p3x * _1pyt1mz;
	res[14] = fact1m3x * _1pyt1pz;
	res[15] = fact1p3x * _1pyt1pz;

	fac = 9.0 / 64.0 * _1my2;
	auto fact1m3y = fac * _1m3y;
	auto fact1p3y = fac * _1p3y;
	res[16] = fact1m3y * _1mxt1mz;
	res[17] = fact1p3y * _1mxt1mz;
	res[18] = fact1m3y * _1pxt1mz;
	res[19] = fact1p3y * _1pxt1mz;
	res[20] = fact1m3y * _1mxt1pz;
	res[21] = fact1p3y * _1mxt1pz;
	res[22] = fact1m3y * _1pxt1pz;
	res[23] = fact1p3y * _1pxt1pz;

	fac = 9.0 / 64.0 * _1mz2;
	auto fact1m3z = fac * _1m3z;
	auto fact1p3z = fac * _1p3z;
	res[24] = fact1m3z * _1mxt1my;
	res[25] = fact1p3z * _1mxt1my;
	res[26] = fact1m3z * _1mxt1py;
	res[27] = fact1p3z * _1mxt1py;
	res[28] = fact1m3z * _1pxt1my;
	res[29] = fact1p3z * _1pxt1my;
	res[30] = fact1m3z * _1pxt1py;
	res[31] = fact1p3z * _1pxt1py;

	if (gradient)
	{
		auto &dN = *gradient;

		auto _9t3x2py2pz2m19 = 9.0 * (3.0 * x2 + y2 + z2) - 19.0;
		auto _9tx2p3y2pz2m19 = 9.0 * (x2 + 3.0 * y2 + z2) - 19.0;
		auto _9tx2py2p3z2m19 = 9.0 * (x2 + y2 + 3.0 * z2) - 19.0;
		auto _18x = 18.0 * x;
		auto _18y = 18.0 * y;
		auto _18z = 18.0 * z;

		auto _3m9x2 = 3.0 - 9.0 * x2;
		auto _3m9y2 = 3.0 - 9.0 * y2;
		auto _3m9z2 = 3.0 - 9.0 * z2;

		auto _2x = 2.0 * x;
		auto _2y = 2.0 * y;
		auto _2z = 2.0 * z;

		auto _18xm9t3x2py2pz2m19 = _18x - _9t3x2py2pz2m19;
		auto _18xp9t3x2py2pz2m19 = _18x + _9t3x2py2pz2m19;
		auto _18ym9tx2p3y2pz2m19 = _18y - _9tx2p3y2pz2m19;
		auto _18yp9tx2p3y2pz2m19 = _18y + _9tx2p3y2pz2m19;
		auto _18zm9tx2py2p3z2m19 = _18z - _9tx2py2p3z2m19;
		auto _18zp9tx2py2p3z2m19 = _18z + _9tx2py2p3z2m19;

		dN(0, 0) = _18xm9t3x2py2pz2m19 * _1myt1mz;
		dN(0, 1) = _1mxt1mz * _18ym9tx2p3y2pz2m19;
		dN(0, 2) = _1mxt1my * _18zm9tx2py2p3z2m19;
		dN(1, 0) = _18xp9t3x2py2pz2m19 * _1myt1mz;
		dN(1, 1) = _1pxt1mz * _18ym9tx2p3y2pz2m19;
		dN(1, 2) = _1pxt1my * _18zm9tx2py2p3z2m19;
		dN(2, 0) = _18xm9t3x2py2pz2m19 * _1pyt1mz;
		dN(2, 1) = _1mxt1mz * _18yp9tx2p3y2pz2m19;
		dN(2, 2) = _1mxt1py * _18zm9tx2py2p3z2m19;
		dN(3, 0) = _18xp9t3x2py2pz2m19 * _1pyt1mz;
		dN(3, 1) = _1pxt1mz * _18yp9tx2p3y2pz2m19;
		dN(3, 2) = _1pxt1py * _18zm9tx2py2p3z2m19;
		dN(4, 0) = _18xm9t3x2py2pz2m19 * _1myt1pz;
		dN(4, 1) = _1mxt1pz * _18ym9tx2p3y2pz2m19;
		dN(4, 2) = _1mxt1my * _18zp9tx2py2p3z2m19;
		dN(5, 0) = _18xp9t3x2py2pz2m19 * _1myt1pz;
		dN(5, 1) = _1pxt1pz * _18ym9tx2p3y2pz2m19;
		dN(5, 2) = _1pxt1my * _18zp9tx2py2p3z2m19;
		dN(6, 0) = _18xm9t3x2py2pz2m19 * _1pyt1pz;
		dN(6, 1) = _1mxt1pz * _18yp9tx2p3y2pz2m19;
		dN(6, 2) = _1mxt1py * _18zp9tx2py2p3z2m19;
		dN(7, 0) = _18xp9t3x2py2pz2m19 * _1pyt1pz;
		dN(7, 1) = _1pxt1pz * _18yp9tx2p3y2pz2m19;
		dN(7, 2) = _1pxt1py * _18zp9tx2py2p3z2m19;

		dN.topRows(8) /= 64.0;

		auto _m3m9x2m2x = -_3m9x2 - _2x;
		auto _p3m9x2m2x = _3m9x2 - _2x;
		auto _1mx2t1m3x = _1mx2 * _1m3x;
		auto _1mx2t1p3x = _1mx2 * _1p3x;
		dN(8, 0) = _m3m9x2m2x * _1myt1mz,
			  dN(8, 1) = -_1mx2t1m3x * _1mz,
			  dN(8, 2) = -_1mx2t1m3x * _1my;
		dN(9, 0) = _p3m9x2m2x * _1myt1mz,
			  dN(9, 1) = -_1mx2t1p3x * _1mz,
			  dN(9, 2) = -_1mx2t1p3x * _1my;
		dN(10, 0) = _m3m9x2m2x * _1myt1pz,
			   dN(10, 1) = -_1mx2t1m3x * _1pz,
			   dN(10, 2) = _1mx2t1m3x * _1my;
		dN(11, 0) = _p3m9x2m2x * _1myt1pz,
			   dN(11, 1) = -_1mx2t1p3x * _1pz,
			   dN(11, 2) = _1mx2t1p3x * _1my;
		dN(12, 0) = _m3m9x2m2x * _1pyt1mz,
			   dN(12, 1) = _1mx2t1m3x * _1mz,
			   dN(12, 2) = -_1mx2t1m3x * _1py;
		dN(13, 0) = _p3m9x2m2x * _1pyt1mz,
			   dN(13, 1) = _1mx2t1p3x * _1mz,
			   dN(13, 2) = -_1mx2t1p3x * _1py;
		dN(14, 0) = _m3m9x2m2x * _1pyt1pz,
			   dN(14, 1) = _1mx2t1m3x * _1pz,
			   dN(14, 2) = _1mx2t1m3x * _1py;
		dN(15, 0) = _p3m9x2m2x * _1pyt1pz,
			   dN(15, 1) = _1mx2t1p3x * _1pz,
			   dN(15, 2) = _1mx2t1p3x * _1py;

		auto _m3m9y2m2y = -_3m9y2 - _2y;
		auto _p3m9y2m2y = _3m9y2 - _2y;
		auto _1my2t1m3y = _1my2 * _1m3y;
		auto _1my2t1p3y = _1my2 * _1p3y;
		dN(16, 0) = -_1my2t1m3y * _1mz,
			   dN(16, 1) = _m3m9y2m2y * _1mxt1mz,
			   dN(16, 2) = -_1my2t1m3y * _1mx;
		dN(17, 0) = -_1my2t1p3y * _1mz,
			   dN(17, 1) = _p3m9y2m2y * _1mxt1mz,
			   dN(17, 2) = -_1my2t1p3y * _1mx;
		dN(18, 0) = _1my2t1m3y * _1mz,
			   dN(18, 1) = _m3m9y2m2y * _1pxt1mz,
			   dN(18, 2) = -_1my2t1m3y * _1px;
		dN(19, 0) = _1my2t1p3y * _1mz,
			   dN(19, 1) = _p3m9y2m2y * _1pxt1mz,
			   dN(19, 2) = -_1my2t1p3y * _1px;
		dN(20, 0) = -_1my2t1m3y * _1pz,
			   dN(20, 1) = _m3m9y2m2y * _1mxt1pz,
			   dN(20, 2) = _1my2t1m3y * _1mx;
		dN(21, 0) = -_1my2t1p3y * _1pz,
			   dN(21, 1) = _p3m9y2m2y * _1mxt1pz,
			   dN(21, 2) = _1my2t1p3y * _1mx;
		dN(22, 0) = _1my2t1m3y * _1pz,
			   dN(22, 1) = _m3m9y2m2y * _1pxt1pz,
			   dN(22, 2) = _1my2t1m3y * _1px;
		dN(23, 0) = _1my2t1p3y * _1pz,
			   dN(23, 1) = _p3m9y2m2y * _1pxt1pz,
			   dN(23, 2) = _1my2t1p3y * _1px;

		auto _m3m9z2m2z = -_3m9z2 - _2z;
		auto _p3m9z2m2z = _3m9z2 - _2z;
		auto _1mz2t1m3z = _1mz2 * _1m3z;
		auto _1mz2t1p3z = _1mz2 * _1p3z;
		dN(24, 0) = -_1mz2t1m3z * _1my,
			   dN(24, 1) = -_1mz2t1m3z * _1mx,
			   dN(24, 2) = _m3m9z2m2z * _1mxt1my;
		dN(25, 0) = -_1mz2t1p3z * _1my,
			   dN(25, 1) = -_1mz2t1p3z * _1mx,
			   dN(25, 2) = _p3m9z2m2z * _1mxt1my;
		dN(26, 0) = -_1mz2t1m3z * _1py,
			   dN(26, 1) = _1mz2t1m3z * _1mx,
			   dN(26, 2) = _m3m9z2m2z * _1mxt1py;
		dN(27, 0) = -_1mz2t1p3z * _1py,
			   dN(27, 1) = _1mz2t1p3z * _1mx,
			   dN(27, 2) = _p3m9z2m2z * _1mxt1py;
		dN(28, 0) = _1mz2t1m3z * _1my,
			   dN(28, 1) = -_1mz2t1m3z * _1px,
			   dN(28, 2) = _m3m9z2m2z * _1pxt1my;
		dN(29, 0) = _1mz2t1p3z * _1my,
			   dN(29, 1) = -_1mz2t1p3z * _1px,
			   dN(29, 2) = _p3m9z2m2z * _1pxt1my;
		dN(30, 0) = _1mz2t1m3z * _1py,
			   dN(30, 1) = _1mz2t1m3z * _1px,
			   dN(30, 2) = _m3m9z2m2z * _1pxt1py;
		dN(31, 0) = _1mz2t1p3z * _1py,
			   dN(31, 1) = _1mz2t1p3z * _1px,
			   dN(31, 2) = _p3m9z2m2z * _1pxt1py;

		dN.bottomRows(32u - 8u) *= 9.0 / 64.0;
	}

	return res;
}

// Determines Morten value according to z-curve.
inline uint64_t
zValue(Vector3d const &x, double invCellSize)
{
	std::array<int, 3> key;
	for (unsigned int i(0); i < 3; ++i)
	{
		if (x[i] >= 0.0)
			key[i] = static_cast<int>(invCellSize * x[i]);
		else
			key[i] = static_cast<int>(invCellSize * x[i]) - 1;
	}

	std::array<unsigned int, 3> p = {
		static_cast<unsigned int>(static_cast<int64_t>(key[0]) - (std::numeric_limits<int>::lowest() + 1)),
		static_cast<unsigned int>(static_cast<int64_t>(key[1]) - (std::numeric_limits<int>::lowest() + 1)),
		static_cast<unsigned int>(static_cast<int64_t>(key[2]) - (std::numeric_limits<int>::lowest() + 1))};

	return morton_lut(p);
}
} // namespace

Vector3d
CubicLagrangeDiscreteGrid::indexToNodePosition(unsigned int l) const
{
	auto x = Vector3d{};

	auto n = Matrix<unsigned int, 3, 1>::Map(m_resolution.data());

	auto nv = (n[0] + 1) * (n[1] + 1) * (n[2] + 1);
	auto ne_x = (n[0] + 0) * (n[1] + 1) * (n[2] + 1);
	auto ne_y = (n[0] + 1) * (n[1] + 0) * (n[2] + 1);
	auto ne_z = (n[0] + 1) * (n[1] + 1) * (n[2] + 0);
	auto ne = ne_x + ne_y + ne_z;

	auto ijk = Matrix<unsigned int, 3, 1>{};
	if (l < nv)
	{
		ijk(2) = l / ((n[1] + 1) * (n[0] + 1));
		auto temp = l % ((n[1] + 1) * (n[0] + 1));
		ijk(1) = temp / (n[0] + 1);
		ijk(0) = temp % (n[0] + 1);

		x = m_domain.min() + m_cell_size.cwiseProduct(ijk.cast<double>());
	}
	else if (l < nv + 2 * ne_x)
	{
		l -= nv;
		auto e_ind = l / 2;
		ijk(2) = e_ind / ((n[1] + 1) * n[0]);
		auto temp = e_ind % ((n[1] + 1) * n[0]);
		ijk(1) = temp / n[0];
		ijk(0) = temp % n[0];

		x = m_domain.min() + m_cell_size.cwiseProduct(ijk.cast<double>());
		x(0) += (1.0 + static_cast<double>(l % 2)) / 3.0 * m_cell_size[0];
	}
	else if (l < nv + 2 * (ne_x + ne_y))
	{
		l -= (nv + 2 * ne_x);
		auto e_ind = l / 2;
		ijk(0) = e_ind / ((n[2] + 1) * n[1]);
		auto temp = e_ind % ((n[2] + 1) * n[1]);
		ijk(2) = temp / n[1];
		ijk(1) = temp % n[1];

		x = m_domain.min() + m_cell_size.cwiseProduct(ijk.cast<double>());
		x(1) += (1.0 + static_cast<double>(l % 2)) / 3.0 * m_cell_size[1];
	}
	else
	{
		l -= (nv + 2 * (ne_x + ne_y));
		auto e_ind = l / 2;
		ijk(1) = e_ind / ((n[0] + 1) * n[2]);
		auto temp = e_ind % ((n[0] + 1) * n[2]);
		ijk(0) = temp / n[2];
		ijk(2) = temp % n[2];

		x = m_domain.min() + m_cell_size.cwiseProduct(ijk.cast<double>());
		x(2) += (1.0 + static_cast<double>(l % 2)) / 3.0 * m_cell_size[2];
	}

	return x;
}

CubicLagrangeDiscreteGrid::CubicLagrangeDiscreteGrid(std::string const &filename)
{
	load(filename);
}

CubicLagrangeDiscreteGrid::CubicLagrangeDiscreteGrid(AlignedBox3d const &domain,
													 std::array<unsigned int, 3> const &resolution)
	: DiscreteGrid(domain, resolution)
{
}

CubicLagrangeDiscreteGrid::CubicLagrangeDiscreteGrid(const AlignedBox3d& minimum_domain,
                                                     const Vector3d& cell_size)
    : DiscreteGrid(minimum_domain, cell_size)
{
}

void CubicLagrangeDiscreteGrid::save(std::string const &filename) const
{
	auto out = std::ofstream(filename, std::ios::binary);
	serialize::write(*out.rdbuf(), m_domain);
	serialize::write(*out.rdbuf(), m_resolution);
	serialize::write(*out.rdbuf(), m_cell_size);
	serialize::write(*out.rdbuf(), m_inv_cell_size);
	serialize::write(*out.rdbuf(), m_n_cells);
	serialize::write(*out.rdbuf(), m_n_fields);

	serialize::write(*out.rdbuf(), m_nodes.size());
	for (auto const &nodes : m_nodes)
	{
		serialize::write(*out.rdbuf(), nodes.size());
		for (auto const &node : nodes)
		{
			serialize::write(*out.rdbuf(), node);
		}
	}

	serialize::write(*out.rdbuf(), m_cells.size());
	for (auto const &cells : m_cells)
	{
		serialize::write(*out.rdbuf(), cells.size());
		for (auto const &cell : cells)
		{
			serialize::write(*out.rdbuf(), cell);
		}
	}

	serialize::write(*out.rdbuf(), m_cell_map.size());
	for (auto const &maps : m_cell_map)
	{
		serialize::write(*out.rdbuf(), maps.size());
		for (auto const &map : maps)
		{
			serialize::write(*out.rdbuf(), map);
		}
	}

	out.close();
}

void CubicLagrangeDiscreteGrid::load(std::string const &filename)
{
	auto in = std::ifstream(filename, std::ios::binary);

	if (!in.good())
	{
		std::cerr << "ERROR: Discrete grid can not be loaded. Input file does not exist!" << std::endl;
		return;
	}

	serialize::read(*in.rdbuf(), m_domain);
	serialize::read(*in.rdbuf(), m_resolution);
	serialize::read(*in.rdbuf(), m_cell_size);
	serialize::read(*in.rdbuf(), m_inv_cell_size);
	serialize::read(*in.rdbuf(), m_n_cells);
	serialize::read(*in.rdbuf(), m_n_fields);

	auto n_nodes = std::size_t{};
	serialize::read(*in.rdbuf(), n_nodes);
	m_nodes.resize(n_nodes);
	for (auto &nodes : m_nodes)
	{
		serialize::read(*in.rdbuf(), n_nodes);
		nodes.resize(n_nodes);
		for (auto &node : nodes)
		{
			serialize::read(*in.rdbuf(), node);
		}
	}

	auto n_cells = std::size_t{};
	serialize::read(*in.rdbuf(), n_cells);
	m_cells.resize(n_cells);
	for (auto &cells : m_cells)
	{
		serialize::read(*in.rdbuf(), n_cells);
		cells.resize(n_cells);
		for (auto &cell : cells)
		{
			serialize::read(*in.rdbuf(), cell);
		}
	}

	auto n_cell_maps = std::size_t{};
	serialize::read(*in.rdbuf(), n_cell_maps);
	m_cell_map.resize(n_cell_maps);
	for (auto &cell_maps : m_cell_map)
	{
		serialize::read(*in.rdbuf(), n_cell_maps);
		cell_maps.resize(n_cell_maps);
		for (auto &cell_map : cell_maps)
		{
			serialize::read(*in.rdbuf(), cell_map);
		}
	}

	in.close();
}

unsigned int
CubicLagrangeDiscreteGrid::addFunction(ContinuousFunction const &func, bool verbose,
									   SamplePredicate const &pred)
{
	using namespace std::chrono;

	auto t0_construction = high_resolution_clock::now();

	auto n = Matrix<unsigned int, 3, 1>::Map(m_resolution.data());

	auto nv = (n[0] + 1) * (n[1] + 1) * (n[2] + 1);
	auto ne_x = (n[0] + 0) * (n[1] + 1) * (n[2] + 1);
	auto ne_y = (n[0] + 1) * (n[1] + 0) * (n[2] + 1);
	auto ne_z = (n[0] + 1) * (n[1] + 1) * (n[2] + 0);
	auto ne = ne_x + ne_y + ne_z;

	auto n_nodes = nv + 2 * ne;

	m_nodes.push_back({});
	auto &coeffs = m_nodes.back();
	coeffs.resize(n_nodes);

	std::atomic_uint counter(0u);
	SpinLock mutex;
	auto t0 = high_resolution_clock::now();

#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static) nowait
		for (int l = 0; l < static_cast<int>(n_nodes); ++l)
		{
			auto x = indexToNodePosition(l);
			auto &c = coeffs[l];

			if (!pred || pred(x))
				c = func(x);
			else
				c = std::numeric_limits<double>::max();

			if (verbose && (++counter == n_nodes || duration_cast<milliseconds>(high_resolution_clock::now() - t0).count() > 1000u))
			{
				std::async(std::launch::async, [&]() {
					mutex.lock();
					t0 = high_resolution_clock::now();
					std::cout << "\r"
							  << "Construction " << std::setw(20)
							  << 100.0 * static_cast<double>(counter) / static_cast<double>(n_nodes) << "%";
					mutex.unlock();
				});
			}
		}
	}

	m_cells.push_back({});
	auto &cells = m_cells.back();
	cells.resize(m_n_cells);
	for (auto l = 0u; l < m_n_cells; ++l)
	{
		auto k = l / (n[1] * n[0]);
		auto temp = l % (n[1] * n[0]);
		auto j = temp / n[0];
		auto i = temp % n[0];

		auto nx = n[0];
		auto ny = n[1];
		auto nz = n[2];

		auto &cell = cells[l];
		cell[0] = (nx + 1) * (ny + 1) * k + (nx + 1) * j + i;
		cell[1] = (nx + 1) * (ny + 1) * k + (nx + 1) * j + i + 1;
		cell[2] = (nx + 1) * (ny + 1) * k + (nx + 1) * (j + 1) + i;
		cell[3] = (nx + 1) * (ny + 1) * k + (nx + 1) * (j + 1) + i + 1;
		cell[4] = (nx + 1) * (ny + 1) * (k + 1) + (nx + 1) * j + i;
		cell[5] = (nx + 1) * (ny + 1) * (k + 1) + (nx + 1) * j + i + 1;
		cell[6] = (nx + 1) * (ny + 1) * (k + 1) + (nx + 1) * (j + 1) + i;
		cell[7] = (nx + 1) * (ny + 1) * (k + 1) + (nx + 1) * (j + 1) + i + 1;

		auto offset = nv;
		cell[8] = offset + 2 * (nx * (ny + 1) * k + nx * j + i);
		cell[9] = cell[8] + 1;
		cell[10] = offset + 2 * (nx * (ny + 1) * (k + 1) + nx * j + i);
		cell[11] = cell[10] + 1;
		cell[12] = offset + 2 * (nx * (ny + 1) * k + nx * (j + 1) + i);
		cell[13] = cell[12] + 1;
		cell[14] = offset + 2 * (nx * (ny + 1) * (k + 1) + nx * (j + 1) + i);
		cell[15] = cell[14] + 1;

		offset += 2 * ne_x;
		cell[16] = offset + 2 * (ny * (nz + 1) * i + ny * k + j);
		cell[17] = cell[16] + 1;
		cell[18] = offset + 2 * (ny * (nz + 1) * (i + 1) + ny * k + j);
		cell[19] = cell[18] + 1;
		cell[20] = offset + 2 * (ny * (nz + 1) * i + ny * (k + 1) + j);
		cell[21] = cell[20] + 1;
		cell[22] = offset + 2 * (ny * (nz + 1) * (i + 1) + ny * (k + 1) + j);
		cell[23] = cell[22] + 1;

		offset += 2 * ne_y;
		cell[24] = offset + 2 * (nz * (nx + 1) * j + nz * i + k);
		cell[25] = cell[24] + 1;
		cell[26] = offset + 2 * (nz * (nx + 1) * (j + 1) + nz * i + k);
		cell[27] = cell[26] + 1;
		cell[28] = offset + 2 * (nz * (nx + 1) * j + nz * (i + 1) + k);
		cell[29] = cell[28] + 1;
		cell[30] = offset + 2 * (nz * (nx + 1) * (j + 1) + nz * (i + 1) + k);
		cell[31] = cell[30] + 1;
	}

	m_cell_map.push_back({});
	auto &cell_map = m_cell_map.back();
	cell_map.resize(m_n_cells);
	std::iota(cell_map.begin(), cell_map.end(), 0u);

	if (verbose)
	{
		std::cout << "\rConstruction took " << std::setw(15) << static_cast<double>(duration_cast<milliseconds>(high_resolution_clock::now() - t0_construction).count()) / 1000.0 << "s" << std::endl;
	}

	return static_cast<unsigned int>(m_n_fields++);
}

bool
CubicLagrangeDiscreteGrid::determineShapeFunctions(unsigned int field_id, Eigen::Vector3d const &x,
	std::array<unsigned int, 32> &cell, Eigen::Vector3d &c0, Eigen::Matrix<double, 32, 1> &N,
	Eigen::Matrix<double, 32, 3> *dN) const
{
	if (!m_domain.contains(x))
		return false;

	auto mi = (x - m_domain.min()).cwiseProduct(m_inv_cell_size).cast<unsigned int>().eval();
	if (mi[0] >= m_resolution[0])
		mi[0] = m_resolution[0] - 1;
	if (mi[1] >= m_resolution[1])
		mi[1] = m_resolution[1] - 1;
	if (mi[2] >= m_resolution[2])
		mi[2] = m_resolution[2] - 1;
	auto i = multiToSingleIndex({ { mi(0), mi(1), mi(2) } });
	auto i_ = m_cell_map[field_id][i];
	if (i_ == std::numeric_limits<unsigned int>::max())
		return false;

	auto sd = subdomain(i);
	i = i_;
	auto d = sd.diagonal().eval();

	auto denom = (sd.max() - sd.min()).eval();
	c0 = Vector3d::Constant(2.0).cwiseQuotient(denom).eval();
	auto c1 = (sd.max() + sd.min()).cwiseQuotient(denom).eval();
	auto xi = (c0.cwiseProduct(x) - c1).eval();

	cell = m_cells[field_id][i];
	N = shape_function_(xi, dN);
	return true;
}

double 
CubicLagrangeDiscreteGrid::interpolate(unsigned int field_id, Eigen::Vector3d const& xi, const std::array<unsigned int, 32> &cell, const Eigen::Vector3d &c0, const Eigen::Matrix<double, 32, 1> &N,
	Eigen::Vector3d* gradient, Eigen::Matrix<double, 32, 3> *dN) const
{
	if (!gradient)
	{
		auto phi = 0.0;
		for (auto j = 0u; j < 32u; ++j)
		{
			auto v = cell[j];
			auto c = m_nodes[field_id][v];
			if (c == std::numeric_limits<double>::max())
			{
				return std::numeric_limits<double>::max();
			}
			phi += c * N[j];
		}

		return phi;
	}

	auto phi = 0.0;
	gradient->setZero();
	for (auto j = 0u; j < 32u; ++j)
	{
		auto v = cell[j];
		auto c = m_nodes[field_id][v];
		if (c == std::numeric_limits<double>::max())
		{
			gradient->setZero();
			return std::numeric_limits<double>::max();
		}
		phi += c * N[j];
		(*gradient)(0) += c * (*dN)(j, 0);
		(*gradient)(1) += c * (*dN)(j, 1);
		(*gradient)(2) += c * (*dN)(j, 2);
	}
	gradient->array() *= c0.array();

	return phi;
}

double
CubicLagrangeDiscreteGrid::interpolate(unsigned int field_id, Vector3d const &x,
									   Vector3d *gradient) const
{
	if (!m_domain.contains(x))
		return std::numeric_limits<double>::max();

	auto mi = (x - m_domain.min()).cwiseProduct(m_inv_cell_size).cast<unsigned int>().eval();
	if (mi[0] >= m_resolution[0])
		mi[0] = m_resolution[0] - 1;
	if (mi[1] >= m_resolution[1])
		mi[1] = m_resolution[1] - 1;
	if (mi[2] >= m_resolution[2])
		mi[2] = m_resolution[2] - 1;
	auto i = multiToSingleIndex({{mi(0), mi(1), mi(2)}});
	auto i_ = m_cell_map[field_id][i];
	if (i_ == std::numeric_limits<unsigned int>::max())
		return std::numeric_limits<double>::max();

	auto sd = subdomain(i);
	i = i_;
	auto d = sd.diagonal().eval();

	auto denom = (sd.max() - sd.min()).eval();
	auto c0 = Vector3d::Constant(2.0).cwiseQuotient(denom).eval();
	auto c1 = (sd.max() + sd.min()).cwiseQuotient(denom).eval();
	auto xi = (c0.cwiseProduct(x) - c1).eval();

	auto const &cell = m_cells[field_id][i];
	if (!gradient)
	{
		//auto phi = m_coefficients[field_id][i].dot(shape_function_(xi, nullptr));
		auto phi = 0.0;
		auto N = shape_function_(xi, nullptr);
		for (auto j = 0u; j < 32u; ++j)
		{
			auto v = cell[j];
			auto c = m_nodes[field_id][v];
			if (c == std::numeric_limits<double>::max())
			{
				return std::numeric_limits<double>::max();
			}
			phi += c * N[j];
		}

		return phi;
	}

	auto dN = Matrix<double, 32, 3>{};
	auto N = shape_function_(xi, &dN);

	// TEST
	//auto eps = 1.0e-6;
	//auto ndN = Matrix<double, 32, 3>{};
	//for (auto j = 0u; j < 3u; ++j)
	//{
	//    auto xip = xi;
	//    xip(j) += eps;
	//    auto xim = xi;
	//    xim(j) -= eps;
	//    auto Np = shape_function_(xip, nullptr);
	//    auto Nm = shape_function_(xim, nullptr);
	//    ndN.col(j) = (Np - Nm) / (2.0 * eps);
	//}
	//std::cout << (dN - ndN).cwiseAbs().maxCoeff() /*/ (dN.maxCoeff())*/ << std::endl;
	///

	auto phi = 0.0;
	gradient->setZero();
	for (auto j = 0u; j < 32u; ++j)
	{
		auto v = cell[j];
		auto c = m_nodes[field_id][v];
		if (c == std::numeric_limits<double>::max())
		{
			gradient->setZero();
			return std::numeric_limits<double>::max();
		}
		phi += c * N[j];
		(*gradient)(0) += c * dN(j, 0);
		(*gradient)(1) += c * dN(j, 1);
		(*gradient)(2) += c * dN(j, 2);
	}
	gradient->array() *= c0.array();

	return phi;
}

void CubicLagrangeDiscreteGrid::reduceField(unsigned int field_id, Predicate pred)
{
	auto &coeffs = m_nodes[field_id];
	auto &cells = m_cells[field_id];
	auto keep = std::vector<bool>(coeffs.size());
	for (auto l = 0u; l < coeffs.size(); ++l)
	{
		auto xi = indexToNodePosition(l);
		keep[l] = pred(xi, coeffs[l]) && coeffs[l] != std::numeric_limits<double>::max();
	}

	auto &cell_map = m_cell_map[field_id];
	cell_map.resize(m_n_cells);
	std::iota(cell_map.begin(), cell_map.end(), 0u);

	auto cells_ = cells;
	cells.clear();
	for (auto i = 0u; i < cells_.size(); ++i)
	{
		auto keep_cell = false;
		auto vals = std::vector<double>{};
		for (auto v : cells_[i])
		{
			keep_cell |= keep[v];
			vals.push_back(coeffs[v]);
		}
		if (keep_cell)
		{
			cells.push_back(cells_[i]);
			cell_map[i] = static_cast<unsigned int>(cells.size() - 1);
		}
		else
			cell_map[i] = std::numeric_limits<unsigned int>::max();
	}

	auto n = Matrix<unsigned int, 3, 1>::Map(m_resolution.data());

	auto nv = (n[0] + 1) * (n[1] + 1) * (n[2] + 1);
	auto ne_x = (n[0] + 0) * (n[1] + 1) * (n[2] + 1);
	auto ne_y = (n[0] + 1) * (n[1] + 0) * (n[2] + 1);
	auto ne_z = (n[0] + 1) * (n[1] + 1) * (n[2] + 0);
	auto ne = ne_x + ne_y + ne_z;

	// Reduce vertices.
	auto xi = Vector3d{};
	auto z_values = std::vector<uint64_t>(coeffs.size());
	for (auto l = 0u; l < coeffs.size(); ++l)
	{
		auto xi = indexToNodePosition(l);
		z_values[l] = zValue(xi, 4.0 * m_inv_cell_size.minCoeff());
	}

	std::fill(keep.begin(), keep.end(), false);

	auto vertex_to_cell = std::vector<std::set<std::pair<unsigned int, unsigned int>>>(coeffs.size());
	for (auto c = 0u; c < cells.size(); ++c)
	{
		auto const &cell = cells[c];

		for (auto j = 0u; j < cell.size(); ++j)
		{
			auto v = cell[j];
			keep[v] = true;
			vertex_to_cell[v].insert({c, j});
		}
	}
	auto last_vertex = static_cast<unsigned int>(coeffs.size() - 1);
	for (auto i = static_cast<int>(coeffs.size() - 1); i >= 0; --i)
	{
		if (!keep[i])
		{
			std::swap(coeffs[i], coeffs[last_vertex]);
			std::swap(z_values[i], z_values[last_vertex]);
			std::swap(vertex_to_cell[i], vertex_to_cell[last_vertex]);
			for (auto const &kvp : vertex_to_cell[i])
			{
				cells[kvp.first][kvp.second] = i;
			}
			for (auto const &kvp : vertex_to_cell[last_vertex])
			{
				cells[kvp.first][kvp.second] = last_vertex;
			}

			last_vertex--;
		}
	}
	coeffs.resize(last_vertex + 1);
	z_values.resize(coeffs.size());

	auto sort_pattern = std::vector<unsigned int>(coeffs.size());
	std::iota(sort_pattern.begin(), sort_pattern.end(), 0u);
	std::sort(sort_pattern.begin(), sort_pattern.end(),
			  [&](unsigned int i, unsigned int j) {
				  return z_values[i] < z_values[j];
			  });

	for (auto i = 0u; i < sort_pattern.size(); ++i)
	{
		auto j = sort_pattern[i];
		for (auto const &kvp : vertex_to_cell[j])
		{
			assert(cells[kvp.first][kvp.second] == j);
			cells[kvp.first][kvp.second] = i;
		}
	}

	auto coeffs_ = coeffs;
	std::transform(sort_pattern.begin(), sort_pattern.end(), coeffs.begin(),
				   [&coeffs_](unsigned int i) { return coeffs_[i]; });
}

void CubicLagrangeDiscreteGrid::forEachCell(unsigned int field_id,
											std::function<void(unsigned int, AlignedBox3d const &, unsigned int)> const &cb) const
{
	auto n = m_resolution[0] * m_resolution[1] * m_resolution[2];
	for (auto i = 0u; i < n; ++i)
	{
		auto domain = AlignedBox3d{};
		auto mi = singleToMultiIndex(i);
		domain.min() = m_domain.min() + Matrix<unsigned int, 3, 1>::Map(mi.data()).cast<double>().cwiseProduct(m_cell_size);
		domain.max() = domain.min() + m_cell_size;

		cb(i, domain, 0);
	}
}

static int edgeTable[256]={
        0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
        0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
        0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
        0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
        0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
        0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
        0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
        0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
        0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
        0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
        0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
        0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
        0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
        0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
        0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
        0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
        0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
        0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
        0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
        0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
        0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
        0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
        0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
        0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
        0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
        0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
        0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
        0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
        0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
        0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
        0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
        0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0   };

static int triTable[256][16] =
        {{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
         {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
         {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
         {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
         {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
         {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
         {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
         {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
         {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
         {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
         {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
         {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
         {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
         {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
         {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
         {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
         {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
         {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
         {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
         {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
         {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
         {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
         {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
         {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
         {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
         {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
         {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
         {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
         {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
         {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
         {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
         {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
         {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
         {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
         {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
         {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
         {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
         {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
         {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
         {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
         {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
         {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
         {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
         {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
         {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
         {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
         {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
         {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
         {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
         {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
         {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
         {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
         {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
         {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
         {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
         {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
         {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
         {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
         {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
         {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
         {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
         {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
         {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
         {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
         {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
         {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
         {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
         {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
         {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
         {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
         {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
         {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
         {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
         {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
         {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
         {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
         {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
         {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
         {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
         {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
         {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
         {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
         {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
         {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
         {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
         {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
         {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
         {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
         {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
         {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
         {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
         {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
         {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
         {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
         {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
         {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
         {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
         {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
         {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
         {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
         {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
         {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
         {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
         {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
         {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
         {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
         {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
         {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
         {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
         {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
         {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
         {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
         {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
         {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
         {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
         {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
         {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
         {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
         {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
         {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
         {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
         {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
         {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
         {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
         {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
         {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
         {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
         {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
         {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
         {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
         {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
         {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
         {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
         {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
         {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
         {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
         {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
         {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
         {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
         {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
         {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
         {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
         {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
         {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
         {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
         {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
         {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
         {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
         {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
         {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
         {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
         {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
         {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
         {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
         {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
         {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
         {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
         {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
         {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
         {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
         {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
         {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
         {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
         {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
         {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
         {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
         {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
         {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
         {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
         {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
         {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
         {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
         {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
         {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
         {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
         {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
         {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
         {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
         {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
         {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
         {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
         {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
         {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
         {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
         {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
         {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
         {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
         {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
         {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}};

Eigen::Vector3d interpVertex(double isoLevel, const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, double valp1, double valp2) {
    if (abs(isoLevel - valp1) < 0.00001) return p1;
    if (abs(isoLevel - valp2) < 0.00001) return p2;
    if (abs(valp1 - valp2) < 0.00001) return p1;
    double mu = (isoLevel - valp1) / (valp2 - valp1);
    Eigen::Vector3d p = p1 + mu * (p2 - p1);
    return p;
}

TriangleMesh CubicLagrangeDiscreteGrid::marchingCubes(double isoLevel) {
    auto n = Matrix<unsigned int, 3, 1>::Map(m_resolution.data());
    auto vpos = [&](unsigned int i, unsigned int j, unsigned int k) { return (n[0] + 1)*(n[1] + 1)*k + (n[0] + 1)*j + i; };

    std::vector<Eigen::Vector3d> vertices;
    std::vector<std::array<unsigned int, 3>> indices;

    size_t batch_size = 1024;
    size_t n_batches = m_n_cells / batch_size;
#pragma omp parallel for
    for (auto bi = 0u; bi < n_batches; ++bi)
    {
        std::vector<Eigen::Vector3d> vertices_buffer;
        size_t ci_end = std::min(batch_size*(bi+1), m_n_cells);
        for (size_t ci = batch_size*bi; ci < ci_end; ++ci) {
            auto mi = singleToMultiIndex(ci);
            double val[8];
            unsigned int i = mi[0], j = mi[1], k = mi[2];
            val[0] = m_nodes[0][vpos(i+0,j+0,k+0)];
            val[1] = m_nodes[0][vpos(i+0,j+1,k+0)];
            val[2] = m_nodes[0][vpos(i+1,j+1,k+0)];
            val[3] = m_nodes[0][vpos(i+1,j+0,k+0)];
            val[4] = m_nodes[0][vpos(i+0,j+0,k+1)];
            val[5] = m_nodes[0][vpos(i+0,j+1,k+1)];
            val[6] = m_nodes[0][vpos(i+1,j+1,k+1)];
            val[7] = m_nodes[0][vpos(i+1,j+0,k+1)];

            int cubeindex = 0;

            if (val[0] < isoLevel) cubeindex |= 1;
            if (val[1] < isoLevel) cubeindex |= 2;
            if (val[2] < isoLevel) cubeindex |= 4;
            if (val[3] < isoLevel) cubeindex |= 8;
            if (val[4] < isoLevel) cubeindex |= 16;
            if (val[5] < isoLevel) cubeindex |= 32;
            if (val[6] < isoLevel) cubeindex |= 64;
            if (val[7] < isoLevel) cubeindex |= 128;

            /* Cube is entirely in/out of the surface */
            if (edgeTable[cubeindex] == 0) continue;

            Eigen::Vector3d p[8];
            p[0] = m_domain.min() + m_cell_size.cwiseProduct(Eigen::Vector3d(i+0,j+0,k+0));
            p[1] = m_domain.min() + m_cell_size.cwiseProduct(Eigen::Vector3d(i+0,j+1,k+0));
            p[2] = m_domain.min() + m_cell_size.cwiseProduct(Eigen::Vector3d(i+1,j+1,k+0));
            p[3] = m_domain.min() + m_cell_size.cwiseProduct(Eigen::Vector3d(i+1,j+0,k+0));
            p[4] = m_domain.min() + m_cell_size.cwiseProduct(Eigen::Vector3d(i+0,j+0,k+1));
            p[5] = m_domain.min() + m_cell_size.cwiseProduct(Eigen::Vector3d(i+0,j+1,k+1));
            p[6] = m_domain.min() + m_cell_size.cwiseProduct(Eigen::Vector3d(i+1,j+1,k+1));
            p[7] = m_domain.min() + m_cell_size.cwiseProduct(Eigen::Vector3d(i+1,j+0,k+1));

            Eigen::Vector3d vertlist[12];

            /* Find the vertices where the surface intersects the cube */
            if (edgeTable[cubeindex] & 1)
                vertlist[0] = interpVertex(isoLevel, p[0], p[1], val[0], val[1]);
            if (edgeTable[cubeindex] & 2)
                vertlist[1] = interpVertex(isoLevel, p[1], p[2], val[1], val[2]);
            if (edgeTable[cubeindex] & 4)
                vertlist[2] = interpVertex(isoLevel, p[2], p[3], val[2], val[3]);
            if (edgeTable[cubeindex] & 8)
                vertlist[3] = interpVertex(isoLevel, p[3], p[0], val[3], val[0]);
            if (edgeTable[cubeindex] & 16)
                vertlist[4] = interpVertex(isoLevel, p[4], p[5], val[4], val[5]);
            if (edgeTable[cubeindex] & 32)
                vertlist[5] = interpVertex(isoLevel, p[5], p[6], val[5], val[6]);
            if (edgeTable[cubeindex] & 64)
                vertlist[6] = interpVertex(isoLevel, p[6], p[7], val[6], val[7]);
            if (edgeTable[cubeindex] & 128)
                vertlist[7] = interpVertex(isoLevel, p[7], p[4], val[7], val[4]);
            if (edgeTable[cubeindex] & 256)
                vertlist[8] = interpVertex(isoLevel, p[0], p[4], val[0], val[4]);
            if (edgeTable[cubeindex] & 512)
                vertlist[9] = interpVertex(isoLevel, p[1], p[5], val[1], val[5]);
            if (edgeTable[cubeindex] & 1024)
                vertlist[10] = interpVertex(isoLevel, p[2], p[6], val[2], val[6]);
            if (edgeTable[cubeindex] & 2048)
                vertlist[11] = interpVertex(isoLevel, p[3], p[7], val[3], val[7]);

            /* Create the triangle */
            for (int t=0;triTable[cubeindex][t]!=-1;t+=3) {
                vertices_buffer.push_back(vertlist[triTable[cubeindex][t  ]]);
                vertices_buffer.push_back(vertlist[triTable[cubeindex][t+1]]);
                vertices_buffer.push_back(vertlist[triTable[cubeindex][t+2]]);
            }
        }

#pragma omp critical
        vertices.insert(vertices.end(), vertices_buffer.begin(), vertices_buffer.end());
    }

    indices.resize(vertices.size()/3);
    for (unsigned int i = 0; i < vertices.size()/3; i++) {
        indices[i] = {3*i, 3*i+1, 3*i+2};
    }

    TriangleMesh trimesh(vertices, indices);
    return trimesh;
}

} // namespace Discregrid
