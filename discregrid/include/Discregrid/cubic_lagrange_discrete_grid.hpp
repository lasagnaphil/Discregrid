#pragma once

#include "discrete_grid.hpp"
#include "mesh/triangle_mesh.hpp"

namespace Discregrid
{

class CubicLagrangeDiscreteGrid : public DiscreteGrid
{
public:

    CubicLagrangeDiscreteGrid() : DiscreteGrid() {}
	CubicLagrangeDiscreteGrid(std::string const& filename);
	CubicLagrangeDiscreteGrid(AlignedBox3r const& domain,
		Eigen::Vector3i const& resolution);
	CubicLagrangeDiscreteGrid(AlignedBox3r const& minimum_domain,
							  Vector3r const& cell_size);

	void save(std::string const& filename) const override;
	void load(std::string const& filename) override;

	int addFunction(ContinuousFunction const& func, bool verbose = false,
		SamplePredicate const& pred = nullptr) override;


	std::size_t nCells() const { return m_n_cells; };
	real interpolate(int field_id, Vector3r const& xi,
		Vector3r* gradient = nullptr) const override;

	/**
	 * @brief Determines the shape functions for the discretization with ID field_id at point xi.
	 * 
	 * @param field_id Discretization ID
	 * @param x Location where the shape functions should be determined
	 * @param cell cell of x
	 * @param c0 vector required for the interpolation
	 * @param N	shape functions for the cell of x
	 * @param dN (Optional) derivatives of the shape functions, required to compute the gradient
	 * @return Success of the function.
	 */
	bool determineShapeFunctions(int field_id, Vector3r const &x,
		std::array<int, 32> &cell, Vector3r &c0, Eigen::Matrix<real, 32, 1> &N,
		Eigen::Matrix<real, 32, 3> *dN = nullptr) const override;

	/**
	 * @brief Evaluates the given discretization with ID field_id at point xi.
	 * 
	 * @param field_id Discretization ID
	 * @param xi Location where the discrete function is evaluated
	 * @param cell cell of xi
	 * @param c0 vector required for the interpolation
	 * @param N	shape functions for the cell of xi
	 * @param gradient (Optional) if a pointer to a vector is passed the gradient of the discrete function will be evaluated
	 * @param dN (Optional) derivatives of the shape functions, required to compute the gradient
	 * @return real Results of the evaluation of the discrete function at point xi
	 */
	real interpolate(int field_id, Vector3r const& xi, const std::array<int, 32> &cell, const Vector3r &c0, const Eigen::Matrix<real, 32, 1> &N,
		Vector3r* gradient = nullptr, Eigen::Matrix<real, 32, 3> *dN = nullptr) const override;

	void reduceField(int field_id, Predicate pred) override;

	void forEachCell(int field_id,
		std::function<void(int, AlignedBox3r const&, int)> const& cb) const;

	TriangleMesh marchingCubes(real isoLevel);

private:

	Vector3r indexToNodePosition(int l) const;


private:

	std::vector<std::vector<real>> m_nodes;
	std::vector<std::vector<std::array<int, 32>>> m_cells;
	std::vector<std::vector<int>> m_cell_map;
};

}