#pragma once

#include <vector>
#include <fstream>
#include <array>
#include <Eigen/Dense>

#include "types.hpp"

namespace Discregrid
{

class DiscreteGrid
{
public:

	using CoefficientVector = Eigen::Matrix<real, 32, 1>;
	using ContinuousFunction = std::function<real(Vector3r const&)>;
	using MultiIndex = Eigen::Vector3i;
	using Predicate = std::function<bool(Vector3r const&, real)>;
	using SamplePredicate = std::function<bool(Vector3r const&)>;

	DiscreteGrid() = default;
	DiscreteGrid(AlignedBox3r const& domain, Eigen::Vector3i const& resolution)
		: m_domain(domain), m_resolution(resolution), m_n_fields(0u)
	{
		m_cell_size = domain.diagonal().cwiseQuotient(resolution.cast<real>());
		m_inv_cell_size = m_cell_size.cwiseInverse();
		m_n_cells = resolution.prod();
	}
	DiscreteGrid(AlignedBox3r const& minimum_domain, Vector3r const& cell_size)
	    : m_cell_size(cell_size), m_n_fields(0u)
	{
        m_inv_cell_size = m_cell_size.cwiseInverse();
	    Vector3r res_f = minimum_domain.diagonal().cwiseProduct(m_inv_cell_size);
	    Eigen::Vector3i res = res_f.array().ceil().cast<int>();
	    m_resolution[0] = res[0]; m_resolution[1] = res[1]; m_resolution[2] = res[2];
	    Vector3r domain_size = m_cell_size.cwiseProduct(res.cast<real>());
	    Vector3r domain_center = minimum_domain.center();
	    m_domain = AlignedBox3r(domain_center - 0.5*domain_size, domain_center + 0.5*domain_size);
	    m_n_cells = res.prod();
	}
	virtual ~DiscreteGrid() = default;

	virtual void save(std::string const& filename) const = 0;
	virtual void load(std::string const& filename) = 0;

	virtual int addFunction(ContinuousFunction const& func, bool verbose = false,
		SamplePredicate const& pred = nullptr) = 0;

	real interpolate(Vector3r const& xi, Vector3r* gradient = nullptr) const
	{
		return interpolate(0u, xi, gradient);
	}

	virtual real interpolate(int field_id, Vector3r const& xi,
		Vector3r* gradient = nullptr) const = 0;

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
	virtual bool determineShapeFunctions(int field_id, Vector3r const &x,
		std::array<int, 32> &cell, Vector3r &c0, Eigen::Matrix<real, 32, 1> &N,
		Eigen::Matrix<real, 32, 3> *dN = nullptr) const = 0;

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
	virtual real interpolate(int field_id, Vector3r const& xi, const std::array<int, 32> &cell, const Vector3r &c0, const Eigen::Matrix<real, 32, 1> &N,
		Vector3r* gradient = nullptr, Eigen::Matrix<real, 32, 3> *dN = nullptr) const = 0;

	virtual void reduceField(int field_id, Predicate pred) {}


	MultiIndex singleToMultiIndex(int i) const;
	int multiToSingleIndex(MultiIndex const& ijk) const;

	AlignedBox3r subdomain(MultiIndex const& ijk) const;
	AlignedBox3r subdomain(int l) const;

	AlignedBox3r const& domain() const { return m_domain; }
	Eigen::Vector3i const& resolution() const { return m_resolution; };
	Vector3r const& cellSize() const { return m_cell_size;}
	Vector3r const& invCellSize() const { return m_inv_cell_size;}

protected:


	AlignedBox3r m_domain;
	Eigen::Vector3i m_resolution;
	Vector3r m_cell_size;
	Vector3r m_inv_cell_size;
	std::size_t m_n_cells;
	std::size_t m_n_fields;
};
}