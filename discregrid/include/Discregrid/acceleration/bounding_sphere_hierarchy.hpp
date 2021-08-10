#pragma once

#include "types.hpp"
#include "bounding_sphere.hpp"
#include "kd_tree.hpp"

namespace Discregrid
{

class TriangleMeshBSH : public KDTree<BoundingSphere>
{

public:

	using super = KDTree<BoundingSphere>;

	TriangleMeshBSH(std::vector<Vector3r> const& vertices,
		std::vector<Eigen::Vector3i> const& faces);

	Vector3r const& entityPosition(int i) const final;
	void computeHull(int b, int n, BoundingSphere& hull) const final;

private:

	std::vector<Vector3r> const& m_vertices;
	std::vector<Eigen::Vector3i> const& m_faces;

	std::vector<Vector3r> m_tri_centers;
};

class TriangleMeshBBH : public KDTree<AlignedBox3r>
{
public:

	using super = KDTree<AlignedBox3r>;

	TriangleMeshBBH(std::vector<Vector3r> const& vertices,
		std::vector<Eigen::Vector3i> const& faces);

	Vector3r const& entityPosition(int i) const final;
	void computeHull(int b, int n, AlignedBox3r& hull) const final;

private:

	std::vector<Vector3r> const& m_vertices;
	std::vector<Eigen::Vector3i> const& m_faces;

	std::vector<Vector3r> m_tri_centers;


};

class PointCloudBSH : public KDTree<BoundingSphere>
{

public:

	using super = KDTree<BoundingSphere>;

	PointCloudBSH();
	PointCloudBSH(std::vector<Vector3r> const& vertices);

	Vector3r const& entityPosition(int i) const final;
	void computeHull(int b, int n, BoundingSphere& hull)
		const final;

private:

	std::vector<Vector3r> const* m_vertices;
};

}
