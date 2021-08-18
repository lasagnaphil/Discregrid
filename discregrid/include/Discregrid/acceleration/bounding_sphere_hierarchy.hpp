#pragma once

#include "types.hpp"
#include "bounding_sphere.hpp"
#include "kd_tree.hpp"

#include <span.hpp>

namespace Discregrid
{

class TriangleMeshBSH : public KDTree<BoundingSphere>
{

public:

	using super = KDTree<BoundingSphere>;

	TriangleMeshBSH(std::span<const Vector3r> vertices,
		std::span<const Eigen::Vector3i> faces);

	Vector3r const& entityPosition(int i) const final;
	void computeHull(int b, int n, BoundingSphere& hull) const final;

private:

    std::span<const Vector3r> m_vertices;
    std::span<const Eigen::Vector3i> m_faces;

	std::vector<Vector3r> m_tri_centers;
};

class TriangleMeshBBH : public KDTree<AlignedBox3r>
{
public:

	using super = KDTree<AlignedBox3r>;

	TriangleMeshBBH(std::span<const Vector3r> vertices,
		std::span<const Eigen::Vector3i> faces);

	Vector3r const& entityPosition(int i) const final;
	void computeHull(int b, int n, AlignedBox3r& hull) const final;

private:

    std::span<const Vector3r> m_vertices;
    std::span<const Eigen::Vector3i> m_faces;

	std::vector<Vector3r> m_tri_centers;


};

class PointCloudBSH : public KDTree<BoundingSphere>
{

public:

	using super = KDTree<BoundingSphere>;

	PointCloudBSH();
	PointCloudBSH(std::span<const Vector3r> vertices);

	Vector3r const& entityPosition(int i) const final;
	void computeHull(int b, int n, BoundingSphere& hull)
		const final;

private:

    std::span<const Vector3r> m_vertices;
};

}
