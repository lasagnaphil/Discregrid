
#include <acceleration/bounding_sphere_hierarchy.hpp>

#include <iostream>
#include <set>

using namespace Eigen;

namespace Discregrid
{

TriangleMeshBSH::TriangleMeshBSH(
        std::span<const Vector3r> vertices,
        std::span<const Eigen::Vector3i> faces)
	: super(faces.size()), m_faces(faces), m_vertices(vertices),
		m_tri_centers(faces.size())
{
	std::transform(m_faces.begin(), m_faces.end(), m_tri_centers.begin(),
		[&](Vector3i const& f)
		{
			return 1.0 / 3.0 * (m_vertices[f[0]] + m_vertices[f[1]] +
				m_vertices[f[2]]);
		});
}

Vector3r const&
TriangleMeshBSH::entityPosition(int i) const
{
	return m_tri_centers[i];
}

void
TriangleMeshBSH::computeHull(int b, int n, BoundingSphere& hull) const
{
	auto vertices_subset = std::vector<Vector3r>(3 * n);
	for (int i(0); i < n; ++i)
	{
		auto const& f = m_faces[m_lst[b + i]];
		{
			vertices_subset[3 * i + 0] = m_vertices[f[0]];
			vertices_subset[3 * i + 1] = m_vertices[f[1]];
			vertices_subset[3 * i + 2] = m_vertices[f[2]];
		}
	}

	const BoundingSphere s(vertices_subset);

	hull.x() = s.x();
	hull.r() = s.r();
}

TriangleMeshBBH::TriangleMeshBBH(
	std::span<const Vector3r> vertices,
	std::span<const Vector3i> faces)
	: super(faces.size()), m_faces(faces), m_vertices(vertices), 
		m_tri_centers(faces.size())
{
	std::transform(m_faces.begin(), m_faces.end(), m_tri_centers.begin(),
		[&](Vector3i const& f)
		{
			return 1.0 / 3.0 * (m_vertices[f[0]] + m_vertices[f[1]] +
				m_vertices[f[2]]);
		});
}

Vector3r const&
TriangleMeshBBH::entityPosition(int i) const
{
	return m_tri_centers[i];
}

void
TriangleMeshBBH::computeHull(int b, int n, AlignedBox3r& hull) const
{
	for (auto i = 0u; i < n; ++i)
	{
		auto const& f = m_faces[m_lst[b + i]];
        hull.extend(m_vertices[f[0]]);
        hull.extend(m_vertices[f[1]]);
        hull.extend(m_vertices[f[2]]);
	}
}


PointCloudBSH::PointCloudBSH()
	: super(0)
{
}

PointCloudBSH::PointCloudBSH(std::span<const Vector3r> vertices)
	: super(vertices.size()), m_vertices(vertices)
{

}

Vector3r const&
PointCloudBSH::entityPosition(int i) const
{
	return m_vertices[i];
}

void
PointCloudBSH::computeHull(int b, int n, BoundingSphere& hull) const
{
	auto vertices_subset = std::vector<Vector3r>(n);
	for (int i = b; i < n + b; ++i)
		vertices_subset[i - b] = m_vertices[m_lst[i]];

	const BoundingSphere s(vertices_subset);

	hull.x() = s.x();
	hull.r() = s.r();
}



}
