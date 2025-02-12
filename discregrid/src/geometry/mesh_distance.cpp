
#include <geometry/mesh_distance.hpp>
#include <mesh/triangle_mesh.hpp>
#include "point_triangle_distance.hpp"

#include <limits>
#include <functional>
#include <omp.h>

using namespace Eigen;

namespace Discregrid
{

MeshDistance::MeshDistance(const TriangleMesh* mesh, bool precompute_normals)
	: m_bsh(mesh->vertex_data(), mesh->face_data()), m_mesh(mesh)
	, m_precomputed_normals(precompute_normals)
{
	auto max_threads = omp_get_max_threads();

	m_bsh.construct();

	if (m_precomputed_normals)
	{
		m_face_normals.resize(m_mesh->nFaces());
		m_vertex_normals.resize(mesh->nVertices(), Vector3r::Zero());
		std::transform(m_mesh->faces().begin(), m_mesh->faces().end(),
			m_face_normals.begin(),
			[&](Eigen::Vector3i const& face)
			{
				auto const& x0 = m_mesh->vertex(face[0]);
				auto const& x1 = m_mesh->vertex(face[1]);
				auto const& x2 = m_mesh->vertex(face[2]);

				auto n = (x1 - x0).cross(x2 - x0).normalized();

				auto e1 = (x1 - x0).normalized();
				auto e2 = (x2 - x1).normalized();
				auto e3 = (x0 - x2).normalized();

				auto alpha = Vector3r{
					std::acos(e1.dot(-e3)), 
					std::acos(e2.dot(-e1)),
					std::acos(e3.dot(-e2)) };
				m_vertex_normals[face[0]] += alpha[0] * n;
				m_vertex_normals[face[1]] += alpha[1] * n;
				m_vertex_normals[face[2]] += alpha[2] * n;

				return n;
			}
		);
	}
}

// Thread-safe.
real
MeshDistance::distance(Vector3r const& x, Vector3r* nearest_point,
	int* nearest_face, NearestEntity* ne) const
{
	using namespace std::placeholders;

	auto dist_candidate = std::numeric_limits<real>::max();
	auto pred = [&](int node_index, int)
	{
		return predicate(node_index, m_bsh, x, dist_candidate);
	};

	auto cb = [&](int node_index, int)
	{
		return callback(node_index, m_bsh, x, dist_candidate, *nearest_face);
	};

	auto pless = [&](std::array<int, 2> const& c)
	{
		//return true;
		auto const& hull0 = m_bsh.hull(c[0]);
		auto const& hull1 = m_bsh.hull(c[1]);
		auto d0_2 = (x - hull0.x()).norm() - hull0.r();
		auto d1_2 = (x - hull1.x()).norm() - hull1.r();
		return d0_2 < d1_2;
	};

	m_bsh.traverseDepthFirst(pred, cb, pless);

	if (nearest_point)
	{
		auto t = std::array<Vector3r const*, 3>{
			&m_mesh->vertex(m_mesh->faceVertex(*nearest_face, 0)),
			&m_mesh->vertex(m_mesh->faceVertex(*nearest_face, 1)),
			&m_mesh->vertex(m_mesh->faceVertex(*nearest_face, 2))
		};
		auto np = Vector3r{};
		auto ne_ = NearestEntity{};
		auto dist2_ = point_triangle_sqdistance(x, t, &np, &ne_);
		dist_candidate = std::sqrt(dist2_);
		if (ne)
			*ne = ne_;
		if (nearest_point)
			*nearest_point = np;
	}
	return dist_candidate;
}

bool
MeshDistance::predicate(int node_index,
	TriangleMeshBSH const& bsh,
	Vector3r const& x,
	real& dist_candidate) const
{
	// If the furthest point on the current candidate hull is closer than the closest point on the next hull then we can skip it
	auto const& hull = bsh.hull(node_index);
	auto const& hull_radius = hull.r();
	auto const& hull_center = hull.x();

	const auto dist_sq_to_center = (x - hull_center).squaredNorm();

	if (dist_candidate > hull_radius) {
		const auto l = dist_candidate - hull_radius;
		if (l * l > dist_sq_to_center)
			dist_candidate = std::sqrt(dist_sq_to_center) + hull_radius;
	}

	const auto d = dist_candidate + hull_radius;
	return dist_sq_to_center <= d * d;
}

void
MeshDistance::callback(int node_index,
	TriangleMeshBSH const& bsh,
	Vector3r const& x,
	real& dist_candidate,
    int& nearest_face) const
{
	auto const& node = m_bsh.node(node_index);
	auto const& hull = m_bsh.hull(node_index);

	if (!node.isLeaf())
		return;

	auto r = hull.r();

	auto temp = (x - hull.x()).eval();
	auto d_center2 = temp[0] * temp[0] + temp[1] * temp[1] + temp[2] * temp[2];

	auto temp_ = dist_candidate + r;
	if (d_center2 > temp_ * temp_)
		return;

	auto dist_candidate_2 = dist_candidate * dist_candidate;
	auto changed = false;
	for (auto i = node.begin; i < node.begin + node.n; ++i)
	{
		auto f = m_bsh.entity(i);
		auto t = std::array<Vector3r const*, 3>{
			&m_mesh->vertex(m_mesh->faceVertex(f, 0)),
			&m_mesh->vertex(m_mesh->faceVertex(f, 1)),
			&m_mesh->vertex(m_mesh->faceVertex(f, 2))
		};
		auto dist2_ = point_triangle_sqdistance(x, t);
		if (dist_candidate_2 > dist2_)
		{
			dist_candidate_2 = dist2_;
			changed = true;
            nearest_face = f;
		}
	}
	if (changed)
	{
		dist_candidate = std::sqrt(dist_candidate_2);
	}
}

real
MeshDistance::signedDistance(Vector3r const& x, Vector3r* nearest_point, Vector3r* normal) const
{
	int nf;
	auto ne = NearestEntity{};
	auto np = Vector3r{};
	auto dist = distance(x, &np, &nf, &ne);
	
	auto n = Vector3r{};
	switch (ne)
	{
	case NearestEntity::VN0:
		n = vertex_normal(m_mesh->faceVertex(nf, 0));
		break;
	case NearestEntity::VN1:
		n = vertex_normal(m_mesh->faceVertex(nf, 1));
		break;
	case NearestEntity::VN2:
		n = vertex_normal(m_mesh->faceVertex(nf, 2));
		break;
	case NearestEntity::EN0:
		n = edge_normal({static_cast<unsigned int>(nf), 0});
		break;
	case NearestEntity::EN1:
		n = edge_normal({static_cast<unsigned int>(nf), 1});
		break;
	case NearestEntity::EN2:
		n = edge_normal({static_cast<unsigned int>(nf), 2});
		break;
	case NearestEntity::FN:
		n = face_normal(nf);
		break;
	default:
		n.setZero();
		break;
	}

	if ((x - np).dot(n) < 0.0)
		dist *= -1.0;

	if (nearest_point)
	    *nearest_point = np;

	if (normal)
	    *normal = n;

	return dist;
}

real
MeshDistance::unsignedDistance(Vector3r const & x) const
{
	return distance(x);
}

Vector3r
MeshDistance::face_normal(int f) const
{
	if (m_precomputed_normals)
		return m_face_normals[f];

	auto const& x0 = m_mesh->vertex(m_mesh->faceVertex(f, 0));
	auto const& x1 = m_mesh->vertex(m_mesh->faceVertex(f, 1));
	auto const& x2 = m_mesh->vertex(m_mesh->faceVertex(f, 2));

	return (x1 - x0).cross(x2 - x0).normalized();
}

Vector3r
MeshDistance::edge_normal(Halfedge const& h) const
{
	auto o = m_mesh->opposite(h);

	if (m_precomputed_normals)
	{
		if (o.isBoundary()) return m_face_normals[h.face()];
		return m_face_normals[h.face()] + m_face_normals[o.face()];
	}

	if (o.isBoundary()) return face_normal(h.face());
	return face_normal(h.face()) + face_normal(o.face());
}

Vector3r
MeshDistance::vertex_normal(int v) const
{
	if (m_precomputed_normals)
		return m_vertex_normals[v];

	auto const& x0 = m_mesh->vertex(v);
	auto n = Vector3r{}; n.setZero();
	for (auto h : m_mesh->incident_faces(v))
	{
		assert(m_mesh->source(h) == v);
		auto ve0 = m_mesh->target(h);
		auto e0 = (m_mesh->vertex(ve0) - x0).eval();
		e0.normalize();
		auto ve1 = m_mesh->target(h.next());
		auto e1 = (m_mesh->vertex(ve1) - x0).eval();
		e1.normalize();
		auto alpha = std::acos((e0.dot(e1)));
		n += alpha * e0.cross(e1);
	}
	return n;
}

}
