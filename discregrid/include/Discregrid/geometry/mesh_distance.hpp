#pragma once

#include <types.hpp>
#include <Discregrid/mesh/triangle_mesh.hpp>

namespace std {
	template <> struct hash<Discregrid::Vector3r>
	{
		std::size_t operator()(Discregrid::Vector3r const& x) const
		{
			std::size_t seed = 0;
			std::hash<Discregrid::real> hasher;
			seed ^= hasher(x[0]) + 0x9e3779b9 + (seed<<6) + (seed>>2);
			seed ^= hasher(x[1]) + 0x9e3779b9 + (seed<<6) + (seed>>2);
			seed ^= hasher(x[2]) + 0x9e3779b9 + (seed<<6) + (seed>>2);
			return seed;
		}
	};

	template <> struct less<Discregrid::Vector3r>
	{
		bool operator()(Discregrid::Vector3r const& left, Discregrid::Vector3r const& right) const
		{
			for (auto i = 0u; i < 3u; ++i)
			{
				if (left(i) < right(i))
					return true;
				else if (left(i) > right(i))
					return false;
			}
			return false;
		}
	};
}
#include <Discregrid/utility/lru_cache.hpp>

#include <Discregrid/acceleration/bounding_sphere_hierarchy.hpp>


#include <array>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>

#include <Eigen/Dense>

namespace Discregrid
{

enum class NearestEntity;
class TriangleMesh;
class Halfedge;
class MeshDistance
{

	struct Candidate
	{
		bool operator<(Candidate const& other) const { return b < other.b; }
		int node_index;
		real b, w;
	};

public:
	MeshDistance(const TriangleMesh* mesh, bool precompute_normals = true);

	// Returns the shortest unsigned distance from a given point x to
	// the stored mesh.
	// Thread-safe function.
	real distance(Vector3r const& x, Vector3r* nearest_point = nullptr,
		int* nearest_face = nullptr, NearestEntity* ne = nullptr) const;

	// Requires a closed two-manifold mesh as input data.
	// Thread-safe function.
	real signedDistance(Vector3r const& x,
                          Vector3r* nearest_point = nullptr, Vector3r* normal = nullptr) const;

	real unsignedDistance(Vector3r const& x) const;

private:

	Vector3r vertex_normal(int v) const;
	Vector3r edge_normal(Halfedge const& h) const;
	Vector3r face_normal(int f) const;

	void callback(int node_index, TriangleMeshBSH const& bsh,
		Vector3r const& x,
		real& dist, int& nearest_face) const;

	bool predicate(int node_index, TriangleMeshBSH const& bsh,
		Vector3r const& x, real& dist) const;

private:

	const TriangleMesh* m_mesh;
	TriangleMeshBSH m_bsh;

	using FunctionValueCache = LRUCache<Vector3r, real>;

	std::vector<Vector3r> m_face_normals;
	std::vector<Vector3r> m_vertex_normals;
	bool m_precomputed_normals;
};

}

