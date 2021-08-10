#pragma once

#include "types.hpp"
#include "halfedge.hpp"
#include "entity_containers.hpp"

#include <vector>
#include <array>
#include <cassert>
#include <string>

#include <Eigen/Dense>

namespace Discregrid
{

class TriangleMesh
{

public:

	TriangleMesh(std::vector<Vector3r> const& vertices,
		std::vector<Eigen::Vector3i> const& faces);

	TriangleMesh(real const* vertices,
		int const* faces,
		std::size_t nv, std::size_t nf);

	TriangleMesh(std::string const& filename);

	void exportOBJ(std::string const& filename) const;

	// Halfedge modifiers.
	int source(Halfedge const h) const
	{
		if (h.isBoundary()) return target(opposite(h));
		return m_faces[h.face()][h.edge()];
	}
	int target(Halfedge const h) const
	{
		if (h.isBoundary()) return source(opposite(h));
		return source(h.next());
	}
	Halfedge opposite(Halfedge const h) const
	{
		if (h.isBoundary()) return m_b2e[h.face()];
		return m_e2e[h.face()][h.edge()];
	}

	// Container getters.
	FaceContainer faces() { return FaceContainer(this); }
	FaceConstContainer faces() const { return FaceConstContainer(this); }
	IncidentFaceContainer incident_faces(int v) const {
		return IncidentFaceContainer(v, this); }
	VertexContainer vertices() { return VertexContainer(this); }
	VertexConstContainer vertices() const { return VertexConstContainer(this); }

	// Entity size getters.
	std::size_t nFaces() const { return m_faces.size(); }
	std::size_t nVertices() const { return m_v2e.size(); }
	std::size_t nBorderEdges() const { return m_b2e.size(); }

	// Entity getters.
	int const& faceVertex(int f, int i) const
	{
		assert(i < 3);
		assert(f < m_faces.size());
		return m_faces[f][i];
	}
	int& faceVertex(int f, int i)
	{
		assert(i < 3);
		assert(f < m_faces.size());
		return m_faces[f][i];
	}

	Vector3r const& vertex(int i) const { return m_vertices[i]; }
	Vector3r& vertex(int i) { return m_vertices[i]; }
	Eigen::Vector3i const& face(int i) const {
		return m_faces[i]; }
	Eigen::Vector3i& face(int i) {
		return m_faces[i];
	}
	Halfedge incident_halfedge(int v) const { return m_v2e[v]; }

	// Data getters.
	std::vector<Vector3r> const& vertex_data() const {
		return m_vertices; }
	std::vector<Vector3r>& vertex_data() { return m_vertices; }
	std::vector<Eigen::Vector3i> const& face_data() const {
		return m_faces; }
	std::vector<Eigen::Vector3i>& face_data() { return m_faces; }

	Vector3r computeFaceNormal(int f) const;

private:

	void construct();

private:

	std::vector<Vector3r> m_vertices;
	std::vector<Eigen::Vector3i> m_faces;
	std::vector<std::array<Halfedge, 3>> m_e2e;
	std::vector<Halfedge> m_v2e;
	std::vector<Halfedge> m_b2e;
};
}

