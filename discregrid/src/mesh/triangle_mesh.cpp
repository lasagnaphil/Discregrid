
#include <mesh/triangle_mesh.hpp>

#include <unordered_set>
#include <cassert>
#include <fstream>
#include <iostream>

using namespace Eigen;

namespace
{
template <class T>
inline void hash_combine(std::size_t& seed, const T& v)
{
	std::hash<T> hasher;
	seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}
}

namespace Discregrid
{

struct HalfedgeHasher
{
	HalfedgeHasher(std::vector<Eigen::Vector3i> const& faces_)
		: faces(&faces_){}

	std::size_t operator()(Halfedge const& he) const
	{
		int f = he.face();
		int e = he.edge();
		std::array<int, 2> v = { (*faces)[f][e], (*faces)[f][(e + 1) % 3] };
		if (v[0] > v[1])
			std::swap(v[0], v[1]);

		std::size_t seed(0);
		hash_combine(seed, v[0]);
		hash_combine(seed, v[1]);
		return seed;
	}

	std::vector<Eigen::Vector3i> const* faces;
};

struct HalfedgeEqualTo
{
	HalfedgeEqualTo(std::vector<Eigen::Vector3i> const& faces_)
		: faces(&faces_){}

	bool operator()(Halfedge const& a, Halfedge const& b) const
	{
		int fa = a.face();
		int ea = a.edge();
		std::array<int, 2> va = { (*faces)[fa][ea], (*faces)[fa][(ea + 1) % 3] };

		int fb = b.face();
		int eb = b.edge();
		std::array<int, 2> vb = { (*faces)[fb][eb], (*faces)[fb][(eb + 1) % 3] };

		return va[0] == vb[1] && va[1] == vb[0];
	}

	std::vector<Eigen::Vector3i> const* faces;
};

typedef std::unordered_set<Halfedge, HalfedgeHasher, HalfedgeEqualTo>
	FaceSet;

TriangleMesh::TriangleMesh(
	std::vector<Vector3r> const& vertices,
	std::vector<Eigen::Vector3i> const& faces)
	: m_faces(faces), m_e2e(3 * faces.size()), m_vertices(vertices)
	, m_v2e(vertices.size())
{
	construct();
}

TriangleMesh::TriangleMesh(real const* vertices,
	int const* faces,
	std::size_t nv, std::size_t nf)
	: m_faces(nf), m_vertices(nv), m_e2e(3 * nf), m_v2e(nv)
{
	std::copy(vertices, vertices + 3 * nv, m_vertices[0].data());
	std::copy(faces, faces + 3 * nf, m_faces[0].data());
	construct();
}



TriangleMesh::TriangleMesh(std::string const& path)
{
	std::ifstream in(path, std::ios::in);
	if (!in) 
	{ 
	    std::cerr << "Cannot open " << path << std::endl; 
	    return;
	}

	std::string line;
	while (getline(in, line)) {
	    if (line.substr(0, 2) == "v ") {
	        std::istringstream s(line.substr(2));
	        Vector3r v; s >> v.x(); s >> v.y(); s >> v.z();
	        m_vertices.push_back(v);
	    }
	    else if (line.substr(0, 2) == "f ") {
	        std::istringstream s(line.substr(2));
	        Eigen::Vector3i f;
	        for (int j(0); j < 3; ++j)
	        {
	            std::string buf;
	            s >> buf;
	            buf = buf.substr(0, buf.find_first_of('/'));
	            f[j] = std::stoi(buf) - 1;
	        }
	        m_faces.push_back(f);
	    }
	    else if (line[0] == '#') { /* ignoring this line */ }
	    else { /* ignoring this line */ }
	}

	construct();
}

void
TriangleMesh::exportOBJ(std::string const& filename) const
{
	auto outfile = std::ofstream(filename.c_str());
	auto str_stream = std::stringstream(std::stringstream::in);

	outfile << "g default" << std::endl;
	for (auto const& pos : m_vertices)
	{
		outfile << "v " << pos[0] << " " << pos[1] << " " << pos[2] << "\n";
	}
	
	for (auto const& f : m_faces)
	{
		outfile << "f";
		for (int i = 0; i < 3; i++)
			outfile << " " << f[i] + 1;
		outfile << std::endl;
	}

	outfile.close();
}

void
TriangleMesh::construct()
{

	m_e2e.resize(3 * m_faces.size());
	m_v2e.resize(m_vertices.size());

	// Build adjacencies for mesh faces.
	FaceSet face_set(
		(m_faces.size() * 3) / 2,
		HalfedgeHasher(m_faces),
		HalfedgeEqualTo(m_faces));
	for (int i(0); i < m_faces.size(); ++i)
		for (unsigned char j(0); j < 3; ++j)
		{
		Halfedge he(i, j);
		auto ret = face_set.insert(he);
		if (!ret.second)
		{
			m_e2e[he.face()][he.edge()] = *(ret.first);
			m_e2e[ret.first->face()][ret.first->edge()] = he;

			face_set.erase(ret.first);
		}

		m_v2e[m_faces[i][j]] = he;
		}

	m_b2e.reserve(face_set.size());

	for (Halfedge const he : face_set)
	{
		m_b2e.push_back(he);
		Halfedge b(static_cast<int>(m_b2e.size()) - 1u, 3);
		m_e2e[he.face()][he.edge()] = b;
		m_v2e[target(he)] = b;

		assert(source(b) == target(he));
	}

#ifdef _DEBUG
	for (int i(0); i < nFaces(); ++i)
	{
		Halfedge h(i, 0);
		for (int j(0); j < 3; ++j)
		{
			assert(faceVertex(i, j) == source(h));
			h = h.next();
		}
	}
#endif

	if (!m_b2e.empty())
	{
		std::cout << std::endl << "WARNING: Mesh not closed!" << std::endl;
	}
}

Vector3r
TriangleMesh::computeFaceNormal(int f) const
{
	Vector3r const& x0 = vertex(faceVertex(f, 0));
	Vector3r const& x1 = vertex(faceVertex(f, 1));
	Vector3r const& x2 = vertex(faceVertex(f, 2));

	return (x1 - x0).cross(x2 - x0).normalized();
}

}

