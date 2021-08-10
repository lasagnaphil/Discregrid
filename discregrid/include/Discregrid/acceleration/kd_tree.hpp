#pragma once

#include <vector>
#include <functional>
#include <algorithm>
#include <numeric>
#include <queue>
#include <iostream>

#include <Eigen/Dense>

#include <array>
#include <list>
#include <stack>

namespace Discregrid
{

template <typename HullType>
class KDTree
{
public:

    using TraversalPredicate = std::function<bool (int node_index, int depth)> ;
    using TraversalCallback = std::function <void (int node_index, int depth)>;
    using TraversalPriorityLess = std::function<bool (std::array<int, 2> const& nodes)>;

    struct Node
    {
        Node(int b_, int n_)
            : children({{-1, -1}})
            , begin(b_), n(n_) {}

        Node() = default;

        bool isLeaf() const { return children[0] < 0 && children[1] < 0; }

        // Index of child nodes in nodes array.
        // -1 if child does not exist.
        std::array<int, 2> children;

        // Index according entries in entity list.
        int begin;

        // Number of owned entries.
        int n;
    };

    struct QueueItem { int n, d; };
    using TraversalQueue = std::queue<QueueItem>;

    KDTree(std::size_t n)
        : m_lst(n) {}

    virtual ~KDTree() {}

    Node const& node(int i) const { return m_nodes[i]; }
    HullType const& hull(int i) const { return m_hulls[i]; }
    int entity(int i) const { return m_lst[i]; }

    void construct();
    void update();
    void traverseDepthFirst(TraversalPredicate pred, TraversalCallback cb,
        TraversalPriorityLess const& pless = nullptr) const;
    void traverseBreadthFirst(TraversalPredicate const& pred, TraversalCallback const& cb, int start_node = 0, TraversalPriorityLess const& pless = nullptr, TraversalQueue& pending = TraversalQueue()) const;

protected:

    void construct(int node, AlignedBox3r const& box,
        int b, int n);
    void traverseDepthFirst(int node, int depth,
        TraversalPredicate pred, TraversalCallback cb, TraversalPriorityLess const& pless) const;
    void traverseBreadthFirst(TraversalQueue& pending,
        TraversalPredicate const& pred, TraversalCallback const& cb, TraversalPriorityLess const& pless = nullptr) const;

    int addNode(int b, int n);

    virtual Vector3r const& entityPosition(int i) const = 0;
    virtual void computeHull(int b, int n, HullType& hull) const = 0;

protected:

    std::vector<int> m_lst;

    std::vector<Node> m_nodes;
    std::vector<HullType> m_hulls;
};

#include "kd_tree.inl"
}

