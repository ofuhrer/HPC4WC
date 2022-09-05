#ifndef MESH
#define MESH

#include <Eigen/Dense>
#include <memory> //shared_ptr
#include <cstddef> //size_t
#include <gmsh.h> // need the library file to work, either gmsh-4.1.0-dll AND gmsh.lib for windows, or libgmsh.so for UNIX. Also need to add "-lgmsh" option to your compiler.
#include "walltime.hpp"

typedef Eigen::Matrix<std::size_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXst;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXd;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXi;
typedef Eigen::Matrix<std::size_t, 2, 1> Vector2st;
typedef Eigen::Matrix<std::size_t, 3, 1> Vector3st;
typedef Eigen::Matrix<std::size_t, Eigen::Dynamic, 1> VectorXst;
//class for triangular and rectangular meshes
class Mesh
{
public: 

    Mesh() {

    } //default constructor (dummy) 

    Mesh(const double x0, const double lx, const double y0, const double ly, double size_fac, bool rect, const std::string& filename);  // Construct from a mesh file, typically "testmesh.msh"
    Mesh(std::size_t nx, std::size_t ny, const double x0, const double lx, const double y0, const double ly);  // Construct rectangular mesh from regular grid

    ~Mesh() {
	//free memory, maybe not neccessary
    }
     
    std::size_t get_num_elements() const; //num cells
    std::size_t get_num_edges() const;
    std::size_t get_num_nodes() const;
    std::size_t get_num_neighbours() const;


    //input: index of cell 
    //output: vector of size with index of each neighbouring cell  
    Eigen::VectorXi get_all_neighbours(const std::size_t index) const;
    int get_neighbour(const std::size_t index, const std::size_t side) const;
    Eigen::Vector2i get_edge_elements(const std::size_t edge) const;
    Vector2st get_edge_nodes(const std::size_t edge) const;

    //returns Matrix of size (num_vertices_per_cell,2) containing the coordinates of each vertex in a cell
    MatrixXd get_cellvertices(const std::size_t cell_id) const;

    //returns vertex coordinate of vertex_id being part of cell cell_id 
    MatrixXd get_cellvertice(const std::size_t cell_id, const unsigned int vertex_id) const;

    MatrixXi get_lookup_table() const; // triangular grid: returns three indices per element (I think the table is ordered by element indices); rectangular grid: returns 4 per element (???)
    MatrixXd get_lookuptable_coeff() const;
    MatrixXd get_nodes() const; // returns x- and y-coordinate of each node

    //double get_area(const std::size_t cell_index) const;
    //Eigen::VectorXd get_areas() const;

    //return edge length 
    double get_edge_length(const std::size_t edge_index) const; 
    void laplacian(const Eigen::VectorXd& in_field, Eigen::VectorXd& out_field) const; 


    MatrixXd get_centers() const; //return centers
    double get_squared_dist(size_t i, size_t j) const; // return squared distance
    double get_dist(size_t i, size_t j) const; // return squared distance
    double get_diff(size_t i, size_t j, size_t k) const; // return x or y difference
    double get_normal(size_t i, size_t j, size_t k) const; // return x or y coordinate of unit normal vector

    VectorXst get_elementEdges(const std::size_t cell_id) const;

    std::size_t get_elementEdge(const std::size_t cell_id, const unsigned int edge_id) const;
    std::size_t get_inv_gilbert(size_t i, size_t j) { return inverse_gilbert[i][j]; } // will crash if used with unstructured mesh
    bool get_has_inv_gilbert() { return has_inverse_gilbert; } // will crash if used with unstructured mesh
    void reindex_cleverly(double x0, double y0, double x1, double y1, int maxiter);

          
private:
    MatrixXst elementNodes; // matrix of size numelementsx(3 or 4 nodes per element) with index of nodes*
    MatrixXd nodes; // (x_coorindate, y_coordinate)
    MatrixXi lookup_table; // three neighbours for triangle, four neighbours for rectangles, if element touches a boundary then this side has a -1
    MatrixXd lookup_table_coeff;
    MatrixXd centers; //(x_coord, y_coord)

    std::vector<std::vector<std::size_t>> inverse_gilbert; // only for mesh from structured mesh, to avoid using interpolation.
    bool has_inverse_gilbert;

    std::size_t numNodes;
    std::size_t numEdges;
    std::size_t numElements;

    int numNodePerElement;  // int because it has to go through a gmsh method. In 2D its equal to the number of edge per element.

    void _update_node_grid(size_t i, size_t j,                     
                           size_t& nodeCounter, 
                           double x0, double y0, double sx, double sy, 
                           std::vector<std::vector<int>>& nodegrid);

    Eigen::Vector2d compute_center(std::size_t cell_id);

    //return surface area of a cell with cell_index
    double compute_area(const std::size_t cell_index) const;
};

#endif
