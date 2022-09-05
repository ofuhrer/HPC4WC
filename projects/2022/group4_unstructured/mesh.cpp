#include <Eigen/Dense>
#include <memory> //shared_ptr
#include <cstddef> //size_t
#include <gmsh.h> // need the library file to work, either gmsh-4.1.0-dll AND gmsh.lib for windows, or libgmsh.so for UNIX. Also need to add "-lgmsh" option to your compiler.
#include "mesh.hpp"
#include "gilbert.cpp"
#include <iostream>
#include <algorithm> // sort
#include <numeric> // iota

#define boundary_distance_x 2
#define boundary_distance_y 2
#define boundary_distance 2
#define boundary_value 0

typedef Eigen::Matrix<std::size_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXst;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXd;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXi;
typedef Eigen::Matrix<std::size_t, 2, 1> Vector2st;
typedef Eigen::Matrix<std::size_t, 3, 1> Vector3st;
typedef Eigen::Matrix<std::size_t, Eigen::Dynamic, 1> VectorXst;


//*is the index+1 the tag (are the tags ordered?) 

//gmsh data format (each element has the following informations: 
//unique identifier
//type (e.g. 15 points, 1 are lines, 2 triangles, 3 rectangles) 
//number of tags
//tag e.g. physical entity it belomgs to e.g. top, bottom
//remaining entries: unique identifier to points belonging to entity


//circumcenters
Eigen::Vector2d Mesh::compute_center(std::size_t cell_id)
{
    //distinguish between rectangular and triangular grids 
    Eigen::Vector2d center; 
    switch(numNodePerElement) 
    {
        case 3: {//triangular grid 
            
        Eigen::Matrix<double, 3, 2, Eigen::RowMajor> thisElementNodes;
            
            for (size_t j = 0; j < numNodePerElement; j++)
            {
                    thisElementNodes.row(j) = nodes.row(elementNodes(cell_id, j));
            }
            
                const Eigen::VectorXd norms = thisElementNodes.rowwise().squaredNorm(); //norms of the three vertice coordinate	
                Eigen::VectorXd diff_y(3); //(y2-y3, y3-y1, y1-y2)
                diff_y << thisElementNodes(1,1)-thisElementNodes(2,1), thisElementNodes(2,1)-thisElementNodes(0,1), thisElementNodes(0,1)-thisElementNodes(1,1);
            

                Eigen::VectorXd diff_x(3);
                diff_x << thisElementNodes(2,0)-thisElementNodes(1,0), thisElementNodes(0,0)-thisElementNodes(2,0), thisElementNodes(1,0)-thisElementNodes(0,0); 
                
                //formula from Wikipedia about circumference 
                const double d = 2*thisElementNodes.col(0).dot(diff_y); 

            center(0) = norms.dot(diff_y)/d;
                center(1) = norms.dot(diff_x)/d;      
            
            break;
        }
            case 4: //rectangular grid 
                    center =  0.25 * (nodes.row(elementNodes(cell_id, 3)) + nodes.row(elementNodes(cell_id, 2)) + nodes.row(elementNodes(cell_id, 1)) + nodes.row(elementNodes(cell_id, 0)));
        break;
    }
    return center;
}


std::string bissect(double x, double left, double midpoint, double right, int maxiter=7, std::string s="")
{
    if (x > midpoint){
        s.append("1");
        if (s.size() < maxiter) {
            return bissect(x, midpoint, (right + midpoint) / 2, right, maxiter, s);
        }
        else {
            return s;
        }
    }
    else {
        s.append("0");
        if (s.size() < maxiter) {
            return bissect(x, left, (left + midpoint) / 2, midpoint, maxiter, s);
        }
        else {
            return s;
        } 
    }
}


int add_and_convert(std::string x, std::string y)
{
    std::string s = "";
    for (size_t i = 0; i < x.size(); i++)
    {
        s.push_back(y[i]);
        s.push_back(x[i]);
    }
    return std::stoi(s, 0, 2);
}


std::vector<size_t> argsort(const std::vector<int> &array) {  // stolen on https://gist.github.com/HViktorTsoi/58eabb4f7c5a303ced400bcfa816f6f5
    std::vector<size_t> indices(array.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&array](int left, int right) -> bool {
                  // sort indices according to corresponding array element
                  return array[left] < array[right];
              });

    return indices;
}


void Mesh::reindex_cleverly(double x0, double y0, double x1, double y1, int maxiter)
{
    double t1 = wall_time();
    if (has_inverse_gilbert) { return; }
    std::vector<int> a (numElements);
    std::string xval, yval;
    for (size_t i = 0; i < numElements; i++)
    {
        xval = bissect(centers(i, 0), x0, (x1 - x0) / 2, x1, maxiter);
        yval = bissect(centers(i, 1), y0, (y1 - y0) / 2, y1, maxiter);
        a[i] = add_and_convert(xval, yval);
    }

    std::vector<size_t> new_indices = argsort(a);

    MatrixXst tmp = elementNodes(new_indices, Eigen::all);
    elementNodes = tmp;
    MatrixXi tmp1 = lookup_table(new_indices, Eigen::all);
    lookup_table = tmp1;
    MatrixXd tmp2 = lookup_table_coeff(new_indices, Eigen::all);
    lookup_table_coeff = tmp2;
    MatrixXd tmp3 = centers(new_indices, Eigen::all);
    centers = tmp3;
    int k, l;
    
    for (size_t i = 0; i < numElements; i++)
    {
        for(unsigned int j = 0; j < numNodePerElement; ++j)
        {
		k = lookup_table(i,j);
		if(k != -1) 
			lookup_table(i,j) = new_indices[k]; 


        }
    }
    double t2 = wall_time();
    std::cout << "Time to reindex cleverly : " << t2 - t1 << " seconds." << std::endl; 
}


Mesh::Mesh(const double x0, const double lx, const double y0, const double ly, double size_fac=1, bool rect=false, const std::string& filename="../testmesh.msh") {  // Construct from a mesh file, typically "testmesh.msh"
    double t1 = wall_time();
    gmsh::initialize();  // Always this first, and finalize() last. This is why I probably have to do all of this in one block
    // gmsh::open(filename); 
    int rect_id = gmsh::model::occ::addRectangle(-50, -50, 0, 100, 100);
    gmsh::model::occ::synchronize();
    if (rect) {gmsh::model::mesh::setRecombine(2, rect_id);}
    gmsh::option::setNumber("Mesh.MeshSizeFactor", size_fac);
    gmsh::model::mesh::generate(2);
    double t2 = wall_time();
    std::cout << "Time for gmsh to create mesh : " << t2 - t1 << " seconds." << std::endl; 
    gmsh::write(filename);
    t1 = wall_time();
    std::cout << "Time for gmsh to write mesh : " << t1 - t2 << " seconds." << std::endl; 
    std::vector<int> types;  // All gmsh::model::mesh methods are void, need to preallocate and send reference
    gmsh::model::mesh::getElementTypes(types); // types is typically (1, 2 or 3, 15), 1 is lines, 2 triangles, 3 rectangles and 15 are points

    std::vector<std::size_t> node_tags;
    std::vector<double> node_coords;
    std::vector<double> parametric_coords; //what are those? 
    gmsh::model::mesh::getNodes(node_tags, node_coords, parametric_coords, -1, -1, false, false);

    /////// get vertice coordinates (no id) is included ////////////
    numNodes = node_tags.size(); // Is one-ordered !
    nodes = MatrixXd::Zero(numNodes, 2); // can finally give it a size  
    for (size_t i = 0; i < node_tags.size(); i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            nodes(i, j) = node_coords[3 * i + j];  // Ditch z coordinate, 3 is for 3d not triangle
        }
    }
    ////////////////////////////////////////////////////////////////////
    
    // Explore gmsh file to get forward tables (elements to edges and nodes)

    int dim, order, numPrimaryNodes;
    std::string elementName;
    std::vector<double> localNodeCoords;
    std::vector<std::size_t> elementTags, elementNodeTags, elementEdgeNodeTags;
    for (const auto& type : types)
    {
        gmsh::model::mesh::getElementProperties(type, elementName, dim, order, numNodePerElement, localNodeCoords, numPrimaryNodes);
        if (dim == 2) {
            // Only 2D elements for the rest. For now it won't work if the mesh file contains several types of 2d elements (a hybrid rectangle-triangle mesh for example)
        
            gmsh::model::mesh::getElementsByType(type, elementTags, elementNodeTags); //elementNodeTags: vector of size N*num_nodes_for_entity N=entities with unique node identifiers/ elementTags = unique identifiers of the elements (e.g.triangles) 
           
            gmsh::model::mesh::getElementEdgeNodes(type, elementEdgeNodeTags); //How is this structured? 
            numElements = elementTags.size();
            break;
        }
    }
        
    gmsh::finalize();

    ////////////////////////////////////// retrieve information from gmsh ////////////////////////////////////////// 

    // cast what I can to Eigen Matrix 
    std::cout << "Num node per element: " <<  numNodePerElement << std::endl;

    elementNodes = MatrixXst::Zero(numElements, numNodePerElement); //element == cell
    for (size_t i = 0; i < numElements; i++)
    {
        for (size_t j = 0; j < numNodePerElement; j++)
        {
            elementNodes(i, j) = elementNodeTags[numNodePerElement * i + j] - 1; // was one-ordered
        }
    }

    //Create backward tables 
    std::vector<std::vector<std::size_t>> uniqueEdges;
    std::vector<std::vector<int>> edgeToElement; // can have -1, will iterate over ints as well for consistency
    lookup_table_coeff = MatrixXd::Zero(numElements, numNodePerElement); // 
    lookup_table = MatrixXi(numElements, numNodePerElement);    
    centers = MatrixXd::Zero(numElements,2);
    std::size_t n11, n12, n21, n22; //node tag of edge 
    double distance;
    double edge_length;
    double area;

    for (int i = 0; i < numElements; i++)
    {
        area = compute_area(i);
        for (int j = 0; j < numNodePerElement; j++)
        {
            n11 = elementEdgeNodeTags[2 * numNodePerElement * i + 2 * j + 0] - 1; // was one-ordered
            n12 = elementEdgeNodeTags[2 * numNodePerElement * i + 2 * j + 1] - 1; // was one-ordered
            bool found = false;
            int k = 0;
            while (!found && k < uniqueEdges.size()) //unique edges is not initialized 
            {
                found = ((n11 == uniqueEdges[k][0] && n12 == uniqueEdges[k][1]) || (n12 == uniqueEdges[k][0] && n11 == uniqueEdges[k][1]));
                k++;
            }
            if (!found)
            {
                uniqueEdges.push_back({n11, n12});
            }
            k = i + 1;
            while (!found && k < numElements)
            {
                int l = 0;
                while (!found && l < numNodePerElement)
                {
                    n21 = elementEdgeNodeTags[2 * numNodePerElement * k + 2 * l + 0] - 1; // was one-ordered
                    n22 = elementEdgeNodeTags[2 * numNodePerElement * k + 2 * l + 1] - 1; // was one-ordered
                    found = ((n11 == n21 && n12 == n22) || (n12 == n21 && n11 == n22));
                    if (found) 
                    {                        
   			            centers.row(i) = compute_center(i);  
    	                centers.row(k) = compute_center(k);
                        distance = (centers.row(i)-centers.row(k)).squaredNorm();
                        edge_length = (nodes.row(n11)-nodes.row(n12)).norm();  
                        lookup_table(i, j) = k;
                        lookup_table(k, l) = i;
   			            lookup_table_coeff(i, j) = edge_length / (distance * area); 
                        lookup_table_coeff(k, l) = lookup_table_coeff(i, j);
                    }
                    l++;
                }
                k++;
            }
            if (!found) // if no matching edge : we are at the boundary
            {
                lookup_table(i, j) = -1;
                edge_length = (nodes.row(n11)-nodes.row(n12)).norm();
                lookup_table_coeff(i, j) = edge_length/(boundary_distance*area);
                
            }
        }
                
    }

    has_inverse_gilbert = false;
    t2 = wall_time();
    std::cout << "Time to create lookup table : " << t2 - t1 << " seconds. The mesh has " << numElements << " elements." << std::endl; 
}

void Mesh::_update_node_grid(size_t i, size_t j,
                             size_t& nodeCounter, 
                             double x0, double y0, double sx, double sy, 
                             std::vector<std::vector<int>>& nodegrid)
{
    if (nodegrid[i][j] == -1) 
    {
        nodes(nodeCounter, 0) = x0 + i * sx;
        nodes(nodeCounter, 1) = y0 + j * sy;
        nodegrid[i][j] = nodeCounter;
        nodeCounter += 1;
    }
    return;
}

//constructor for rectangular meshes, much easier for benchmarking as we do not need to generate rectangular meshes in gmsh and convert such that we can pass them to the laplace function from the lecture 
Mesh::Mesh(std::size_t nx, std::size_t ny, const double x0, const double lx, const double y0, const double ly)
{
    double t1 = wall_time();
    /* 
    As a convention, x0, y0, etc mark the edges of the region covered by the mesh defined by the constructor to remain consistent with gmsh. This means the bottomleft corner will be
    at (x0, y0), but the bottomleftmost point in the grid (a.k.a. the "center" of the bottomleft square) will (x0 + 0.5 * lx / nx, y0 + 0.5 * ly / ny). 
    */

    numNodePerElement = 4;
    numNodes = (nx + 1) * (ny + 1); 
    numEdges = nx * (ny + 1) + ny * (nx + 1);
    numElements = nx * ny;

    nodes = MatrixXd::Zero(numNodes, 2);
    elementNodes = MatrixXst::Zero(numElements, numNodePerElement);
    lookup_table_coeff = MatrixXd::Zero(numElements, numNodePerElement); //store edge_length/(distance*area) coefficient //3
    lookup_table = MatrixXi(numElements, numNodePerElement);
    
    centers = MatrixXd::Zero(numElements, 2);

    std::vector<std::vector<std::size_t>> gilbert = gilbert2d(nx, ny);
    inverse_gilbert = std::vector<std::vector<std::size_t>> (nx, std::vector<std::size_t> (ny, 0));
    has_inverse_gilbert = true;

    std::cout << "gilbert worked fine" << std::endl;
        
    float sx = lx / (float)(nx);
    float sy = ly / (float)(ny);
    std::cout << "sx " << sx << std::endl;
    std::cout << "sy " << sy << std::endl;

    std::vector<std::vector<int>> nodegrid(nx + 1, std::vector<int> (ny + 1, -1));
    std::vector<std::vector<int>> vedgegrid(nx + 1, std::vector<int> (ny, -1));
    std::vector<std::vector<int>> hedgegrid(nx, std::vector<int> (ny + 1, -1));
    size_t nodeCounter = 0;
    size_t edgeCounter = 0;
    std::vector<std::vector<int>> clockwise {
                { 0, 0 },
                { 1, 0 },
                { 1, 1 },
                { 0, 1 }
            };

    // node loop
    
    std::cout << "gilbert size: "<< gilbert.size() << std::endl; // 3500 * 2
    std::cout << "inverse gilbert size: " << inverse_gilbert.size() << ", " << inverse_gilbert[0].size() << std::endl; // 50 * 70
    for (size_t k = 0; k < gilbert.size(); k++)
    {
        std::size_t i, j = 0;
        if (nx >= ny) {
            i = gilbert[k][0];
            j = gilbert[k][1];
        }
        else {
            j = gilbert[k][0];
            i = gilbert[k][1];
        }
        inverse_gilbert[i][j] = k;
        size_t elementNodeCounter = 0;
        for (auto const& pair : clockwise)
        {
            _update_node_grid(i + pair[0], j + pair[1], nodeCounter, 
                               x0, y0, sx, sy, nodegrid);
            elementNodes(k, elementNodeCounter) = nodegrid[i + pair[0]][j + pair[1]];
            elementNodeCounter += 1;
        }
        centers(k, 0) = x0 + (i + 0.5) * sx;
        centers(k, 1) = y0 + (j + 0.5) * sy;
    }

    std::cout << "computation of distances and centers worked fine" << std::endl;
    
    // edge loop : need all nodes already stored and inverse gilbert
    int neighbour = 0;
    int thisedge = 0;
    double edge_length;
    double area =sx*sy;
    double distance;
    for (int k = 0; k < gilbert.size(); k++)
    {
        std::size_t i, j = 0;
        if (nx >= ny) {
            i = gilbert[k][0];
            j = gilbert[k][1];
        }
        else {
            j = gilbert[k][0];
            i = gilbert[k][1];
        }

        if (hedgegrid[i][j] == -1) // bottom
        {
            hedgegrid[i][j] = edgeCounter;
            edgeCounter += 1;
        }
        thisedge = hedgegrid[i][j];
        neighbour = (j == 0 ? -1 : inverse_gilbert[i][j - 1]);
           
        lookup_table(k, 0) = neighbour;
        edge_length = (nodes(elementNodes(k,0),0) == nodes(elementNodes(k,1),0)) ? sy : sx;
        distance = (neighbour == -1) ? boundary_distance : ((centers.row(k) - centers.row(neighbour)).squaredNorm()); 
        lookup_table_coeff(k,0) = edge_length/(area*distance); 

        if (vedgegrid[i + 1][j] == -1) // right
        {
            vedgegrid[i + 1][j] = edgeCounter;
            edgeCounter += 1;
        } 

        thisedge = vedgegrid[i + 1][j];
        neighbour = (i == nx - 1 ? -1 : inverse_gilbert[i + 1][j]);
        
        lookup_table(k, 1) = neighbour;
        edge_length = (nodes(elementNodes(k,1),0) == nodes(elementNodes(k,2),0)) ? sy : sx; 
        distance = (neighbour == -1) ? boundary_distance : ((centers.row(k) - centers.row(neighbour)).squaredNorm());
        lookup_table_coeff(k, 1) = edge_length/(area*distance);

        if (hedgegrid[i][j + 1] == -1) // top
        {
            hedgegrid[i][j + 1] = edgeCounter;
            edgeCounter += 1;
        }

        thisedge = hedgegrid[i][j + 1];
        neighbour = (j == ny - 1 ? -1 : inverse_gilbert[i][j + 1]);
        
        lookup_table(k, 2) = neighbour;
        edge_length = (nodes(elementNodes(k,1),0) == nodes(elementNodes(k,2),0)) ? sy : sx; 
        distance = (neighbour == -1) ? boundary_distance : ((centers.row(k) - centers.row(neighbour)).squaredNorm());
        lookup_table_coeff(k, 2) = edge_length/(area*distance);
        
        if (vedgegrid[i][j] == -1) // left
        {
            vedgegrid[i][j] = edgeCounter;
            edgeCounter += 1;
        }

        thisedge = vedgegrid[i][j];
        neighbour = (i == 0 ? -1 : inverse_gilbert[i - 1][j]);
        
        lookup_table(k, 3) = neighbour;
        edge_length = (nodes(elementNodes(k,1),0) == nodes(elementNodes(k,2),0)) ? sy : sx; 
        distance = (neighbour == -1) ? boundary_distance : ((centers.row(k) - centers.row(neighbour)).squaredNorm());
        lookup_table_coeff(k, 3) = edge_length/(area*distance);
    }

    double t2 = wall_time();
    std::cout << "Time to explicitly construct square mesh : " << t2 - t1 << " seconds." << std::endl; 
}


/////////////////////////////////// member functions ////////////////////////////////////////////////////////////

std::size_t Mesh::get_num_elements() const 
{
	return numElements;
}

std::size_t Mesh::get_num_nodes() const
{
    return numNodes;
}

std::size_t Mesh::get_num_neighbours() const
{
   return numNodePerElement;
}

Eigen::VectorXi Mesh::get_all_neighbours(const std::size_t cell_index) const
{
    return lookup_table.row(cell_index); //neighbours;
}

int Mesh::get_neighbour(const std::size_t index, const std::size_t side) const
{
    return int(lookup_table(index, side));
}

MatrixXd Mesh::get_cellvertice(const std::size_t cell_id, unsigned int vertice_id) const {

	return nodes.row(elementNodes(cell_id, vertice_id));

}

MatrixXd Mesh::get_cellvertices(const std::size_t cell_id) const {
    MatrixXd cell_vertices(numNodePerElement, 2);
    for (size_t i = 0; i < numNodePerElement; i++)
    {
         cell_vertices.row(i) =  nodes.row(elementNodes(cell_id,i));
 
    }
    return cell_vertices;
}

//return vector of areas for each cell/element
double Mesh::compute_area(const std::size_t cell_index) const {
    
	const MatrixXd cell_vertices = get_cellvertices(cell_index);

	//distinguish rectangles and triangles dont know whats the correct gmsh function
	switch(numNodePerElement) {

		case 3:
		//area of triangle using determinant method
		  return 0.5*std::abs((cell_vertices(0,0)*(cell_vertices(1,1)-cell_vertices(2,1))+cell_vertices(1,0)*(cell_vertices(2,1) - cell_vertices(0,1))+cell_vertices(2,0)*(cell_vertices(0,1) - cell_vertices(1,1)))); 
		  break;

		case 4: //rectangle 
			return (0.5*(3*cell_vertices.row(0) - cell_vertices.row(1) - cell_vertices.row(2) - cell_vertices.row(3))).cwiseAbs().prod();
            break;
	}
    return 0.;  // made my compiler angry
}

MatrixXi Mesh::get_lookup_table() const{
    return lookup_table;
}

MatrixXd Mesh::get_lookuptable_coeff() const {

	return lookup_table_coeff;
}


MatrixXd Mesh::get_nodes() const {
    return nodes;
}

MatrixXd Mesh::get_centers() const //return centers
{
    return centers;
}

void Mesh::laplacian(const Eigen::VectorXd& field_prev, Eigen::VectorXd& out_field) const
{
    int neighbour; 
       
    for(std::size_t i = 0; i < numElements; ++i){  // this loop can be optimized
        out_field(i) = 0; 
        for(std::size_t j = 0; j < numNodePerElement; ++j){  
            neighbour =  lookup_table(i,j);
            if (neighbour == -1)
                out_field(i) += lookup_table_coeff(i, j) * (boundary_value - field_prev(i));
            else
                out_field(i) += lookup_table_coeff(i, j) * (field_prev(neighbour) - field_prev(i));
	
       }
    }
}
