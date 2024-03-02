#include "mesh.h"
#include <iostream>
#include <igl/read_triangle_mesh.h>
#include <Eigen/Sparse>


HEdge::HEdge(bool b) {
	mBoundary = b;

	mTwin = nullptr;
	mPrev = nullptr;
	mNext = nullptr;

	mStart = nullptr;
	mFace = nullptr;

	mFlag = false;
	mValid = true;
}

HEdge* HEdge::twin() const {
	return mTwin;
}

HEdge* HEdge::setTwin(HEdge* e) {
	mTwin = e;
	return mTwin;
}

HEdge* HEdge::prev() const {
	return mPrev;
}

HEdge* HEdge::setPrev(HEdge* e) {
	mPrev = e;
	return mPrev;
}

HEdge* HEdge::next() const {
	return mNext;
}

HEdge* HEdge::setNext(HEdge* e) {
	mNext = e;
	return mNext;
}

Vertex* HEdge::start() const {
	return mStart;
}

Vertex* HEdge::setStart(Vertex* v) {
	mStart = v;
	return mStart;
}

Vertex* HEdge::end() const {
	return mNext->start();
}

Face* HEdge::leftFace() const {
	return mFace;
}

Face* HEdge::setFace(Face* f) {
	mFace = f;
	return mFace;
}

bool HEdge::flag() const {
	return mFlag;
}

bool HEdge::setFlag(bool b) {
	mFlag = b;
	return mFlag;
}

bool HEdge::isBoundary() const {
	return mBoundary;
}

bool HEdge::isValid() const {
	return mValid;
}

bool HEdge::setValid(bool b) {
	mValid = b;
	return mValid;
}

OneRingHEdge::OneRingHEdge(const Vertex* v) {
	if (v == nullptr) {
		mStart = nullptr;
		mNext = nullptr;
	} else {
		mStart = v->halfEdge();
		mNext = v->halfEdge();
	}
}

HEdge* OneRingHEdge::nextHEdge() {
	HEdge* ret = mNext;
	if (mNext != nullptr && mNext->prev()->twin() != mStart) {
		mNext = mNext->prev()->twin();
	} else {
		mNext = nullptr;
	}
	return ret;
}

OneRingVertex::OneRingVertex(const Vertex* v): ring(v) {
}

Vertex* OneRingVertex::nextVertex() {
	HEdge* he = ring.nextHEdge();
	return he != nullptr ? he->end() : nullptr;
}

Vertex::Vertex() : mHEdge(nullptr), mFlag(0) {
	mPosition = Eigen::Vector3f::Zero();
	mColor = VCOLOR_BLUE;
	mNormal = Eigen::Vector3f::Zero();
}

Vertex::Vertex(const Eigen::Vector3f& v): mPosition(v), mHEdge(nullptr), mFlag(0) {
	mColor = VCOLOR_BLUE;
	mNormal = Eigen::Vector3f::Zero();
}

Vertex::Vertex(float x, float y, float z): mHEdge(nullptr), mFlag(0) {
	mPosition = Eigen::Vector3f(x, y, z);
	mColor = VCOLOR_BLUE;
	mNormal = Eigen::Vector3f::Zero();
}


const Eigen::Vector3f& Vertex::position() const {
	return mPosition;
}

const Eigen::Vector3f& Vertex::setPosition(const Eigen::Vector3f& p) {
	mPosition = p;
	return mPosition;
}

const Eigen::Vector3f& Vertex::normal() const {
	return mNormal;
}

const Eigen::Vector3f& Vertex::setNormal(const Eigen::Vector3f& n) {
	mNormal = n;
	return mNormal;
}

const Eigen::Vector3f& Vertex::color() const {
	return mColor;
}

const Eigen::Vector3f& Vertex::setColor(const Eigen::Vector3f& c) {
	mColor = c;
	return mColor;
}

HEdge* Vertex::halfEdge() const {
	return mHEdge;
}

HEdge* Vertex::setHalfEdge(HEdge* he) {
	mHEdge = he;
	return mHEdge;
}

int Vertex::index() const {
	return mIndex;
}

int Vertex::setIndex(int i) {
	mIndex = i;
	return mIndex;
}

int Vertex::flag() const {
	return mFlag;
}

int Vertex::setFlag(int f) {
	mFlag = f;
	return mFlag;
}

bool Vertex::isValid() const {
	return mValid;
}

bool Vertex::setValid(bool b) {
	mValid = b;
	return mValid;
}

bool Vertex::isBoundary() const {
	OneRingHEdge ring(this);
	HEdge* curr = nullptr;
	while (curr = ring.nextHEdge()) {
		if (curr->isBoundary()) {
			return true;
		}
	}
	return false;
}

int Vertex::valence() const {
	int count = 0;
	OneRingVertex ring(this);
	Vertex* curr = nullptr;
	while (curr = ring.nextVertex()) {
		++count;
	}
	return count;
}

Face::Face() : mHEdge(nullptr), mValid(true) {
}

HEdge* Face::halfEdge() const {
	return mHEdge;
}

HEdge* Face::setHalfEdge(HEdge* he) {
	mHEdge = he;
	return mHEdge;
}

bool Face::isBoundary() const {
	HEdge* curr = mHEdge;
	do {
		if (curr->twin()->isBoundary()) {
			return true;
		}
		curr = curr->next();
	} while (curr != mHEdge);
	return false;
}

bool Face::isValid() const {
	return mValid;
}

bool Face::setValid(bool b) {
	mValid = b;
	return mValid;
}

Mesh::Mesh() {
	mVertexPosFlag = true;
	mVertexNormalFlag = true;
	mVertexColorFlag = true;
}

Mesh::~Mesh() {
	clear();
}

const std::vector< HEdge* >& Mesh::edges() const {
	return mHEdgeList;
}

const std::vector< HEdge* >& Mesh::boundaryEdges() const {
	return mBHEdgeList;
}

const std::vector< Vertex* >& Mesh::vertices() const {
	return mVertexList;
}

const std::vector< Face* >& Mesh::faces() const {
	return mFaceList;
}


bool Mesh::isVertexPosDirty() const {
	return mVertexPosFlag;
}

void Mesh::setVertexPosDirty(bool b) {
	mVertexPosFlag = b;
}

bool Mesh::isVertexNormalDirty() const {
	return mVertexNormalFlag;
}

void Mesh::setVertexNormalDirty(bool b) {
	mVertexNormalFlag = b;
}

bool Mesh::isVertexColorDirty() const {
	return mVertexColorFlag;
}

void Mesh::setVertexColorDirty(bool b) {
	mVertexColorFlag = b;
}

bool Mesh::loadMeshFile(const std::string filename) {
	// Use libigl to parse the mesh file
	bool iglFlag = igl::read_triangle_mesh(filename, mVertexMat, mFaceMat);
	if (iglFlag) {
		clear();

		// Construct the half-edge data structure.
		int numVertices = mVertexMat.rows();
		int numFaces = mFaceMat.rows();

		// Fill in the vertex list
		for (int vidx = 0; vidx < numVertices; ++vidx) {
			mVertexList.push_back(new Vertex(mVertexMat(vidx, 0),
			                                 mVertexMat(vidx, 1),
			                                 mVertexMat(vidx, 2)));
		}
		// Fill in the face list
		for (int fidx = 0; fidx < numFaces; ++fidx) {
			addFace(mFaceMat(fidx, 0), mFaceMat(fidx, 1), mFaceMat(fidx, 2));
		}

		std::vector< HEdge* > hedgeList;
		for (int i = 0; i < mBHEdgeList.size(); ++i) {
			if (mBHEdgeList[i]->start()) {
				hedgeList.push_back(mBHEdgeList[i]);
			}
			// TODO
		}
		mBHEdgeList = hedgeList;

		for (int i = 0; i < mVertexList.size(); ++i) {
			mVertexList[i]->adjHEdges.clear();
			mVertexList[i]->setIndex(i);
			mVertexList[i]->setFlag(0);
		}
	} else {
		std::cout << __FUNCTION__ << ": mesh file loading failed!\n";
	}
	return iglFlag;
}

static void _setPrevNext(HEdge* e1, HEdge* e2) {
	e1->setNext(e2);
	e2->setPrev(e1);
}

static void _setTwin(HEdge* e1, HEdge* e2) {
	e1->setTwin(e2);
	e2->setTwin(e1);
}

static void _setFace(Face* f, HEdge* e) {
	f->setHalfEdge(e);
	e->setFace(f);
}

void Mesh::addFace(int v1, int v2, int v3) {
	Face* face = new Face();

	HEdge* hedge[3];
	HEdge* bhedge[3]; // Boundary half-edges
	Vertex* vert[3];

	for (int i = 0; i < 3; ++i) {
		hedge[i] = new HEdge();
		bhedge[i] = new HEdge(true);
	}
	vert[0] = mVertexList[v1];
	vert[1] = mVertexList[v2];
	vert[2] = mVertexList[v3];

	// Connect prev-next pointers
	for (int i = 0; i < 3; ++i) {
		_setPrevNext(hedge[i], hedge[(i + 1) % 3]);
		_setPrevNext(bhedge[i], bhedge[(i + 1) % 3]);
	}

	// Connect twin pointers
	_setTwin(hedge[0], bhedge[0]);
	_setTwin(hedge[1], bhedge[2]);
	_setTwin(hedge[2], bhedge[1]);

	// Connect start pointers for bhedge
	bhedge[0]->setStart(vert[1]);
	bhedge[1]->setStart(vert[0]);
	bhedge[2]->setStart(vert[2]);
	for (int i = 0; i < 3; ++i) {
		hedge[i]->setStart(vert[i]);
	}

	// Connect start pointers
	// Connect face-hedge pointers
	for (int i = 0; i < 3; ++i) {
		vert[i]->setHalfEdge(hedge[i]);
		vert[i]->adjHEdges.push_back(hedge[i]);
		_setFace(face, hedge[i]);
	}
	vert[0]->adjHEdges.push_back(bhedge[1]);
	vert[1]->adjHEdges.push_back(bhedge[0]);
	vert[2]->adjHEdges.push_back(bhedge[2]);

	// Merge boundary if needed
	for (int i = 0; i < 3; ++i) {
		Vertex* start = bhedge[i]->start();
		Vertex* end = bhedge[i]->end();

		for (int j = 0; j < end->adjHEdges.size(); ++j) {
			HEdge* curr = end->adjHEdges[j];
			if (curr->isBoundary() && curr->end() == start) {
				_setPrevNext(bhedge[i]->prev(), curr->next());
				_setPrevNext(curr->prev(), bhedge[i]->next());
				_setTwin(bhedge[i]->twin(), curr->twin());
				bhedge[i]->setStart(nullptr); // Mark as unused
				curr->setStart(nullptr); // Mark as unused
				break;
			}
		}
	}

	// Finally add hedges and faces to list
	for (int i = 0; i < 3; ++i) {
		mHEdgeList.push_back(hedge[i]);
		mBHEdgeList.push_back(bhedge[i]);
	}
	mFaceList.push_back(face);
}

Eigen::Vector3f Mesh::initBboxMin() const {
	return (mVertexMat.colwise().minCoeff()).transpose();
}

Eigen::Vector3f Mesh::initBboxMax() const {
	return (mVertexMat.colwise().maxCoeff()).transpose();
}

void Mesh::groupingVertexFlags() {
	// Init to 255
	for (Vertex* vert : mVertexList) {
		if (vert->flag() != 0) {
			vert->setFlag(255);
		}
	}
	// Group handles
	int id = 0;
	std::vector< Vertex* > tmpList;
	for (Vertex* vert : mVertexList) {
		if (vert->flag() == 255) {
			++id;
			vert->setFlag(id);

			// Do search
			tmpList.push_back(vert);
			while (!tmpList.empty()) {
				Vertex* v = tmpList.back();
				tmpList.pop_back();

				OneRingVertex orv = OneRingVertex(v);
				while (Vertex* v2 = orv.nextVertex()) {
					if (v2->flag() == 255) {
						v2->setFlag(id);
						tmpList.push_back(v2);
					}
				}
			}
		}
	}
}

void Mesh::clear() {
	for (int i = 0; i < mHEdgeList.size(); ++i) {
		delete mHEdgeList[i];
	}
	for (int i = 0; i < mBHEdgeList.size(); ++i) {
		delete mBHEdgeList[i];
	}
	for (int i = 0; i < mVertexList.size(); ++i) {
		delete mVertexList[i];
	}
	for (int i = 0; i < mFaceList.size(); ++i) {
		delete mFaceList[i];
	}

	mHEdgeList.clear();
	mBHEdgeList.clear();
	mVertexList.clear();
	mFaceList.clear();
}

void Mesh::setlambda(float val) {
	LAMBDA_ = val;
}

float Mesh::readlambda() const {
	return LAMBDA_;
}

void Mesh::setiternum(int val) {
	ITER_NUM_ = val;
};

int Mesh::readiternum() const {
	return ITER_NUM_;
};


std::vector< int > Mesh::collectMeshStats() {
	int V = 0; // # of vertices
	int E = 0; // # of half-edges
	int F = 0; // # of faces
	int B = 0; // # of boundary loops
	int C = 0; // # of connected components
	int G = 0; // # of genus

	/*====== Programming Assignment 0 ======*/

	/**********************************************/
	/*          Insert your code here.            */
	/**********************************************/
	/*
	/* Collect mesh information as listed above.
	/**********************************************/

	/*====== Programming Assignment 0 ======*/

	std::vector< int > stats;
	stats.push_back(V);
	stats.push_back(E);
	stats.push_back(F);
	stats.push_back(B);
	stats.push_back(C);
	stats.push_back(G);
	return stats;
}

int Mesh::countBoundaryLoops() {
	int count = 0;

	/*====== Programming Assignment 0 ======*/

	/**********************************************/
	/*          Insert your code here.            */
	/**********************************************/
	/*
	/* Helper function for Mesh::collectMeshStats()
	/**********************************************/

	/*====== Programming Assignment 0 ======*/

	return count;
}

int Mesh::countConnectedComponents() {
	int count = 0;

	/*====== Programming Assignment 0 ======*/

	/**********************************************/
	/*          Insert your code here.            */
	/**********************************************/
	/*
	/* Helper function for Mesh::collectMeshStats()
	/* Count the number of connected components of
	/* the mesh. (Hint: use a stack)
	/**********************************************/

	/*====== Programming Assignment 0 ======*/

	return count;
}

void Mesh::computeVertexNormals() {
	/*====== Programming Assignment 0 ======*/

	/**********************************************/
	/*          Insert your code here.            */
	/**********************************************/
	/*
	/* Compute per-vertex normal using neighboring
	/* facet information. (Hint: remember using a 
	/* weighting scheme. Plus, do you notice any
	/* disadvantages of your weighting scheme?)
	/**********************************************/

	/*====== Programming Assignment 0 ======*/
	
	// Notify mesh shaders
	setVertexNormalDirty(true);
}

//HL-Code
typedef Eigen::Triplet<double, int> Triplet;
/// declares a column-major sparse matrix type of double
typedef Eigen::SparseMatrix<double> Sparse_mat;

static std::vector<std::vector<Triplet> > get_normalized_laplacian(const std::vector< Eigen::Vector3f >& vertices,
                         const std::vector< std::vector<int> >& edges )
{
    int nv = vertices.size();
    std::vector<std::vector<Triplet> > mat_elemts(nv);
    for(int i = 0; i < nv; ++i)
        mat_elemts[i].reserve(10);

    for(int i = 0; i < nv; ++i)
    {
        const Eigen::Vector3f c_pos = vertices[i];

        //get laplacian
        double sum = 0.;
        int nb_edges = edges[i].size();
        for(int e = 0; e < nb_edges; ++e)
        {
            int next_edge = (e + 1           ) % nb_edges;
            int prev_edge = (e + nb_edges - 1) % nb_edges;

            Eigen::Vector3f v1 = c_pos                 - vertices[edges[i][prev_edge]];
            Eigen::Vector3f v2 = vertices[edges[i][e]] - vertices[edges[i][prev_edge]];
            Eigen::Vector3f v3 = c_pos                 - vertices[edges[i][next_edge]];
            Eigen::Vector3f v4 = vertices[edges[i][e]] - vertices[edges[i][next_edge]];

            double cotan1 = (v1.dot(v2)) / (1e-6 + (v1.cross(v2)).norm() );
            double cotan2 = (v3.dot(v4)) / (1e-6 + (v3.cross(v4)).norm() );

            double w = (cotan1 + cotan2)*0.5;
            sum += w;
            mat_elemts[i].push_back( Triplet(i, edges[i][e], w) );
        }

        for( Triplet& t : mat_elemts[i] )
            t = Triplet( t.row(), t.col(), t.value() / sum);

        mat_elemts[i].push_back( Triplet(i, i, -1.0) );
    }
    return mat_elemts;
}


static
std::vector< std::vector<float> >
get_cotan_weights(const std::vector< Eigen::Vector3f >& vertices,
                  const std::vector< std::vector<int> >& edges)
{
    int nv = vertices.size();
    std::vector< std::vector<float> > weights(nv);

    for(int i = 0; i < nv; ++i)
    {
        const Eigen::Vector3f c_pos = vertices[i];
        double sum = 0.;
        int nb_edges = edges[i].size();
        weights[i].resize( nb_edges );
        for(int e = 0; e < nb_edges; ++e)
        {
            int next_edge = (e + 1           ) % nb_edges;
            int prev_edge = (e + nb_edges - 1) % nb_edges;

            Eigen::Vector3f v1 = c_pos                 - vertices[edges[i][prev_edge]];
            Eigen::Vector3f v2 = vertices[edges[i][e]] - vertices[edges[i][prev_edge]];
            Eigen::Vector3f v3 = c_pos                 - vertices[edges[i][next_edge]];
            Eigen::Vector3f v4 = vertices[edges[i][e]] - vertices[edges[i][next_edge]];

            double cotan1 = (v1.dot(v2)) / (1e-6 + (v1.cross(v2)).norm() );
            double cotan2 = (v3.dot(v4)) / (1e-6 + (v3.cross(v4)).norm() );

            double w = (cotan1 + cotan2)*0.5;
            weights[i][e] = w;
        }
    }
    return weights;
}


//HL-Code


void Mesh::explicitSmooth(bool cotangentWeights) {
	/*====== Programming Assignment 1 ======*/
	// The lambda value is predefined by variable LAMBDA_, please use it directly

	if (cotangentWeights) {
		/**********************************************/
		/*          Insert your code here.            */
		/**********************************************/
		/*
		/* Step 1: Implement the cotangent weighting 
		/* scheme for explicit mesh smoothing. 
		/*
		/* Hint:
		/* It is advised to double type to store the 
		/* weights to avoid numerical issues.
		/**********************************************/
		// const std::vector< Eigen::Vector3f >& in_vertices = vertices();
		const std::vector< Vertex* >& in_vertices = vertices();
		// const std::vector< std::vector<int> >& edges = edge();
		//有一种可能就是你得按照你自己的理解来实现这个代码，因为没有什么代码可以用诶
		const std::vector< HEdge* >& edges = this->edges();
		float alpha = readlambda();
		
		// Get the 
		int nb_vertices = in_vertices.size();
		std::vector< std::vector<float> > cotan_weights = get_cotan_weights(in_vertices, edges);
	
		std::vector< Eigen::Vector3f > buffer_vertices(nb_vertices);
		//std::vector< Eigen::Vector3f > source = in_vertices;
		const std::vector< Vertex* >& source = in_vertices;
		//how to transform a Vertex to a Vector3f here? or just use Vertex
		//here I decide to use directly the Vertex
	
		Eigen::Vector3f* src_vertices = source.data();
		Eigen::Vector3f* dst_vertices = buffer_vertices.data();

		for( int i = 0; i < nb_vertices; i++)
		{
			Eigen::Vector3f cog(0.f);
			float sum = 0.f;
			size_t nb_neighs = edges[i].size();
			for(size_t n = 0; n < nb_neighs; n++)
			{
				int neigh = edges[i][n];
				float w = cotan_weights[i][n];
				cog += src_vertices[neigh] * w;
				sum += w;
			}
			float t = alpha;
			dst_vertices[i] = (cog / sum) * t + src_vertices[i] * (1.f - t);

		}
		std::swap(dst_vertices, src_vertices);


		return (nb_iter%2 == 1) ? buffer_vertices : source;
		mVertexList = & buffer_vertices;

	} else {
		/**********************************************/
		/*          Insert your code here.            */
		/**********************************************/
		/*
		/* Step 2: Implement the uniform weighting 
		/* scheme for explicit mesh smoothing.
		/**********************************************/

		const std::vector< Eigen::Vector3f >& in_vertices = vertices();
		const std::vector< std::vector<int> >& edges = boundaryEdges();
		float alpha = readlambda();
	
		int nb_vertices = in_vertices.size();

		std::vector<Eigen::VectorXd> xyz;
		std::vector<Eigen::VectorXd> rhs;

		xyz.resize(3, Eigen::VectorXd::Zero(nb_vertices));
		rhs.resize(3, Eigen::VectorXd::Zero(nb_vertices));

		for(int i = 0; i < nb_vertices; ++i)
		{
			Eigen::Vector3f pos = in_vertices[i];
			xyz[0][i] = pos.x;
			xyz[1][i] = pos.y;
			xyz[2][i] = pos.z;
		}

		// Build laplacian normalized matrix
		std::vector<std::vector<Triplet> > mat_elemts = get_normalized_laplacian(in_vertices, edges);
		Eigen::SparseMatrix<double> L(nb_vertices, nb_vertices);
		std::vector<Triplet> triplets;
		triplets.reserve(nb_vertices * 10);
		for( const std::vector<Triplet>& row : mat_elemts)
			for( const Triplet& elt : row )
				triplets.push_back( elt );

		L.setFromTriplets(triplets.begin(), triplets.end());

		Eigen::SparseMatrix<double> I = Eigen::MatrixXd::Identity(nb_vertices, nb_vertices).sparseView();
		L = I + L*alpha;

		//L = L*L*L*L*L*L*L*L*L*L;
		rhs = xyz;
		for(int k = 0; k < 3; k++){
			xyz[k] = (L * xyz[k]);
		}

		std::vector< Eigen::Vector3f > out_verts(nb_vertices);
		for(int i = 0; i < nb_vertices; ++i){
			Eigen::Vector3f v;
			v.x = xyz[0][i];
			v.y = xyz[1][i];
			v.z = xyz[2][i];
			out_verts[i] = v;
		}
		//return out_verts;
		for (int i = 0; i < mVertexList.size(); ++i) {
			mVertexList[i]->setPosition(out_verts[i])
			mVertexList[i]->setIndex(i);
			mVertexList[i]->setFlag(0);
		}
	}

	/*====== Programming Assignment 1 ======*/

	computeVertexNormals();
	// Notify mesh shaders
	setVertexPosDirty(true);
}

void Mesh::implicitSmooth(bool cotangentWeights) {
	/*====== Programming Assignment 1 ======*/
	// The lambda value is predefined by variable LAMBDA_, please use it directly

	/* A sparse linear system Ax=b solver using the conjugate gradient method. */
	auto fnConjugateGradient = [](const Eigen::SparseMatrix< float >& A,
	                              const Eigen::VectorXf& b,
	                              int maxIterations,
	                              float errorTolerance,
	                              Eigen::VectorXf& x)
	{
		/**********************************************/
		/*          Insert your code here.            */
		/**********************************************/
		/*
		/* Params:
		/*  A: 
		/*  b: 
		/*  maxIterations:	Max number of iterations
		/*  errorTolerance: Error tolerance for the early stopping condition
		/*  x:				Stores the final solution, but should be initialized. 
		/**********************************************/
		/*
		/* Step 1: Implement the biconjugate gradient
		/* method.
		/* Hint: https://en.wikipedia.org/wiki/Biconjugate_gradient_method
		/**********************************************/
	};

	/* IMPORTANT:
	/* Please refer to the following link about the sparse matrix construction in Eigen. */
	/* http://eigen.tuxfamily.org/dox/group__TutorialSparse.html#title3 */

	if (cotangentWeights) {
		/**********************************************/
		/*          Insert your code here.            */
		/**********************************************/
		/*
		/* Step 2: Implement the cotangent weighting 
		/* scheme for implicit mesh smoothing. Use
		/* the above fnConjugateGradient for solving
		/* sparse linear systems.
		/*
		/* Hint:
		/* It is advised to double type to store the
		/* weights to avoid numerical issues.
		/**********************************************/


	} else {
		/**********************************************/
		/*          Insert your code here.            */
		/**********************************************/
		/*
		/* Step 3: Implement the uniform weighting 
		/* scheme for implicit mesh smoothing. Use
		/* the above fnConjugateGradient for solving
		/* sparse linear systems.
		/**********************************************/

	}

	/*====== Programming Assignment 1 ======*/

	computeVertexNormals();
	// Notify mesh shaders
	setVertexPosDirty(true);
}
