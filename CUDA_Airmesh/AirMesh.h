#ifndef __AIR_MESH_H__
#define __AIR_MESH_H__

#pragma once
#include "GraphColoring.h"

#define AIR_CONST					2.3094010767585030580365951220078
#define SQRT3						1.7320508075688772935274463415059
#define AIR_COLLISION_EPSILON		1.0e-10

#define AIR_QUALITY_EPSILON		0.0

#define AIR_FLIP_SORT			0
#define AIR_FLIP_ITERATION		300u

#define REST_LENGTH				1.0

//#define AIR_FLIP_CPU

struct FlipElement {
	uint _id;
	REAL _quality;
};
struct FlipElement_CMP
{
	__host__ __device__
		bool operator()(const FlipElement& a, const FlipElement& b) {
		if (a._quality != b._quality)
			return a._quality < b._quality;
		return a._id < b._id;
	}
};

struct ContactElement {
	uint _id;
	REAL _Cair;
};
struct ContactElement_CMP
{
	__host__ __device__
		bool operator()(const ContactElement& a, const ContactElement& b) {
		if (a._Cair != b._Cair)
			return a._Cair < b._Cair;
		return a._id < b._id;
	}
};

class AirMesh {
public:
	Dvector<uint>	d_fs;
	Dvector<uint>	d_es;
	Dvector<uint>	d_nbEFs;
	Dvector<uint>	d_nbFEs;
public:
	vector<uint>	h_fs;
	vector<uint>	h_es;
	uint _testId;
	uint _testtimer;
public:
	bool _collision;
	bool _flip;
public:
	AirMesh() {
		_testId = 0u;
		_testtimer = 0u;
		_collision = true;
		_flip = true;
	}
	virtual ~AirMesh() {}
public:
	inline void clear(void) {
		_testId = 0u;
		_testtimer = 0u;
		d_fs.clear();
		d_es.clear();
		d_nbEFs.clear();
		d_nbFEs.clear();
		h_fs.clear();
		h_es.clear();
	}
public:
	static void CDT(const Dvector<uint>& segs, Dvector<uint>& fs, Dvector<uint>& es, Dvector<REAL>& ns, vector<REAL>& holes);
	static void getRigidSpring(Dvector<uint>& fs, Dvector<uint>& es, Dvector<REAL>& ns, Dvector<uint>& nbEFs, Dvector<uint>& nbFEs);
	static void reorderElements(
		const Dvector<uint>& fs, Dvector<uint>& es,
		Dvector<REAL>& ns, Dvector<uint>& nbEFs, Dvector<uint>& nbFEs);
public:
	void init(Dvector<uint>& es, Dvector<REAL>& ns, Dvector<uint>& nodePhases, vector<REAL>& holes);
public:
	void getFlipElements(
		Dvector<REAL>& ns, DPrefixArray<uint>& nbNs, DPrefixArray<uint>& nbNs2, Dvector<uint> nodePhases, REAL delta, Dvector<FlipElement>& elems, uint* numElems);
	void getContactElements(
		Dvector<REAL>& ns, Dvector<REAL>& cs, DPrefixArray<uint>& nbNs, REAL delta, Dvector<ContactElement>& elems, uint* numElems);
	void flip(Dvector<REAL>& ns, DPrefixArray<uint>& nbNs, DPrefixArray<uint>& nbNs2, Dvector<uint> nodePhases, REAL delta);
	void collision(
		Dvector<REAL>& ns, Dvector<REAL>& n0s, Dvector<REAL>& invMs,
		Dvector<REAL>& cs, DPrefixArray<uint>& nbNs, REAL delta, REAL friction, uint iteration);
	void calcPredictPosition(
		Dvector<REAL>& pns, uint iteration);
	void resolveCollision(
		Dvector<REAL>& ns, Dvector<REAL>& n0s, Dvector<REAL>& invMs,
		Dvector<REAL>& cs, DPrefixArray<uint>& nbNs, DPrefixArray<uint>& nbNs2, Dvector<uint>& nodePhases, REAL delta,
		REAL friction, uint iteration);
public:
	void testEdge(vector<REAL>& ns, REAL2 mousePos = make_REAL2(0, 0));
	void draw(vector<REAL>& ns);
};

#endif