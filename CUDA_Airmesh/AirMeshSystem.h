#ifndef __AIR_MESH_SYSTEM_H__
#define __AIR_MESH_SYSTEM_H__

#pragma once
#include "Object.h"
#include "Constraint.h"

#define GRAVITY		0.98

#define DRAW_MESH		1u
#define DRAW_AIRMESH	2u

class AirMeshSystem {
public:
	Dvector<uint>		d_es;
	Dvector<REAL>		d_cs;
	Dvector<uint>		d_bes;
	Dvector<REAL>		d_bcs;
	Dvector<REAL>		d_ns;
	Dvector<REAL>		d_n0s;
	Dvector<REAL>		d_vs;
	Dvector<REAL>		d_invMs;
	DPrefixArray<uint>	d_nbNs;
	DPrefixArray<uint>	d_nbNs2;
	Dvector<uint>		d_nodePhases;
public:
	vector<uint>		h_es;
	vector<REAL>		h_cs;
	vector<uint>		h_bes;
	vector<REAL>		h_bcs;
	vector<REAL>		h_bks;
	vector<REAL>		h_ns;
	vector<REAL>		h_vs;
	vector<REAL>		h_invMs;
	PrefixArray<uint>	h_nbNs;
	vector<uint>		h_nodePhases;
public:
	vector<Object*>		_objs;
	vector<uint>		h_hols;
public:
	Constraint		*_constraints;
public:
	AirMesh			*_air;
public:
	vector<uint>	_spawnEdges;
	vector<REAL>	_spawnPoints;
	vector<REAL>	_spawnInvMs;
	REAL			_spawnLength;
	uint			_spawnType;
	float			_spawnColor[3];
	bool			_spawning;
public:
	uint			_moveNodeId;
	REAL2			_movePos;
public:
	uint			_numFaces;
	uint			_numEdges;
	uint			_numNodes;
	uint			_numBounEdges;
	uint			_numBounNodes;
	uint			_drawType;
	AABB			_boun;
	REAL			_dt;
	REAL			_invdt;
	REAL			_mass;
	REAL			_invMass;
	REAL			_thickness;
	uint			_maxIter;
public:
	AirMeshSystem() {}
	~AirMeshSystem() { free(); }
public:
	void init(REAL width, REAL height);
	void updateNums(void);
	void initBoundary(void);
	void initNbNs(void);
	void buildAirMesh(void);
	void updateConstraintsInf(REAL stretchMaterial, REAL bendMaterial);
public:
	void freeObjs(void);
	void free(void);
	void reset(void);
public:
	void changeSpawnType(void);
	void spawn(REAL x, REAL y);
	void spawnMove(REAL x, REAL y);
public:
	void clickNode(REAL2 point);
	void updateMovePos(REAL2 point);
	void moveNode(void);
	void lockNode(void);
public:
	void longHair(void);
public:
	void updateVelocities(void);
	void update(void);
	void simulation(void);
public:
	void draw(void);
	void drawBoundary(void);
	void drawSpawn(void);
	void drawMove(void);
	void drawObjects(void);
	void drawSprings(void);
};


#endif