#ifndef __CONSTRAINT_H__
#define __CONSTRAINT_H__

#pragma once
#include "GraphColoring.h"

struct SpringParam {
	uint	*_ids;
	uint	*_colors;
	REAL	*_cs;
	REAL	*_lambdas;
	uint	_size;
};

class Constraint {
public:
	Dvector<uint>	d_ids;
	Dvector<REAL>	d_cs;
	Dvector<REAL>	d_lambdas;
public:
	vector<uint>	h_ids;
	vector<REAL>	h_cs;
public:
	Dvector<uint>	d_colors;
	uint			_colorSize;
	uint			_numSprings;
public:
	Constraint() {
		_colorSize = _numSprings = 0u;
	}
	~Constraint() {}
public:
	SpringParam param(void) {
		SpringParam p;
		p._ids = d_ids._list;
		p._cs = d_cs._list;
		p._lambdas = d_lambdas._list;
		p._colors = d_colors._list;
		p._size = _numSprings;
		return p;
	}
	void clear(void) {
		d_ids.clear();
		d_cs.clear();
		d_lambdas.clear();
		d_colors.clear();

		h_ids.clear();
		h_cs.clear();
		_colorSize = _numSprings = 0u;
	}
public:
	void update(vector<uint>& es, vector<REAL>& cs, vector<uint>& bes, vector<REAL>& bcs, uint numNodes);
public:
	void projectXPBD(Dvector<REAL>& ns, Dvector<REAL>& invMs);
};

#endif