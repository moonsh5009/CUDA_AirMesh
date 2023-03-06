#ifndef __OBJECT_H__
#define __OBJECT_H__

#pragma once
#include "AirMesh.h"

#define OBJ_BOUNDARY_TYPE		0u
#define OBJ_HAIR_TYPE			1u
#define OBJ_RIGID_TYPE			2u

inline void drawNode(const REAL2& pos, REAL thickness) {
	uint itr = 15;
	glPushMatrix();
	glTranslated(pos.x, pos.y, 0);
	glBegin(GL_POLYGON);
	for (uint i = 0; i < itr; i++) {
		double rad = (REAL)i * 6.283184 / (REAL)itr;
		glVertex2f(cos(rad) * thickness, sin(rad) * thickness);
	}
	glEnd();
	glPopMatrix();
}

class Object {
public:
	vector<uint>	_es;
	float			_color[3];
	uint			_type;
public:
	Object() {
		_color[0] = 0.f;
		_color[1] = 0.f;
		_color[2] = 0.f;
		_type = 0xffffffff;
	}
	virtual ~Object() {}
public:
	virtual void draw(vector<REAL>& ns) = 0;
	virtual void build(vector<uint>& es, vector<uint>& bes, Dvector<REAL>& ns) = 0;
public:
	void setColor(float r, float g, float b);
};
class Hair : public Object {
public:
	Hair() { _type = OBJ_HAIR_TYPE; }
	virtual ~Hair() {}
public:
	virtual void build(vector<uint>& es, vector<uint>& bes, Dvector<REAL>& ns);
	virtual void draw(vector<REAL>& ns);
};
class Rigid : public Object{
public:
	vector<uint>	_fs;
public:
	Rigid() { _type = OBJ_RIGID_TYPE; }
	virtual ~Rigid() {}
public:
	virtual void getHole(vector<REAL>& holes, vector<REAL>& ns);
	virtual void build(vector<uint>& es, vector<uint>& bes, Dvector<REAL>& ns);
	virtual void draw(vector<REAL>& ns);
};

#endif