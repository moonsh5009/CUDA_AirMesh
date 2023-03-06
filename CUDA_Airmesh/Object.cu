#include "Object.cuh"
#include "triangle.h"

void Object::setColor(float r, float g, float b) {
	_color[0] = r;
	_color[1] = g;
	_color[2] = b;
}

void Hair::build(vector<uint>& es, vector<uint>& bes, Dvector<REAL>& ns) {
	_es = es;
	bes.clear();
}
void Hair::draw(vector<REAL>& ns) {
	glColor3fv(_color);
	glBegin(GL_LINES);
	for (uint i = 0; i < _es.size(); i += 2u) {
		uint ino0 = _es[i + 0u] << 1u;
		uint ino1 = _es[i + 1u] << 1u;
		REAL2 p0 = make_REAL2(ns[ino0 + 0u], ns[ino0 + 1u]);
		REAL2 p1 = make_REAL2(ns[ino1 + 0u], ns[ino1 + 1u]);
		glVertex2f(p0.x, p0.y);
		glVertex2f(p1.x, p1.y);
	}
	glEnd();
}

void Rigid::getHole(vector<REAL>& holes, vector<REAL>& ns) {
	REAL2 p0 = make_REAL2(ns[_fs[0] * 2], ns[_fs[0] * 2 + 1]);
	REAL2 p1 = make_REAL2(ns[_fs[1] * 2], ns[_fs[1] * 2 + 1]);
	REAL2 p2 = make_REAL2(ns[_fs[2] * 2], ns[_fs[2] * 2 + 1]);
	REAL2 cen = (p0 + p1 + p2) / 3.0;
	holes.push_back(cen.x);
	holes.push_back(cen.y);
}
void Rigid::build(vector<uint>& es, vector<uint>& bes, Dvector<REAL>& ns) {
	_es = es;

	Dvector<uint> d_fs;
	Dvector<uint> d_es;
	Dvector<uint> d_nbEFs;
	Dvector<uint> d_nbFEs;
	d_es = es;

	AirMesh::getRigidSpring(d_fs, d_es, ns, d_nbEFs, d_nbFEs);
	d_fs.copyToHost(_fs);
	d_es.copyToHost(es);

	vector<uint> h_nbEFs;
	d_nbEFs.copyToHost(h_nbEFs);
	uint numNbEFs = h_nbEFs.size() >> 1u;

	uint i, j;
	for (i = 0u; i < numNbEFs; i++) {
		uint if0 = h_nbEFs[(i << 1u) + 0u];
		uint if1 = h_nbEFs[(i << 1u) + 1u];
		if (if1 != 0xffffffff) {
			uint i0 = es[(i << 1u) + 0u];
			uint i1 = es[(i << 1u) + 1u];
			uint b0, b1;
			for (j = 0u; j < 3u; j++) {
				b0 = _fs[if0 * 3u + j];
				if (b0 != i0 && b0 != i1)
					break;
			}
			for (j = 0u; j < 3u; j++) {
				b1 = _fs[if1 * 3u + j];
				if (b1 != i0 && b1 != i1)
					break;
			}
			if (b0 < b1) {
				bes.push_back(b0);
				bes.push_back(b1);
			}
			else {
				bes.push_back(b1);
				bes.push_back(b0);
			}
		}
	}
}
void Rigid::draw(vector<REAL>& ns) {
	glColor3fv(_color);
	glBegin(GL_LINES);
	for (uint i = 0; i < _es.size(); i += 2u) {
		uint ino0 = _es[i + 0u] << 1u;
		uint ino1 = _es[i + 1u] << 1u;
		REAL2 p0 = make_REAL2(ns[ino0 + 0u], ns[ino0 + 1u]);
		REAL2 p1 = make_REAL2(ns[ino1 + 0u], ns[ino1 + 1u]);
		glVertex2f(p0.x, p0.y);
		glVertex2f(p1.x, p1.y);
	}
	glEnd();

	glColor3f(_color[0] * 0.5, _color[1] * 0.5, _color[2] * 0.5);
	for (uint i = 0; i < _fs.size(); i += 3u) {
		uint ino0 = _fs[i + 0u] << 1u;
		uint ino1 = _fs[i + 1u] << 1u;
		uint ino2 = _fs[i + 2u] << 1u;
		REAL2 p0 = make_REAL2(ns[ino0 + 0u], ns[ino0 + 1u]);
		REAL2 p1 = make_REAL2(ns[ino1 + 0u], ns[ino1 + 1u]);
		REAL2 p2 = make_REAL2(ns[ino2 + 0u], ns[ino2 + 1u]);
		glBegin(GL_TRIANGLES);
		glVertex2f(p0.x, p0.y);
		glVertex2f(p1.x, p1.y);
		glVertex2f(p2.x, p2.y);
		glEnd();
	}
}