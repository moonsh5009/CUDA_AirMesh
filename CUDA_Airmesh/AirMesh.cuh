#include "AirMesh.h"
#include "DeviceManager.cuh"

__global__ void getNbNsIds_kernel2(uint* es, uint* ids, uint numEdges) {
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numEdges)
		return;

	uint ino = id << 1u;
	uint ino0 = es[ino + 0u];
	uint ino1 = es[ino + 1u];
	atomicAdd(ids + ino0 + 1u, 1u);
	atomicAdd(ids + ino1 + 1u, 1u);
}
__global__ void getNbNs_kernel2(uint* es, uint* nbNs, uint* ids, uint numEdges) {
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numEdges)
		return;

	uint ino = id << 1u;
	uint ino0 = es[ino + 0u];
	uint ino1 = es[ino + 1u];
	ino = atomicAdd(ids + ino0, 1u);
	nbNs[ino] = ino1;
	ino = atomicAdd(ids + ino1, 1u);
	nbNs[ino] = ino0;
}

__global__ void reorderTriangles_kernel(uint* fs, REAL* ns, uint numFaces) {
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numFaces)
		return;

	uint ino = id * 3u;
	uint ino0 = fs[ino + 0u];
	uint ino1 = fs[ino + 1u];
	uint ino2 = fs[ino + 2u];
	REAL2 p0, p1, p2;
	p0.x = ns[(ino0 << 1u) + 0u]; p0.y = ns[(ino0 << 1u) + 1u];
	p1.x = ns[(ino1 << 1u) + 0u]; p1.y = ns[(ino1 << 1u) + 1u];
	p2.x = ns[(ino2 << 1u) + 0u]; p2.y = ns[(ino2 << 1u) + 1u];
}
__global__ void getNbFsIds_kernel(uint* fs, uint* ids, uint numFaces) {
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numFaces)
		return;

	uint ino = id * 3u;
	uint ino0 = fs[ino + 0u];
	uint ino1 = fs[ino + 1u];
	uint ino2 = fs[ino + 2u];
	atomicAdd(ids + ino0 + 1u, 1u);
	atomicAdd(ids + ino1 + 1u, 1u);
	atomicAdd(ids + ino2 + 1u, 1u);
}
__global__ void getNbFs_kernel(uint* fs, uint* nbFs, uint* ids, uint numFaces) {
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numFaces)
		return;

	uint ino = id * 3u;
	uint ino0 = fs[ino + 0u];
	uint ino1 = fs[ino + 1u];
	uint ino2 = fs[ino + 2u];
	ino = atomicAdd(ids + ino0, 1u);
	nbFs[ino] = id;
	ino = atomicAdd(ids + ino1, 1u);
	nbFs[ino] = id;
	ino = atomicAdd(ids + ino2, 1u);
	nbFs[ino] = id;
}
__global__ void getEdgesSize_kernel(uint* fs, uint* nbFs, uint* ids, uint* iedge, uint numFaces) {
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numFaces)
		return;

	uint ino = id * 3u, jno;
	uint inos[3];
	inos[0] = fs[ino + 0u];
	inos[1] = fs[ino + 1u];
	inos[2] = fs[ino + 2u];
	uint istart, iend, jstart, jend;
	uint inbf, jnbf;
	uint tris[2];
	jstart = ids[inos[0]];
	jend = ids[inos[0] + 1u];
	uint n, i, j, ie;
	for (i = 0u; i < 3u; i++) {
		j = (i + 1u) % 3u;
		istart = jstart;
		iend = jend;
		jstart = ids[inos[j]];
		jend = ids[inos[j] + 1u];

		tris[0] = 0xffffffff;
		tris[1] = 0xffffffff;
		n = 0u;
		for (ino = istart; ino < iend; ino++) {
			inbf = nbFs[ino];
			for (jno = jstart; jno < jend; jno++) {
				jnbf = nbFs[jno];
				if (inbf == jnbf) {
					tris[n++] = inbf;
					break;
				}
			}
			if (n == 2u)
				break;
		}
		if (tris[0] > tris[1]) {
			n = tris[0];
			tris[0] = tris[1];
			tris[1] = n;
		}
		if (tris[0] == id)
			ie = atomicAdd(iedge, 1u);
	}
}
__global__ void buildEdgeNbFs_kernel(uint* fs, uint* es, uint* nbFs, uint* ids, uint* nbEFs, uint* iedge, uint numFaces) {
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numFaces)
		return;

	uint ino = id * 3u, jno;
	uint inos[3];
	inos[0] = fs[ino + 0u];
	inos[1] = fs[ino + 1u];
	inos[2] = fs[ino + 2u];
	uint istart, iend, jstart, jend;
	uint inbf, jnbf;
	uint tris[2];
	jstart = ids[inos[0]];
	jend = ids[inos[0] + 1u];
	uint n, i, j, ie, ie0, ie1;
	for (i = 0u; i < 3u; i++) {
		j = (i + 1u) % 3u;
		istart = jstart;
		iend = jend;
		jstart = ids[inos[j]];
		jend = ids[inos[j] + 1u];

		tris[0] = 0xffffffff;
		tris[1] = 0xffffffff;
		n = 0u;
		for (ino = istart; ino < iend; ino++) {
			inbf = nbFs[ino];
			for (jno = jstart; jno < jend; jno++) {
				jnbf = nbFs[jno];
				if (inbf == jnbf) {
					tris[n++] = inbf;
					break;
				}
			}
			if (n == 2u)
				break;
		}
		if (tris[0] > tris[1]) {
			n = tris[0];
			tris[0] = tris[1];
			tris[1] = n;
		}
		if (tris[0] == id) {
			ie = atomicAdd(iedge, 1u);
			ie <<= 1u;
			if (inos[i] < inos[j]) {
				es[ie + 0u] = inos[i];
				es[ie + 1u] = inos[j];
			}
			else {
				es[ie + 0u] = inos[j];
				es[ie + 1u] = inos[i];
			}
			nbEFs[ie + 0u] = tris[0];
			nbEFs[ie + 1u] = tris[1];
		}
	}
}
__global__ void buildNbFEs_kernel(uint* nbEFs, uint* nbFEs, uint* ids, uint numEdges) {
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numEdges)
		return;

	uint ind = id << 1u;
	uint ino, offset, i;
	for (i = 0u; i < 2u; i++) {
		ino = nbEFs[ind + i];
		if (ino != 0xffffffff) {
			offset = atomicAdd(ids + ino, 1u);
			ino = ino * 3u + offset;
			nbFEs[ino] = id;
		}
	}
}
__global__ void reorderNbFEs_kernel(uint* fs, uint* es, uint* nbFEs, uint numFaces) {
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numFaces)
		return;

	uint ino = id * 3u;
	uint inos[3], enos[3], seqs[3];
	uint e0, e1, tmp, i;
	inos[0] = fs[ino + 0u];
	inos[1] = fs[ino + 1u];
	inos[2] = fs[ino + 2u];
	enos[0] = nbFEs[ino + 0u];
	enos[1] = nbFEs[ino + 1u];
	enos[2] = nbFEs[ino + 2u];
	for (i = 0u; i < 3u; i++) {
		e0 = es[(enos[i] << 1u) + 0u];
		e1 = es[(enos[i] << 1u) + 1u];
		if (e0 == inos[0])
			e0 = 0u;
		else if (e0 == inos[1])
			e0 = 1u;
		else e0 = 2u;
		if (e1 == inos[0])
			e1 = 0u;
		else if (e1 == inos[1])
			e1 = 1u;
		else e1 = 2u;
		if (e0 > e1) {
			tmp = e0;
			e0 = e1;
			e1 = tmp;
		}

		if (e0 == 0u && e1 == 1u)
			seqs[i] = 0u;
		else if (e0 == 1u && e1 == 2u)
			seqs[i] = 1u;
		else seqs[i] = 2u;
	}
	for (i = 0u; i < 3u; i++)
		nbFEs[ino + seqs[i]] = enos[i];
}
__device__ bool isEdge_device(uint* nbNs, uint* inbNs, uint i0, uint i1) {
	//if (i0 == i1)
	//	return;
	uint ino, iend, nv;
	if (i0 > i1) {
		ino = i0;
		i0 = i1;
		i1 = ino;
	}

	nv = 0xffffffff;
	iend = inbNs[i0 + 1u];
	for (ino = inbNs[i0]; ino < iend; ino++) {
		nv = nbNs[ino];
		if (nv >= i1)
			break;
	}
	return nv == i1;
}
__device__ bool getRestLength_device(REAL* cs, uint* nbNs, uint* inbNs, uint i0, uint i1, REAL& restLength) {
	//if (i0 == i1)
	//	return;
	bool result = false;
	uint ino, iend, nv;
	if (i0 > i1) {
		ino = i0;
		i0 = i1;
		i1 = ino;
	}

	nv = 0xffffffff;
	iend = inbNs[i0 + 1u];
	for (ino = inbNs[i0]; ino < iend; ino++) {
		nv = nbNs[ino];
		if (nv >= i1)
			break;
	}

	if (nv == i1) {
		result = true;
		restLength = cs[ino << 1u];
	}
	return result;
}
__global__ void sortEdgeIndex(uint* es, uint numEdges) {
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numEdges)
		return;

	uint ino = id << 1u;
	uint ino0 = es[ino + 0u];
	uint ino1 = es[ino + 1u];
	if (ino0 > ino1) {
		es[ino + 0u] = ino1;
		es[ino + 1u] = ino0;
	}
}
//__global__ void buildNbs_kernel(uint* fs, uint* es, uint* nbEFs, uint* nbFEs, uint* ids, uint numFaces) {
//	uint id = threadIdx.x + blockDim.x * blockIdx.x;
//	if (id >= numEdges)
//		return;
//
//	uint ind = id << 1u;
//	uint ino, offset, i;
//	for (i = 0u; i < 2u; i++) {
//		ino = nbEFs[ind + i];
//		if (ino != 0xffffffff) {
//			offset = atomicAdd(ids + ino, 1u);
//			ino = ino * 3u + offset;
//			nbFEs[ino] = id;
//		}
//	}
//}

__device__ void getCairEV(
	uint* inos, REAL2* ps, REAL* ls, bool* isEdges,
	REAL delta, REAL& Cair)
{
	uint seq[3], i;
	seq[0] = 0u;
	seq[1] = 1u;
	seq[2] = 2u;
	if (ls[seq[0]] < ls[seq[1]]) {
		i = seq[0];
		seq[0] = seq[1];
		seq[1] = i;
	}
	if (ls[seq[0]] < ls[seq[2]]) {
		i = seq[0];
		seq[0] = seq[2];
		seq[2] = i;
	}
	if (ls[seq[1]] < ls[seq[2]]) {
		i = seq[1];
		seq[1] = seq[2];
		seq[2] = i;
	}

	Cair = Cross(ps[1] - ps[0], ps[2] - ps[0]);
	for (i = 0u; i < 3u; i++) {
		if (isEdges[seq[i]])
			break;
	}
	REAL offset = 0.0;
	offset = SQRT3 * delta * 0.5;
	if (i < 3u) {
		uint i0 = seq[i];
		uint i1 = (i0 + 1u) % 3u;
		uint i2 = (i1 + 1u) % 3u;
		REAL l = (ls[i0] + delta * 0.5) * ls[i0];
		if (Dot(ps[i1] - ps[i0], ps[i2] - ps[i0]) > l ||
			Dot(ps[i0] - ps[i1], ps[i2] - ps[i1]) > l)
			offset = ls[i0] * 0.96;
		else offset = ls[i0];
	}
	Cair -= offset * delta;
}
__device__ bool resolveCollision(
	uint* inos, uint iedge, REAL2* ps, REAL* ws, REAL* ls, REAL Cair,
	REAL delta)
{
	bool result = false;

	if (Cair < -AIR_COLLISION_EPSILON) {
		REAL2 dirs[3];
		REAL lenDirCs[3];
		REAL s = 0.0;

		{
			//if (iedge == 0xffffffff) {
			//	dirs[0].x = ps[1].y - ps[2].y; dirs[0].y = ps[2].x - ps[1].x;
			//	dirs[1].x = ps[2].y - ps[0].y; dirs[1].y = ps[0].x - ps[2].x;
			//	dirs[2].x = ps[0].y - ps[1].y; dirs[2].y = ps[1].x - ps[0].x;
			//	lenDirCs[0] = ls[1] * ls[1];
			//	lenDirCs[1] = ls[2] * ls[2];
			//	lenDirCs[2] = ls[0] * ls[0];
			//}
			//else {
			//	uint i0 = iedge;
			//	uint i1 = (i0 + 1u) % 3u;
			//	uint i2 = (i1 + 1u) % 3u;

			//	REAL l2 = ls[i0] * ls[i0];
			//	REAL w = Dot(ps[i1] - ps[i0], ps[i2] - ps[i0]) / (l2 + FLT_EPSILON);
			//	if (w < 0.0)		w = 0.0;
			//	else if (w > 1.0)	w = 1.0;

			//	dirs[i2].x = ps[i0].y - ps[i1].y; dirs[i2].y = ps[i1].x - ps[i0].x;
			//	dirs[i0].x = -dirs[i2].x * (1.0 - w); dirs[i0].y = -dirs[i2].y * (1.0 - w);
			//	dirs[i1].x = -dirs[i2].x * w; dirs[i1].y = -dirs[i2].y * w;

			//	lenDirCs[i0] = l2 * (1.0 - w) * (1.0 - w);
			//	lenDirCs[i1] = l2 * w * w;
			//	lenDirCs[i2] = l2;
			//	/*uint i0 = iedge;
			//	uint i1 = (i0 + 1u) % 3u;
			//	uint i2 = (i1 + 1u) % 3u;

			//	REAL2 norm;
			//	norm.x = ps[i0].y - ps[i1].y; norm.y = ps[i1].x - ps[i0].x;

			//	REAL l2 = ls[i0] * ls[i0];
			//	REAL w = Dot(ps[i1] - ps[i0], ps[i2] - ps[i0]) / (l2 + FLT_EPSILON);

			//	if (w >= 0.0 && w <= 1.0)
			//		dirs[i2] = norm;
			//	else {
			//		if (w <= 0.0) {
			//			w = 0.0;
			//			dirs[i2] = ps[i2] - ps[i0];
			//		}
			//		else  {
			//			w = 1.0;
			//			dirs[i2] = ps[i2] - ps[i1];
			//		}
			//		if (Dot(norm, dirs[i2]) > 0.0) {
			//			dirs[i2].x = -dirs[i2].x;
			//			dirs[i2].y = -dirs[i2].y;
			//		}
			//		dirs[i2] *= Cair / LengthSquared(dirs[i2]);
			//	}

			//	dirs[i0].x = -dirs[i2].x * (1.0 - w); dirs[i0].y = -dirs[i2].y * (1.0 - w);
			//	dirs[i1].x = -dirs[i2].x * w; dirs[i1].y = -dirs[i2].y * w;

			//	lenDirCs[i0] = l2 * (1.0 - w) * (1.0 - w);
			//	lenDirCs[i1] = l2 * w * w;
			//	lenDirCs[i2] = l2;*/
			//}
		}
		{
			dirs[0].x = ps[1].y - ps[2].y; dirs[0].y = ps[2].x - ps[1].x;
			dirs[1].x = ps[2].y - ps[0].y; dirs[1].y = ps[0].x - ps[2].x;
			dirs[2].x = ps[0].y - ps[1].y; dirs[2].y = ps[1].x - ps[0].x;
			lenDirCs[0] = ls[1] * ls[1];
			lenDirCs[1] = ls[2] * ls[2];
			lenDirCs[2] = ls[0] * ls[0];
		}

		s += ws[0] * lenDirCs[0];
		s += ws[1] * lenDirCs[1];
		s += ws[2] * lenDirCs[2];
		if (s > 1.0e-10) {
			s = -Cair / s;
			ps[0] += s * ws[0] * dirs[0];
			ps[1] += s * ws[1] * dirs[1];
			ps[2] += s * ws[2] * dirs[2];
			result = true;
		}
	}
	{
		/*for (uint i0 = 0u; i0 < 3u; i0++) {
			uint i1 = (i0 + 1u) % 3u;
			REAL2 dir = (ps[i1] - ps[i0]);
			REAL len = Length(dir);
			REAL C = len - delta;
			if (C < -AIR_COLLISION_EPSILON) {
				REAL s = len * (ws[i0] + ws[i1]);
				if (s > 1.0e-20) {
					s = -C / s;
					ps[i0] += s * ws[i0] * dir;
					ps[i1] -= s * ws[i1] * dir;
					result = true;
				}
			}
		}*/
	}
	/*{
		for (uint i0 = 0u; i0 < 3u; i0++) {
			uint i1 = (i0 + 1u) % 3u;
			if (ws[i0] || ws[i1]) {
				REAL2 dir = ps[i1] - ps[i0];
				REAL length = Length(dir);
				if (length < delta && length) {
					REAL C = delta - length;
					REAL s = C / (length * (ws[i0] + ws[i1]));
					ps[i0] -= s * ws[i0] * dir;
					ps[i1] += s * ws[i1] * dir;
					result = true;
				}
			}
		}
	}*/
	return result;
}
__device__ bool resolveFriction(
	uint* inos, uint iedge,
	REAL2* ps, REAL2* p0s, REAL* ws, REAL* ls, REAL* l0s,
	REAL delta, REAL friction)
{
	bool result = false;
	if (l0s[iedge]) {
		uint i0 = iedge;
		uint i1 = (i0 + 1u) % 3u;
		uint i2 = (i1 + 1u) % 3u;
		REAL invL = 1.0 / l0s[i0];
		REAL2 norm;
		//norm.x = ps[i0].y - ps[i1].y; norm.y = ps[i1].x - ps[i0].x;
		norm.x = p0s[i0].y - p0s[i1].y; norm.y = p0s[i1].x - p0s[i0].x;
		norm *= invL;

		//REAL w = Dot(ps[i1] - ps[i0], ps[i2] - ps[i0]) * invL;
		REAL w = Dot(p0s[i1] - p0s[i0], p0s[i2] - p0s[i0]);
		if (w >= -delta * 0.5 || w <= l0s[i0] + delta * 0.5) {
			w *= invL;
			if (w < 0.0)		w = 0.0;
			else if (w > 1.0)	w = 1.0;

			REAL2 p01 = p0s[i0] + (p0s[i1] - p0s[i0]) * w;
			REAL pene = Dot(p0s[i2] - p01, norm);
			if (pene < delta + AIR_COLLISION_EPSILON) {
				REAL2 q01 = ps[i0] + (ps[i1] - ps[i0]) * w;
				REAL2 relV = ps[i2] - q01 - (p0s[i2] - p01);
				pene = Dot(relV, norm);

				REAL2 dir = relV - pene * norm;
				REAL lrelT = Length(dir);
				if (lrelT > 0.0) {
					REAL C = max(1.0 - Length(relV) * friction / lrelT, 0.0);

					REAL lambdas[3];
					REAL w0 = ws[i0] * (1.0 - w);
					REAL w1 = ws[i1] * w;
					lambdas[0] = w0 * (1.0 - w);
					lambdas[1] = w1 * w;
					lambdas[2] = ws[i2];

					REAL s = 0.0;
					s += lambdas[0];
					s += lambdas[1];
					s += lambdas[2];
					if (s > 1.0e-20) {
						s = C / s;
						ps[i0] += s * w0 * dir;
						ps[i1] += s * w1 * dir;
						ps[i2] -= s * ws[i2] * dir;
						result = true;
					}
				}
			}
		}
	}
	return result;
}

__global__ void getContactElements_kernel(
	uint* fs, REAL* ns, uint* nbNs, uint* inbNs,
	REAL delta, uint numFaces, ContactElement* elems, uint* num)
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numFaces)
		return;

	uint ind = id * 3u;
	uint inos[3];
	REAL2 ps[3];
	REAL ls[3];
	uint ino;
	for (ino = 0u; ino < 3u; ino++) {
		inos[ino] = fs[ind + ino];
		ps[ino].x = ns[(inos[ino] << 1u) + 0u];
		ps[ino].y = ns[(inos[ino] << 1u) + 1u];
	}
	ls[0] = Length(ps[1] - ps[0]);
	ls[1] = Length(ps[2] - ps[1]);
	ls[2] = Length(ps[0] - ps[2]);

	uint iend, i0, i1, i2;
	bool isEdges[3];
	isEdges[0] = isEdge_device(nbNs, inbNs, inos[0], inos[1]);
	isEdges[1] = isEdge_device(nbNs, inbNs, inos[1], inos[2]);
	isEdges[2] = isEdge_device(nbNs, inbNs, inos[2], inos[0]);

	REAL Cair = Cross(ps[1] - ps[0], ps[2] - ps[0]);

	if (!(isEdges[0] && isEdges[1] && isEdges[2])) {
		uint iedge = 0xffffffff;
		if (isEdges[0] && (!isEdges[1] || ls[0] >= ls[1]) && (!isEdges[2] || ls[0] >= ls[2]))
			iedge = 0u;
		else if (isEdges[1] && (!isEdges[2] || ls[1] >= ls[2]))
			iedge = 1u;
		else if (isEdges[2])
			iedge = 2u;

		REAL offset = 0.0;
		if (iedge != 0xffffffff && offset < min(ls[iedge], REST_LENGTH))
			offset = min(ls[iedge], REST_LENGTH);
		Cair -= offset * delta;
	}

	if (Cair < -AIR_COLLISION_EPSILON) {
		ino = atomicAdd(num, 1u);
		ContactElement elem;
		elem._id = id;
		elem._Cair = Cair;
		elems[ino] = elem;
	}
}
__global__ void getContactElementBuffer_kernel(
	uint* fs, ContactElement* elems, uint numElems, uint* buffer)
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numElems)
		return;

	ContactElement elem = elems[id];
	uint ino = elem._id * 3u;
	uint inos[3];
	inos[0] = fs[ino + 0u];
	inos[1] = fs[ino + 1u];
	inos[2] = fs[ino + 2u];
	ino = id * 3u;
	buffer[ino + 0u] = inos[0];
	buffer[ino + 1u] = inos[1];
	buffer[ino + 2u] = inos[2];
}
__global__ void Collision_kernel(
	uint* fs, REAL* ns, REAL* invMs, REAL* cs, uint* nbNs, uint* inbNs,
	REAL delta, ContactElement* elems, uint numElems,
	uint* seqs, uint currSeq)
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numElems)
		return;

	uint seq = seqs[id];
	if (seq == currSeq) {
		ContactElement elem = elems[id];
		uint ind = elem._id * 3u;
		uint inos[3];
		REAL2 ps[3];
		REAL ws[3], ls[3];
		uint ino;
		for (ino = 0u; ino < 3u; ino++) {
			inos[ino] = fs[ind + ino];
			ps[ino].x = ns[(inos[ino] << 1u) + 0u];
			ps[ino].y = ns[(inos[ino] << 1u) + 1u];
			ws[ino] = invMs[inos[ino]];
		}

		bool isEdges[3];
		isEdges[0] = getRestLength_device(cs, nbNs, inbNs, inos[0], inos[1], ls[0]);
		isEdges[1] = getRestLength_device(cs, nbNs, inbNs, inos[1], inos[2], ls[1]);
		isEdges[2] = getRestLength_device(cs, nbNs, inbNs, inos[2], inos[0], ls[2]);

		if (!isEdges[0]) ls[0] = Length(ps[1] - ps[0]);
		if (!isEdges[1]) ls[1] = Length(ps[2] - ps[1]);
		if (!isEdges[2]) ls[2] = Length(ps[0] - ps[2]);

		REAL Cair = Cross(ps[1] - ps[0], ps[2] - ps[0]);

		uint iedge = 0xffffffff;
		uint seqs[3];
		seqs[0] = 0u; seqs[1] = 1u; seqs[2] = 2u;
		if (ls[seqs[0]] < ls[seqs[1]]) {
			ino = seqs[0];
			seqs[0] = seqs[1];
			seqs[1] = ino;
		}
		if (ls[seqs[0]] < ls[seqs[2]]) {
			ino = seqs[0];
			seqs[0] = seqs[2];
			seqs[2] = ino;
		}
		if (ls[seqs[1]] < ls[seqs[2]]) {
			ino = seqs[1];
			seqs[1] = seqs[2];
			seqs[2] = ino;
		}
		if (!(isEdges[0] && isEdges[1] && isEdges[2])) {
			for (ino = 0u; ino < 3u; ino++) {
				if (isEdges[seqs[ino]]) {
					iedge = seqs[ino];
					break;
				}
			}
		}
		REAL offset = 0.0;
		if (iedge != 0xffffffff)
			offset = ls[iedge];
		//	offset = min(ls[iedge], REST_LENGTH);
		//else if (ls[seqs[2]] < delta && ls[seqs[2]])
		//	offset = max(fabs(Cair / ls[seqs[2]]), SQRT3 * 0.5 * delta);
		Cair -= offset * delta;

		bool applied = resolveCollision(inos, iedge, ps, ws, ls, Cair, delta);
		if (applied) {
			for (ino = 0u; ino < 3u; ino++) {
				ns[(inos[ino] << 1u) + 0u] = ps[ino].x;
				ns[(inos[ino] << 1u) + 1u] = ps[ino].y;
			}
		}
	}
}
__global__ void Collision_kernel(
	uint* fs, REAL* ns, REAL* invMs, REAL* cs, uint* nbNs, uint* inbNs, REAL delta,
	uint numFaces, uint* colors, uint currColor, uint* isApplied)
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numFaces)
		return;

	uint color = colors[id];
	if (color == currColor) {
		uint ind = id * 3u;
		uint inos[3];
		REAL2 ps[3];
		REAL ws[3], ls[3];
		uint ino;
		for (ino = 0u; ino < 3u; ino++) {
			inos[ino] = fs[ind + ino];
			ps[ino].x = ns[(inos[ino] << 1u) + 0u];
			ps[ino].y = ns[(inos[ino] << 1u) + 1u];
			ws[ino] = invMs[inos[ino]];
		}

		bool isEdges[3];
		isEdges[0] = getRestLength_device(cs, nbNs, inbNs, inos[0], inos[1], ls[0]);
		isEdges[1] = getRestLength_device(cs, nbNs, inbNs, inos[1], inos[2], ls[1]);
		isEdges[2] = getRestLength_device(cs, nbNs, inbNs, inos[2], inos[0], ls[2]);

		if (!isEdges[0]) ls[0] = Length(ps[1] - ps[0]);
		if (!isEdges[1]) ls[1] = Length(ps[2] - ps[1]);
		if (!isEdges[2]) ls[2] = Length(ps[0] - ps[2]);

		uint iedge = 0xffffffff;
		REAL Cair = Cross(ps[1] - ps[0], ps[2] - ps[0]);
		if (!isEdges[0] || !isEdges[1] || !isEdges[2]) {
			uint seqs[3];

			seqs[0] = 0u; seqs[1] = 1u; seqs[2] = 2u;
			if (ls[seqs[0]] < ls[seqs[1]]) {
				ino = seqs[0];
				seqs[0] = seqs[1];
				seqs[1] = ino;
			}
			if (ls[seqs[0]] < ls[seqs[2]]) {
				ino = seqs[0];
				seqs[0] = seqs[2];
				seqs[2] = ino;
			}
			if (ls[seqs[1]] < ls[seqs[2]]) {
				ino = seqs[1];
				seqs[1] = seqs[2];
				seqs[2] = ino;
			}
			for (ino = 0u; ino < 3u; ino++) {
				if (isEdges[seqs[ino]]) {
					iedge = seqs[ino];
					break;
				}
			}

			REAL offset = SQRT3 * 0.5 * delta;
			if (iedge != 0xffffffff)
				offset = ls[iedge];
			else if (ls[seqs[2]] < delta && Cair > 0.0)
				offset = max(Cair / (ls[seqs[2]] + FLT_EPSILON), offset);
			Cair -= offset * delta;
		}

		bool applied = resolveCollision(inos, iedge, ps, ws, ls, Cair, delta);
		if (applied) {
			for (ino = 0u; ino < 3u; ino++) {
				ns[(inos[ino] << 1u) + 0u] = ps[ino].x;
				ns[(inos[ino] << 1u) + 1u] = ps[ino].y;
			}
			*isApplied = true;
			//atomicAdd(isApplied , 1u);
		}
	}
}
__global__ void CollisionJacobi_kernel(
	uint* fs, REAL* ns, REAL* invMs, REAL* cs, uint* nbNs, uint* inbNs, REAL delta,
	uint numFaces, REAL* ds, uint* invds, uint* isApplied)
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numFaces)
		return;

	uint ind = id * 3u;
	uint inos[3];
	REAL2 ps[3], qs[3];
	REAL ws[3], ls[3];
	uint ino;
	for (ino = 0u; ino < 3u; ino++) {
		inos[ino] = fs[ind + ino];
		ps[ino].x = ns[(inos[ino] << 1u) + 0u];
		ps[ino].y = ns[(inos[ino] << 1u) + 1u];
		ws[ino] = invMs[inos[ino]];
		qs[ino] = ps[ino];
	}

	bool isEdges[3];
	isEdges[0] = getRestLength_device(cs, nbNs, inbNs, inos[0], inos[1], ls[0]);
	isEdges[1] = getRestLength_device(cs, nbNs, inbNs, inos[1], inos[2], ls[1]);
	isEdges[2] = getRestLength_device(cs, nbNs, inbNs, inos[2], inos[0], ls[2]);

	if (!isEdges[0]) ls[0] = Length(ps[1] - ps[0]);
	if (!isEdges[1]) ls[1] = Length(ps[2] - ps[1]);
	if (!isEdges[2]) ls[2] = Length(ps[0] - ps[2]);

	uint iedge = 0xffffffff;
	REAL Cair = Cross(ps[1] - ps[0], ps[2] - ps[0]);
	if (!isEdges[0] || !isEdges[1] || !isEdges[2]) {
		uint seqs[3];

		seqs[0] = 0u; seqs[1] = 1u; seqs[2] = 2u;
		if (ls[seqs[0]] < ls[seqs[1]]) {
			ino = seqs[0];
			seqs[0] = seqs[1];
			seqs[1] = ino;
		}
		if (ls[seqs[0]] < ls[seqs[2]]) {
			ino = seqs[0];
			seqs[0] = seqs[2];
			seqs[2] = ino;
		}
		if (ls[seqs[1]] < ls[seqs[2]]) {
			ino = seqs[1];
			seqs[1] = seqs[2];
			seqs[2] = ino;
		}
		for (ino = 0u; ino < 3u; ino++) {
			if (isEdges[seqs[ino]]) {
				iedge = seqs[ino];
				break;
			}
		}

		REAL offset = SQRT3 * 0.5 * delta;
		if (iedge != 0xffffffff)
			offset = ls[iedge];
		else if (ls[seqs[2]] < delta && Cair > 0.0)
			offset = max(Cair / (ls[seqs[2]] + FLT_EPSILON), offset);
		Cair -= offset * delta;
	}

	bool applied = resolveCollision(inos, iedge, ps, ws, ls, Cair, delta);
	if (applied) {
		for (ino = 0u; ino < 3u; ino++) {
			if (ws[ino]) {
				atomicAdd_REAL(ds + (inos[ino] << 1u) + 0u, ps[ino].x - qs[ino].x);
				atomicAdd_REAL(ds + (inos[ino] << 1u) + 1u, ps[ino].y - qs[ino].y);
				atomicAdd(invds + inos[ino], 1u);
			}
		}
		*isApplied = 1.0;
	}
}
__global__ void FrictionJacobi_kernel(
	uint* fs, REAL* ns, REAL* n0s, REAL* invMs, REAL* cs, uint* nbNs, uint* inbNs, REAL delta, REAL friction,
	uint numFaces, REAL* ds, uint* invds, uint* isApplied)
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numFaces)
		return;

	uint ind = id * 3u;
	uint inos[3];
	REAL2 ps[3], qs[3];
	REAL ws[3], ls[3];
	uint ino;
	for (ino = 0u; ino < 3u; ino++) {
		inos[ino] = fs[ind + ino];
		ps[ino].x = ns[(inos[ino] << 1u) + 0u];
		ps[ino].y = ns[(inos[ino] << 1u) + 1u];
		ws[ino] = invMs[inos[ino]];
		qs[ino] = ps[ino];
	}
	bool isEdges[3];
	isEdges[0] = getRestLength_device(cs, nbNs, inbNs, inos[0], inos[1], ls[0]);
	isEdges[1] = getRestLength_device(cs, nbNs, inbNs, inos[1], inos[2], ls[1]);
	isEdges[2] = getRestLength_device(cs, nbNs, inbNs, inos[2], inos[0], ls[2]);

	if (!isEdges[0]) ls[0] = Length(ps[1] - ps[0]);
	if (!isEdges[1]) ls[1] = Length(ps[2] - ps[1]);
	if (!isEdges[2]) ls[2] = Length(ps[0] - ps[2]);

	uint iedge = 0xffffffff;
	if (isEdges[0] && (!isEdges[1] || ls[0] >= ls[1]) && (!isEdges[2] || ls[0] >= ls[2]))
		iedge = 0u;
	else if (isEdges[1] && (!isEdges[2] || ls[1] >= ls[2]))
		iedge = 1u;
	else if (isEdges[2])
		iedge = 2u;

	if (iedge != 0xffffffff) {
		if (Cross(ps[1] - ps[0], ps[2] - ps[0]) - REST_LENGTH * delta < AIR_COLLISION_EPSILON) {
			REAL2 p0s[3];
			REAL l0s[3];
			for (ino = 0u; ino < 3u; ino++) {
				p0s[ino].x = n0s[(inos[ino] << 1u) + 0u];
				p0s[ino].y = n0s[(inos[ino] << 1u) + 1u];
			}
			l0s[0] = Length(p0s[1] - p0s[0]);
			l0s[1] = Length(p0s[2] - p0s[1]);
			l0s[2] = Length(p0s[0] - p0s[2]);
			bool applied = resolveFriction(inos, iedge, ps, p0s, ws, ls, l0s, delta, friction);
			if (applied) {
				for (ino = 0u; ino < 3u; ino++) {
					if (ws[ino]) {
						atomicAdd_REAL(ds + (inos[ino] << 1u) + 0u, ps[ino].x - qs[ino].x);
						atomicAdd_REAL(ds + (inos[ino] << 1u) + 1u, ps[ino].y - qs[ino].y);
						atomicAdd(invds + inos[ino], 1u);
					}
				}
			}
		}
	}
	*isApplied = 1.0;
}
__global__ void ApplyJacobi_kernel(
	REAL* ns, REAL* ds, uint* invds, REAL omega, uint numNodes)
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numNodes)
		return;

	uint num = invds[id];
	if (num) {
		uint ino = id << 1u;
		REAL invd = min(1.0 + omega, (REAL)num);
		invd /= (REAL)num;

		REAL2 p, d;
		p.x = ns[ino + 0u];
		p.y = ns[ino + 1u];
		d.x = ds[ino + 0u];
		d.y = ds[ino + 1u];
		p += d * invd;

		ns[ino + 0u] = p.x;
		ns[ino + 1u] = p.y;
	}
}

__device__ bool checkNeiFlip_device(
	uint* fs, uint* es, uint* nbEFs, uint* nbFEs,
	uint fstart0, uint fstart1, uint pivot, 
	uint ino0, uint ino1, uint vno)
{
	bool result = false;
	uint n0, n1, ne, nf, nv;
	uint ino, jno;
	uint iface = fstart0;
	uint if_3 = iface * 3u;
	uint curr = ino0;
	do {
		for (ino = 0u; ino < 3u; ino++) {
			ne = nbFEs[if_3 + ino];
			ne <<= 1u;
			n0 = es[ne + 0u];
			n1 = es[ne + 1u];
			if ((n0 == pivot || n0 == curr) && (n1 == pivot || n1 == curr)) {
				nf = nbEFs[ne + 1u];
				if (nf != 0xffffffff) {
					if (nf == iface)
						nf = nbEFs[ne + 0u];
				}
				iface = nf;
				if (nf == 0xffffffff)
					break;

				nf *= 3u;
				if_3 = nf;
				for (jno = 0u; jno < 3u; jno++) {
					nv = fs[nf + jno];
					if (nv != n0 && nv != n1)
						break;
				}
				curr = nv;
				result = (nv == vno);
				break;
			}
		}
	} while (curr != ino1 && iface != 0xffffffff && !result);
	if (iface == 0xffffffff) {
		iface = fstart1;
		if_3 = iface * 3u;
		curr = ino1;
		do {
			for (ino = 0u; ino < 3u; ino++) {
				ne = nbFEs[if_3 + ino];
				ne <<= 1u;
				n0 = es[ne + 0u];
				n1 = es[ne + 1u];
				if ((n0 == pivot || n0 == curr) && (n1 == pivot || n1 == curr)) {
					nf = nbEFs[ne + 1u];
					if (nf != 0xffffffff) {
						if (nf == iface)
							nf = nbEFs[ne + 0u];
					}
					iface = nf;
					if (nf == 0xffffffff)
						break;

					nf *= 3u;
					if_3 = nf;
					for (jno = 0u; jno < 3u; jno++) {
						nv = fs[nf + jno];
						if (nv != n0 && nv != n1)
							break;
					}
					result = (nv == vno);
					curr = nv;
					break;
				}
			}
		} while (iface != 0xffffffff && !result);
	}
	return result;
}
__device__ void getNeiNode_device(
	uint* fs, uint* es, uint* nbEFs, uint* nbFEs,
	uint iface, uint pivot, uint* ibuffer)
{
	uint if3, nf, ne, nv, n0, n1;
	uint ino, jno, n = 0u;
	ibuffer[0] = ibuffer[1] = 0xffffffff;
	if3 = iface * 3u;
	for (ino = 0u; ino < 3u; ino++) {
		ne = nbFEs[if3 + ino];
		ne <<= 1u;
		n0 = es[ne + 0u];
		n1 = es[ne + 1u];
		if (n0 == pivot || n1 == pivot) {
			nf = nbEFs[ne + 1u];
			if (nf != 0xffffffff) {
				if (nf == iface)
					nf = nbEFs[ne + 0u];

				nf *= 3u;
				for (jno = 0u; jno < 3u; jno++) {
					nv = fs[nf + jno];
					if (nv != n0 && nv != n1) {
						ibuffer[n++] = nv;
						break;
					}
				}
			}
		}
	}
}
__device__ bool checkNeiFlip_device(
	uint* fs, uint* es, uint* nbEFs, uint* nbFEs, 
	REAL* ns, uint* nodePhases, REAL2& p0, REAL2& p1, REAL2 p2, REAL2 p3,
	uint fstart0, uint fstart1, uint pivot, uint ino0, uint ino1)
{
	bool result = false;
	uint n0, n1, ne, nf, nv;
	uint ino, jno;
	uint iface = fstart0;
	uint if_3 = iface * 3u;
	uint curr = ino0;
	uint phase = nodePhases[pivot], currPhase;
	REAL2 p;
	do {
		for (ino = 0u; ino < 3u; ino++) {
			ne = nbFEs[if_3 + ino];
			ne <<= 1u;
			n0 = es[ne + 0u];
			n1 = es[ne + 1u];
			if ((n0 == pivot || n0 == curr) && (n1 == pivot || n1 == curr)) {
				nf = nbEFs[ne + 1u];
				if (nf != 0xffffffff) {
					if (nf == iface)
						nf = nbEFs[ne + 0u];
				}
				iface = nf;
				if (nf == 0xffffffff)
					break;

				nf *= 3u;
				if_3 = nf;
				for (jno = 0u; jno < 3u; jno++) {
					nv = fs[nf + jno];
					if (nv != n0 && nv != n1)
						break;
				}
				curr = nv;
				//if (curr != ino1) {
				//	currPhase = nodePhases[nv];
				//	//if (currPhase == phase) {
				//		p.x = ns[(nv << 1u) + 0u];
				//		p.y = ns[(nv << 1u) + 1u];
				//		//result = (Cross(p - p0, p1 - p0) > 0.0 && Cross(p2 - p0, p - p0) > 0.0);
				//		result = 
				//			(Cross(p - p0, p1 - p0) > 1.0e-10 && Cross(p3 - p0, p - p0) > 1.0e-10) ||
				//			(Cross(p - p0, p3 - p0) > 1.0e-10 && Cross(p2 - p0, p - p0) > 1.0e-10) ||
				//			(Cross(p - p0, p1 - p0) > 1.0e-10 && Cross(p2 - p0, p - p0) > 1.0e-10);
				//	//}
				//}
				REAL2 ps[3];
				for (jno = 0u; jno < 3u; jno++) {
					ps[jno].x = ns[(fs[nf + jno] << 1u) + 0u];
					ps[jno].y = ns[(fs[nf + jno] << 1u) + 1u];
				}
				result = (Cross(ps[1] - ps[0], ps[2] - ps[0]) < 1.0e-10);
				break;
			}
		}
	} while (curr != ino1 && iface != 0xffffffff && !result);
	if (iface == 0xffffffff) {
		iface = fstart1;
		if_3 = iface * 3u;
		curr = ino1;
		do {
			for (ino = 0u; ino < 3u; ino++) {
				ne = nbFEs[if_3 + ino];
				ne <<= 1u;
				n0 = es[ne + 0u];
				n1 = es[ne + 1u];
				if ((n0 == pivot || n0 == curr) && (n1 == pivot || n1 == curr)) {
					nf = nbEFs[ne + 1u];
					if (nf != 0xffffffff) {
						if (nf == iface)
							nf = nbEFs[ne + 0u];
					}
					iface = nf;
					if (nf == 0xffffffff)
						break;

					nf *= 3u;
					if_3 = nf;
					for (jno = 0u; jno < 3u; jno++) {
						nv = fs[nf + jno];
						if (nv != n0 && nv != n1)
							break;
					}
					curr = nv;
					//if (curr != ino0) {
					//	currPhase = nodePhases[nv];
					//	//if (currPhase == phase) {
					//		p.x = ns[(nv << 1u) + 0u];
					//		p.y = ns[(nv << 1u) + 1u];
					//		//result = (Cross(p - p0, p1 - p0) > 0.0 && Cross(p2 - p0, p - p0) > 0.0);
					//		result =
					//			(Cross(p - p0, p1 - p0) > 1.0e-10 && Cross(p3 - p0, p - p0) > 1.0e-10) ||
					//			(Cross(p - p0, p3 - p0) > 1.0e-10 && Cross(p2 - p0, p - p0) > 1.0e-10) ||
					//			(Cross(p - p0, p1 - p0) > 1.0e-10 && Cross(p2 - p0, p - p0) > 1.0e-10);
					//	//}
					//}
					REAL2 ps[3];
					for (jno = 0u; jno < 3u; jno++) {
						ps[jno].x = ns[(fs[nf + jno] << 1u) + 0u];
						ps[jno].y = ns[(fs[nf + jno] << 1u) + 1u];
					}
					result = (Cross(ps[1] - ps[0], ps[2] - ps[0]) < 1.0e-10);
					break;
				}
			}
		} while (iface != 0xffffffff && !result);
	}
	return result;
}
__device__ bool checkNeiFlip_device(
	uint* fs, uint* es, uint* nbEFs, uint* nbFEs,
	REAL* ns, uint* nbNs, uint* inbNs, REAL2& p0, REAL2& p1, REAL2 p2, REAL2 p3,
	uint fstart0, uint fstart1, uint pivot, uint ino0, uint ino1)
{
	bool result = false;
	uint n0, n1, ne, nf, nv;
	uint ino, jno;
	uint iface = fstart0;
	uint if_3 = iface * 3u;
	uint curr = ino0;
	REAL2 p;
	do {
		for (ino = 0u; ino < 3u; ino++) {
			ne = nbFEs[if_3 + ino];
			ne <<= 1u;
			n0 = es[ne + 0u];
			n1 = es[ne + 1u];
			if ((n0 == pivot || n0 == curr) && (n1 == pivot || n1 == curr)) {
				nf = nbEFs[ne + 1u];
				if (nf != 0xffffffff) {
					if (nf == iface)
						nf = nbEFs[ne + 0u];
				}
				iface = nf;
				if (nf == 0xffffffff)
					break;

				nf *= 3u;
				if_3 = nf;
				for (jno = 0u; jno < 3u; jno++) {
					nv = fs[nf + jno];
					if (nv != n0 && nv != n1)
						break;
				}
				curr = nv;
				if (curr != ino1 && isEdge_device(nbNs, inbNs, pivot, curr)) {
					p.x = ns[(nv << 1u) + 0u];
					p.y = ns[(nv << 1u) + 1u];
					result =
						(Cross(p - p0, p1 - p0) > -1.0e-10 && Cross(p2 - p0, p - p0) > -1.0e-10) ||
						(Cross(p0 - p1, p - p1) > -1.0e-10 && Cross(p - p1, p3 - p1) > -1.0e-10);
				}
				break;
			}
		}
	} while (curr != ino1 && iface != 0xffffffff && !result);
	if (iface == 0xffffffff) {
		iface = fstart1;
		if_3 = iface * 3u;
		curr = ino1;
		do {
			for (ino = 0u; ino < 3u; ino++) {
				ne = nbFEs[if_3 + ino];
				ne <<= 1u;
				n0 = es[ne + 0u];
				n1 = es[ne + 1u];
				if ((n0 == pivot || n0 == curr) && (n1 == pivot || n1 == curr)) {
					nf = nbEFs[ne + 1u];
					if (nf != 0xffffffff) {
						if (nf == iface)
							nf = nbEFs[ne + 0u];
					}
					iface = nf;
					if (nf == 0xffffffff)
						break;

					nf *= 3u;
					if_3 = nf;
					for (jno = 0u; jno < 3u; jno++) {
						nv = fs[nf + jno];
						if (nv != n0 && nv != n1)
							break;
					}
					curr = nv;
					if (curr != ino0 && isEdge_device(nbNs, inbNs, pivot, curr)) {
						p.x = ns[(nv << 1u) + 0u];
						p.y = ns[(nv << 1u) + 1u];
						result =
							(Cross(p - p0, p1 - p0) > -1.0e-10 && Cross(p2 - p0, p - p0) > -1.0e-10) ||
							(Cross(p3 - p1, p - p1) > -1.0e-10 && Cross(p - p1, p0 - p1) > -1.0e-10);
					}
					break;
				}
			}
		} while (iface != 0xffffffff && !result);
	}
	return result;
}


__device__ bool checkRevFlip_device(
	uint* fs, uint* es, uint* nbEFs, uint* nbFEs,
	uint fstart0, uint fstart1, uint pivot,
	uint ino0, uint ino1, uint dir)
{
	bool result = false;
	uint n0, n1, ne, nf, nv;
	uint ino, jno, i0, i1;
	uint iface = fstart0;
	uint if_3 = iface * 3u;
	uint curr = ino0;
	do {
		for (ino = 0u; ino < 3u; ino++) {
			ne = nbFEs[if_3 + ino];
			ne <<= 1u;
			n0 = es[ne + 0u];
			n1 = es[ne + 1u];
			if ((n0 == pivot || n0 == curr) && (n1 == pivot || n1 == curr)) {
				nf = nbEFs[ne + 1u];
				if (nf != 0xffffffff) {
					if (nf == iface)
						nf = nbEFs[ne + 0u];
				}
				iface = nf;
				if (nf == 0xffffffff)
					break;

				nf *= 3u;
				if_3 = nf;
				for (jno = 0u; jno < 3u; jno++) {
					nv = fs[nf + jno];
					if (nv == pivot)
						i0 = jno;
					else if (nv == curr)
						i1 = jno;
					else
						n0 = nv;
				}
				curr = n0;
				i0 += 2u - i1;
				result |= (i0 == dir || i0 == dir + 3u);
				break;
			}
		}
	} while (curr != ino1 && iface != 0xffffffff && !result);
	if (iface == 0xffffffff) {
		iface = fstart1;
		if_3 = iface * 3u;
		curr = ino1;
		do {
			for (ino = 0u; ino < 3u; ino++) {
				ne = nbFEs[if_3 + ino];
				ne <<= 1u;
				n0 = es[ne + 0u];
				n1 = es[ne + 1u];
				if ((n0 == pivot || n0 == curr) && (n1 == pivot || n1 == curr)) {
					nf = nbEFs[ne + 1u];
					if (nf != 0xffffffff) {
						if (nf == iface)
							nf = nbEFs[ne + 0u];
					}
					iface = nf;
					if (nf == 0xffffffff)
						break;

					nf *= 3u;
					if_3 = nf;
					for (jno = 0u; jno < 3u; jno++) {
						nv = fs[nf + jno];
						if (nv == pivot)
							i1 = jno;
						else if (nv == curr)
							i0 = jno;
						else
							n0 = nv;
					}
					curr = n0;
					i0 += 2u - i1;
					result |= (i0 == dir || i0 == dir + 3u);
					break;
				}
			}
		} while (iface != 0xffffffff && !result);
	}
	return result;
}

__device__ bool isTriangleOn(REAL2 v0, REAL2 v1, REAL2 v2, REAL2 p) {
	bool result = true;
	REAL2 v20 = v0 - v2;
	REAL2 v21 = v1 - v2;
	REAL t0 = Dot(v20, v20);
	REAL t1 = Dot(v21, v21);
	REAL t2 = Dot(v20, v21);
	REAL t3 = Dot(v20, p - v2);
	REAL t4 = Dot(v21, p - v2);
	REAL det = t0 * t1 - t2 * t2;
	if (fabs(det) <= 1.0e-20) 
		result = false;
	else {
		REAL invdet = 1.0 / det;
		REAL w0 = (+t1 * t3 - t2 * t4) * invdet, w1, w2;
		if (w0 <= 0.0 || w0 >= 1.0)
			result = false;
		else {
			w1 = (-t2 * t3 + t0 * t4) * invdet;
			if (w1 <= 0.0 || w1 >= 1.0)
				result = false;
			else {
				w2 = 1 - w0 - w1;
				if (w2 <= 0.0 || w2 >= 1.0)
					result = false;
			}
		}
	}
	return result;
}
__device__ bool checkFlip_device(
	uint* fs, uint* es, REAL* ns,
	uint* nbEFs, uint* nbFEs, uint* nbNs, uint* inbNs, uint* nodePhases,
	uint id, uint* inos, uint if0_3, uint if1_3)
{
	bool result = false;
	uint iface0 = if0_3 / 3u;
	uint iface1 = if1_3 / 3u;

	result = checkNeiFlip_device(fs, es, nbEFs, nbFEs, iface0, iface0, inos[2], inos[0], inos[1], inos[3]);
	if (!result) {
		/*uint ino, n;
		REAL2 ps[4], p;
		for (ino = 0u; ino < 4u; ino++) {
			ps[ino].x = ns[(inos[ino] << 1u) + 0u];
			ps[ino].y = ns[(inos[ino] << 1u) + 1u];
		}
		uint iend = inbNs[inos[0] + 1u];
		for (ino = inbNs[inos[0]]; ino < iend && !result; ino++) {
			n = nbNs[ino];
			if (n != inos[2] && n != inos[3]) {
				p.x = ns[(n << 1u) + 0u];
				p.y = ns[(n << 1u) + 1u];

				result = isTriangleOn(ps[0], ps[2], ps[3], p) || isTriangleOn(ps[1], ps[2], ps[3], p) || isTriangleOn(ps[2], ps[0], ps[1], p) || isTriangleOn(ps[3], ps[0], ps[1], p);
			}
		}
		iend = inbNs[inos[1] + 1u];
		for (ino = inbNs[inos[1]]; ino < iend && !result; ino++) {
			n = nbNs[ino];
			if (n != inos[2] && n != inos[3]) {
				p.x = ns[(n << 1u) + 0u];
				p.y = ns[(n << 1u) + 1u];

				result = isTriangleOn(ps[0], ps[2], ps[3], p) || isTriangleOn(ps[1], ps[2], ps[3], p) || isTriangleOn(ps[2], ps[0], ps[1], p) || isTriangleOn(ps[3], ps[0], ps[1], p);
			}
		}
		iend = inbNs[inos[2] + 1u];
		for (ino = inbNs[inos[2]]; ino < iend && !result; ino++) {
			n = nbNs[ino];
			if (n != inos[0] && n != inos[1]) {
				p.x = ns[(n << 1u) + 0u];
				p.y = ns[(n << 1u) + 1u];

				result = isTriangleOn(ps[0], ps[2], ps[3], p) || isTriangleOn(ps[1], ps[2], ps[3], p) || isTriangleOn(ps[2], ps[0], ps[1], p) || isTriangleOn(ps[3], ps[0], ps[1], p);
			}
		}
		iend = inbNs[inos[3] + 1u];
		for (ino = inbNs[inos[3]]; ino < iend && !result; ino++) {
			n = nbNs[ino];
			if (n != inos[0] && n != inos[1]) {
				p.x = ns[(n << 1u) + 0u];
				p.y = ns[(n << 1u) + 1u];

				result = isTriangleOn(ps[0], ps[2], ps[3], p) || isTriangleOn(ps[1], ps[2], ps[3], p) || isTriangleOn(ps[2], ps[0], ps[1], p) || isTriangleOn(ps[3], ps[0], ps[1], p);
			}
		}*/
	}
	return result;
}


__global__ void calcPredictPositionGC_kernel(
	uint* fs, REAL* ns, uint numFaces, uint* colors, uint currColor, uint* isApplied)
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numFaces)
		return;

	uint color = colors[id];
	if (color == currColor) {
		uint ind = id * 3u;
		uint inos[3];
		REAL2 ps[3];
		uint ino;
		for (ino = 0u; ino < 3u; ino++) {
			inos[ino] = fs[ind + ino];
			ps[ino].x = ns[(inos[ino] << 1u) + 0u];
			ps[ino].y = ns[(inos[ino] << 1u) + 1u];
		}

		uint iedge = 0xffffffff;
		REAL Cair = Cross(ps[1] - ps[0], ps[2] - ps[0]);
		if (Cair < -1.0e-10) {
			REAL2 gc = (ps[0] + ps[1] + ps[2]) * 0.333333333333333333334;
			for (ino = 0u; ino < 3u; ino++) {
				ns[(inos[ino] << 1u) + 0u] = gc.x;
				ns[(inos[ino] << 1u) + 1u] = gc.y;
			}
			*isApplied = 1.0;
		}
	}
}
__global__ void calcPredictPositionJacobi_kernel(
	uint* fs, REAL* ns, uint numFaces, REAL* ds, uint* invds, uint* isApplied)
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numFaces)
		return;

	uint ind = id * 3u;
	uint inos[3];
	REAL2 ps[3];
	uint ino;
	for (ino = 0u; ino < 3u; ino++) {
		inos[ino] = fs[ind + ino];
		ps[ino].x = ns[(inos[ino] << 1u) + 0u];
		ps[ino].y = ns[(inos[ino] << 1u) + 1u];
	}


	uint iedge = 0xffffffff;
	REAL Cair = Cross(ps[1] - ps[0], ps[2] - ps[0]);
	if (Cair < -1.0e-10) {
		REAL2 gc = (ps[0] + ps[1] + ps[2]) * 0.333333333333333333334;
		ps[0] = (gc - ps[0]) * 1.1;
		ps[1] = (gc - ps[1]) * 1.1;
		ps[2] = (gc - ps[2]) * 1.1;
		for (ino = 0u; ino < 3u; ino++) {
			atomicAdd_REAL(ds + (inos[ino] << 1u) + 0u, ps[ino].x);
			atomicAdd_REAL(ds + (inos[ino] << 1u) + 1u, ps[ino].y);
			atomicAdd(invds + inos[ino], 1u);
		}
		*isApplied = 1.0;
	}
}
__global__ void calcPredictPosition_kernel(REAL* ns, REAL* n0s, REAL* pns, REAL delta, uint numNodes) {
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numNodes)
		return;
	id <<= 1u;
	REAL2 p, p0;
	p.x = ns[id + 0u];
	p.y = ns[id + 1u];
	p0.x = n0s[id + 0u];
	p0.y = n0s[id + 1u];
	
	p0 -= p;
	REAL l = Length(p0);
	if (l > delta)
		p0 *= delta / l;
	p -= p0;
	//p += p - p0;

	pns[id + 0u] = p.x;
	pns[id + 1u] = p.y;
}

__device__ REAL compQualityDet_device(REAL l0, REAL l1, REAL l2) {
	REAL det;
	/*if (l1 < l2)
		det = l0 + l1 * 4.0 + l2;
	else
		det = l0 + l1 + l2 * 4.0;*/
	/*det = l0 + l1 + l2;
		- 1.1 * (l1 - l2) * (l1 - l2) / (l1 + l2);*/
	//det = l0 * 16.0 + l1 + l2;
	//det = l0 * 0.25 + l1 + l2 - (l1 - l2) * (l1 - l2) / (l1 + l2);
	det = l0 + l1 + l2;

	if (det <= AIR_QUALITY_EPSILON)
		det = 0.0;
	else det = 1.0 / det;

	return det;
}
__device__ void computeQuaility(
	REAL* ns, uint* nbNs, uint* inbNs, uint* nodePhases, REAL delta,
	uint* inos, REAL& quality0, REAL& quality1)
{
	uint ino;

	uint iend, nv;
	bool isEdges[4];
	isEdges[0] = isEdge_device(nbNs, inbNs, inos[0], inos[2]);
	isEdges[1] = isEdge_device(nbNs, inbNs, inos[0], inos[3]);
	isEdges[2] = isEdge_device(nbNs, inbNs, inos[1], inos[2]);
	isEdges[3] = isEdge_device(nbNs, inbNs, inos[1], inos[3]);

	uint phases[4];
	for (ino = 0u; ino < 4u; ino++)
		phases[ino] = nodePhases[inos[ino]];

	REAL2 ps[4];
	for (ino = 0u; ino < 4u; ino++) {
		ps[ino].x = ns[(inos[ino] << 1u) + 0u];
		ps[ino].y = ns[(inos[ino] << 1u) + 1u];
	}

	REAL2 dirs[6];
	dirs[0] = ps[2] - ps[0];
	dirs[1] = ps[3] - ps[0];
	dirs[2] = ps[2] - ps[1];
	dirs[3] = ps[3] - ps[1];
	dirs[4] = ps[1] - ps[0];
	dirs[5] = ps[3] - ps[2];

#if 0
	REAL lengthSqs[4];
	lengthSqs[0] = LengthSquared(ps[0] - ps[2]);
	lengthSqs[1] = LengthSquared(ps[0] - ps[3]);
	lengthSqs[2] = LengthSquared(ps[1] - ps[2]);
	lengthSqs[3] = LengthSquared(ps[1] - ps[3]);
	REAL tmp, det, area;

	{
		REAL lengthSq = LengthSquared(ps[0] - ps[1]);

		det = compQualityDet_device(lengthSq, lengthSqs[0], lengthSqs[2]);
		quality0 = Cross(dirs[2], dirs[0]) * det;

		det = compQualityDet_device(lengthSq, lengthSqs[1], lengthSqs[3]);
		tmp = Cross(dirs[1], dirs[3]) * det;

		if (quality0 > tmp) quality0 = tmp;

		lengthSq = LengthSquared(ps[2] - ps[3]);

		det = compQualityDet_device(lengthSq, lengthSqs[0], lengthSqs[1]);
		quality1 = Cross(dirs[0], dirs[1]) * det;

		det = compQualityDet_device(lengthSq, lengthSqs[2], lengthSqs[3]);
		tmp = Cross(dirs[3], dirs[2]) * det;

		if (quality1 > tmp) quality1 = tmp;
	}
#else
	REAL ls[6];
	ls[0] = Length(dirs[0]);
	ls[1] = Length(dirs[1]);
	ls[2] = Length(dirs[2]);
	ls[3] = Length(dirs[3]);
	ls[4] = Length(dirs[4]);
	ls[5] = Length(dirs[5]);
	if (ls[0]) dirs[0] *= 1.0 / ls[0];
	if (ls[1]) dirs[1] *= 1.0 / ls[1];
	if (ls[2]) dirs[2] *= 1.0 / ls[2];
	if (ls[3]) dirs[3] *= 1.0 / ls[3];
	if (ls[4]) dirs[4] *= 1.0 / ls[4];
	if (ls[5]) dirs[5] *= 1.0 / ls[5];

	REAL tmp;
	{
		quality0 = Dot(dirs[0], dirs[2]);
		tmp = Dot(dirs[1], dirs[3]);
		if (quality0 > tmp) quality0 = tmp;

		quality1 = Dot(dirs[0], dirs[1]);
		tmp = Dot(dirs[2], dirs[3]);
		if (quality1 > tmp) quality1 = tmp;


		if (fabs(quality0 - quality1) < 1.0e-2 && ((isEdges[0] && isEdges[3]) || (isEdges[1] && isEdges[2]))) {
		//if ((isEdges[0] || isEdges[3]) || (isEdges[1] || isEdges[2])) {
			REAL sin0 = Cross(dirs[2], dirs[0]);
			REAL sin1 = Cross(dirs[1], dirs[3]);
			REAL sin2 = Cross(dirs[0], dirs[1]);
			REAL sin3 = Cross(dirs[3], dirs[2]);

			quality0 = 1000000000000.0;
			quality1 = 1000000000000.0;
			if (isEdges[0]) {
				if (quality0 > sin0 * ls[2]) quality0 = sin0 * ls[2];
				if (quality1 > sin2 * ls[1]) quality1 = sin2 * ls[1];
			}
			if (isEdges[1]) {
				if (quality0 > sin1 * ls[3]) quality0 = sin1 * ls[3];
				if (quality1 > sin2 * ls[0]) quality1 = sin2 * ls[0];
			}
			if (isEdges[2]) {
				if (quality0 > sin0 * ls[0]) quality0 = sin0 * ls[0];
				if (quality1 > sin3 * ls[3]) quality1 = sin3 * ls[3];
			}
			if (isEdges[3]) {
				if (quality0 > sin1 * ls[1]) quality0 = sin1 * ls[1];
				if (quality1 > sin3 * ls[2]) quality1 = sin3 * ls[2];
			}
		}
	}
#endif
}
__global__ void getFlipElements_kernel(
	uint* fs, uint* es, uint* nbEFs, uint* nbFEs,
	REAL* ns, uint* nbNs, uint* inbNs, uint* nbNs2, uint* inbNs2, uint* nodePhases, REAL delta,
	uint numEdges, FlipElement* elems, uint* num)
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numEdges)
		return;

	uint ind = id << 1u;
	uint ino;
	uint inos[4];
	inos[0] = es[ind + 0u];
	inos[1] = es[ind + 1u];
	uint if0 = nbEFs[ind + 0u];
	uint if1 = nbEFs[ind + 1u];
	if (if1 == 0xffffffff || if0 == 0xffffffff)
		return;

	if (isEdge_device(nbNs, inbNs, inos[0], inos[1]))
		return;

	if0 *= 3u; if1 *= 3u;
	{
		uint pos0, tmp;
		for (ino = 0u; ino < 3u; ino++) {
			ind = fs[if0 + ino];
			if (ind == inos[0])
				tmp = ino;
			else if (ind == inos[1])
				pos0 = ino;
			else
				inos[2] = ind;
		}
		for (ino = 0u; ino < 3u; ino++) {
			ind = fs[if1 + ino];
			if (ind != inos[0] && ind != inos[1]) {
				inos[3] = ind;
				break;
			}
		}
		pos0 += 2u - tmp;
		if (pos0 == 0u || pos0 == 3u) {
			tmp = inos[0];
			inos[0] = inos[1];
			inos[1] = tmp;
		}
	}

	REAL quality0, quality1;
	computeQuaility(ns, nbNs, inbNs, nodePhases, delta, inos, quality0, quality1);
	if (quality0 < quality1) {
		if (checkFlip_device(fs, es, ns, nbEFs, nbFEs, nbNs2, inbNs2, nodePhases, id, inos, if0, if1))
			return;

		ino = atomicAdd(num, 1u);
		FlipElement elem;
		elem._id = id;
#if (AIR_FLIP_SORT == 1u)
		elem._quality = quality0;
#elif (AIR_FLIP_SORT == 2u)
		elem._quality = quality1;
#elif (AIR_FLIP_SORT == 3u)
		elem._quality = quality1 - quality0;
#endif
		elems[ino] = elem;
	}
}
__global__ void Flip_kernel(
	uint* fs, uint* es, uint* nbEFs, uint* nbFEs, 
	REAL* ns, uint* nbNs, uint* inbNs, uint* nbNs2, uint* inbNs2, uint* nodePhases, REAL delta,
	FlipElement* elems, uint numElems)
{
	for (uint ielem = 0u; ielem < numElems; ielem++) {
		FlipElement elem = elems[ielem];
		uint id = elem._id;
		uint ind = id << 1u;
		uint ino;
		uint inos[4];
		inos[0] = es[ind + 0u];
		inos[1] = es[ind + 1u];
		uint if0 = nbEFs[ind + 0u];
		uint if1 = nbEFs[ind + 1u];
		if (if1 == 0xffffffff || if0 == 0xffffffff)
			continue;

		if (isEdge_device(nbNs, inbNs, inos[0], inos[1]))
			continue;

		if0 *= 3u; if1 *= 3u;
		{
			uint pos0, tmp;
			for (ino = 0u; ino < 3u; ino++) {
				ind = fs[if0 + ino];
				if (ind == inos[0])
					tmp = ino;
				else if (ind == inos[1])
					pos0 = ino;
				else
					inos[2] = ind;
			}
			for (ino = 0u; ino < 3u; ino++) {
				ind = fs[if1 + ino];
				if (ind != inos[0] && ind != inos[1]) {
					inos[3] = ind;
					break;
				}
			}
			pos0 += 2u - tmp;
			if (pos0 == 0u || pos0 == 3u) {
				tmp = inos[0];
				inos[0] = inos[1];
				inos[1] = tmp;
			}
		}

		REAL quality0, quality1;
		computeQuaility(ns, nbNs, inbNs, nodePhases, delta, inos, quality0, quality1);
		if (quality0 < quality1) {
			if (checkFlip_device(fs, es, ns, nbEFs, nbFEs, nbNs2, inbNs2, nodePhases, id, inos, if0, if1))
				continue;

			//printf("%d\n", id);
			for (ino = 0u; ino < 3u; ino++) {
				ind = fs[if0 + ino];
				if (ind == inos[1]) {
					fs[if0 + ino] = inos[3];
					break;
				}
			}
			for (ino = 0u; ino < 3u; ino++) {
				ind = fs[if1 + ino];
				if (ind == inos[0]) {
					fs[if1 + ino] = inos[2];
					break;
				}
			}
			uint i0, i1, ie0, ie1, ind0, ind1, jnd0, jnd1;
			for (ino = 0u; ino < 3u; ino++) {
				ind = nbFEs[if0 + ino];
				if (ind != id) {
					i0 = es[(ind << 1u) + 0u];
					i1 = es[(ind << 1u) + 1u];
					if (i0 == inos[1] || i1 == inos[1]) {
						ind0 = ino;
						ie0 = ind;
					}
				}
				else jnd0 = ino;
			}
			for (ino = 0u; ino < 3u; ino++) {
				ind = nbFEs[if1 + ino];
				if (ind != id) {
					i0 = es[(ind << 1u) + 0u];
					i1 = es[(ind << 1u) + 1u];
					if (i0 == inos[0] || i1 == inos[0]) {
						ind1 = ino;
						ie1 = ind;
					}
				}
				else jnd1 = ino;
			}
			nbFEs[if0 + jnd0] = ie1;
			nbFEs[if0 + ind0] = id;
			nbFEs[if1 + jnd1] = ie0;
			nbFEs[if1 + ind1] = id;
			ie0 <<= 1u; ie1 <<= 1u;
			if0 /= 3u; if1 /= 3u;
			for (ino = 0u; ino < 2u; ino++) {
				ind = nbEFs[ie0 + ino];
				if (ind == if0) {
					nbEFs[ie0 + ino] = if1;
					break;
				}
			}
			for (ino = 0u; ino < 2u; ino++) {
				ind = nbEFs[ie1 + ino];
				if (ind == if1) {
					nbEFs[ie1 + ino] = if0;
					break;
				}
			}

			ind = id << 1u;
			if (inos[2] < inos[3]) {
				es[ind + 0u] = inos[2];
				es[ind + 1u] = inos[3];
			}
			else {
				es[ind + 0u] = inos[3];
				es[ind + 1u] = inos[2];
			}
		}
	}
}

//__global__ void Flip_kernel(
//	uint* fs, uint* es, uint* nbEFs, uint* nbFEs,
//	REAL* ns, uint* nbNs, uint* inbNs, REAL delta,
//	uint numEdges, uint* isApplied)
//{
//	for (uint id = 0u; id < numEdges; id++) {
//		uint ind = id << 1u;
//		uint ino;
//		uint inos[4];
//		inos[0] = es[ind + 0u];
//		inos[1] = es[ind + 1u];
//		uint if0 = nbEFs[ind + 0u];
//		uint if1 = nbEFs[ind + 1u];
//		if (if1 == 0xffffffff || if0 == 0xffffffff)
//			continue;
//
//		uint istart = inbNs[inos[0]];
//		uint iend = inbNs[inos[0] + 1u];
//		for (ino = istart; ino < iend; ino++) {
//			ind = nbNs[ino];
//			if (ind == inos[1])
//				break;
//		}
//		if (ino < iend)
//			continue;
//
//		if0 *= 3u; if1 *= 3u;
//		{
//			uint pos0, tmp;
//			for (ino = 0u; ino < 3u; ino++) {
//				ind = fs[if0 + ino];
//				if (ind == inos[0])
//					tmp = ino;
//				else if (ind == inos[1])
//					pos0 = ino;
//				else
//					inos[2] = ind;
//			}
//			for (ino = 0u; ino < 3u; ino++) {
//				ind = fs[if1 + ino];
//				if (ind != inos[0] && ind != inos[1]) {
//					inos[3] = ind;
//					break;
//				}
//			}
//			pos0 += 2u - tmp;
//			if (pos0 == 0u || pos0 == 3u) {
//				tmp = inos[0];
//				inos[0] = inos[1];
//				inos[1] = tmp;
//			}
//
//			if (checkFlip_device(fs, es, ns, nbEFs, nbFEs, nbNs, inbNs, id, inos, if0, if1))
//				continue;
//
//			istart = inbNs[inos[2]];
//			iend = inbNs[inos[2] + 1u];
//			for (ino = istart; ino < iend; ino++) {
//				ind = nbNs[ino];
//				if (ind == inos[3])
//					break;
//			}
//			if (ino < iend)
//				continue;
//		}
//
//		REAL quality0, quality1;
//		computeQuaility(ns, nbNs, inbNs, delta, inos, quality0, quality1);
//		if (quality0 < quality1) {
//			//printf("%d\n", id);
//			for (ino = 0u; ino < 3u; ino++) {
//				ind = fs[if0 + ino];
//				if (ind == inos[1]) {
//					fs[if0 + ino] = inos[3];
//					break;
//				}
//			}
//			for (ino = 0u; ino < 3u; ino++) {
//				ind = fs[if1 + ino];
//				if (ind == inos[0]) {
//					fs[if1 + ino] = inos[2];
//					break;
//				}
//			}
//			uint i0, i1, ie0, ie1, ind0, ind1, jnd0, jnd1;
//			for (ino = 0u; ino < 3u; ino++) {
//				ind = nbFEs[if0 + ino];
//				if (ind != id) {
//					i0 = es[(ind << 1u) + 0u];
//					i1 = es[(ind << 1u) + 1u];
//					if (i0 == inos[1] || i1 == inos[1]) {
//						ind0 = ino;
//						ie0 = ind;
//					}
//				}
//				else jnd0 = ino;
//			}
//			for (ino = 0u; ino < 3u; ino++) {
//				ind = nbFEs[if1 + ino];
//				if (ind != id) {
//					i0 = es[(ind << 1u) + 0u];
//					i1 = es[(ind << 1u) + 1u];
//					if (i0 == inos[0] || i1 == inos[0]) {
//						ind1 = ino;
//						ie1 = ind;
//					}
//				}
//				else jnd1 = ino;
//			}
//			nbFEs[if0 + jnd0] = ie1;
//			nbFEs[if0 + ind0] = id;
//			nbFEs[if1 + jnd1] = ie0;
//			nbFEs[if1 + ind1] = id;
//			ie0 <<= 1u; ie1 <<= 1u;
//			if0 /= 3u; if1 /= 3u;
//			for (ino = 0u; ino < 2u; ino++) {
//				ind = nbEFs[ie0 + ino];
//				if (ind == if0) {
//					nbEFs[ie0 + ino] = if1;
//					break;
//				}
//			}
//			for (ino = 0u; ino < 2u; ino++) {
//				ind = nbEFs[ie1 + ino];
//				if (ind == if1) {
//					nbEFs[ie1 + ino] = if0;
//					break;
//				}
//			}
//
//			ind = id << 1u;
//			if (inos[2] < inos[3]) {
//				es[ind + 0u] = inos[2];
//				es[ind + 1u] = inos[3];
//			}
//			else {
//				es[ind + 0u] = inos[3];
//				es[ind + 1u] = inos[2];
//			}
//			*isApplied = 1u;
//		}
//	}
//}