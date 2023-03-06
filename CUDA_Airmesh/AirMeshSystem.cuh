#include "AirMeshSystem.h"
#include "DeviceManager.cuh"

__global__ void getNbNs_kernel(uint* es, uint* nbNs, uint numEdges) {
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numEdges)
		return;

	uint x = es[(id << 1u) + 1u];
	nbNs[id] = x;
}
__global__ void getNbNs2Buffer_kernel(uint* es, uint* buffer, uint numEdges) {
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numEdges)
		return;

	uint x = es[(id << 1u) + 0u];
	uint y = es[(id << 1u) + 1u];
	buffer[(id << 2u) + 0u] = x;
	buffer[(id << 2u) + 1u] = y;
	buffer[(id << 2u) + 2u] = y;
	buffer[(id << 2u) + 3u] = x;
}
__global__ void compConstraintsInf_kernel(
	uint* ids, REAL* ns, REAL* oldRs, REAL* newRs, REAL material, uint oldSize, uint newSize)
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= newSize)
		return;

	uint ino = id << 1u;
	REAL length, k;
	if (id < oldSize) {
		length = oldRs[ino + 0u];
		k = oldRs[ino + 1u];
	}
	else {
		uint ino0 = ids[ino + 0u];
		uint ino1 = ids[ino + 1u];

		REAL2 v0, v1;
		ino0 <<= 1u; ino1 <<= 1u;
		v0.x = ns[ino0 + 0u]; v0.y = ns[ino0 + 1u];
		v1.x = ns[ino1 + 0u]; v1.y = ns[ino1 + 1u];
		
		length = Length(v1 - v0);
		k = material;
	}
	newRs[ino + 0u] = length;
	newRs[ino + 1u] = k;
}

__global__ void initBoundaryVertices_kernel(
	REAL* ns, AABB boundary, REAL wgap, REAL hgap,
	uint wsize, uint hsize)
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	hsize -= 2u;
	REAL x, gap;
	uint ino;
	if (id < wsize) {
		gap = (REAL)id * wgap;
		ino = id << 1u;
		x = boundary._min.x + gap;
		ns[ino + 0u] = x;
		ns[ino + 1u] = boundary._min.y;

		ino = (id + wsize + hsize) << 1u;
		x = boundary._max.x - gap;
		ns[ino + 0u] = x;
		ns[ino + 1u] = boundary._max.y;
	}
	if (id < hsize) {
		gap = (REAL)(id + 1u) * hgap;
		ino = (id + wsize) << 1u;
		x = boundary._min.y + gap;
		ns[ino + 0u] = boundary._max.x;
		ns[ino + 1u] = x;

		ino = (id + (wsize << 1u) + hsize) << 1u;
		x = boundary._max.y - gap;
		ns[ino + 0u] = boundary._min.x;
		ns[ino + 1u] = x;
	}
}
__global__ void initBoundaryEdges_kernel(
	uint* es, uint size)
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < size) {
		uint ino0 = id;
		uint ino1 = id + 1u;
		if (id == size - 1u)
			ino1 = 0u;
		id <<= 1u;
		es[id + 0u] = ino0;
		es[id + 1u] = ino1;
	}
}

__global__ void compPredictPosition_kernel(
	REAL* ns, REAL* vs, REAL* invMs, REAL2 force, REAL dt, uint numNodes)
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numNodes)
		return;

	REAL invM = invMs[id];
	id <<= 1u;

	REAL2 n, v;
	n.x = ns[id + 0u];
	n.y = ns[id + 1u];
	v.x = vs[id + 0u];
	v.y = vs[id + 1u];

	v += dt * force * invM;
	n += dt * v;

	ns[id + 0u] = n.x;
	ns[id + 1u] = n.y;
	vs[id + 0u] = v.x;
	vs[id + 1u] = v.y;
}
__global__ void updateVelocities_kernel(
	REAL* n0s, REAL* ns, REAL* vs, REAL invdt, uint numNodes)
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numNodes)
		return;

	uint ino = id << 1u;
	REAL2 n0, n;
	n0.x = n0s[ino + 0u]; n0.y = n0s[ino + 1u];
	n.x = ns[ino + 0u]; n.y = ns[ino + 1u];
	
	n -= n0;
	n *= invdt;

	vs[ino + 0u] = n.x; vs[ino + 1u] = n.y;
}