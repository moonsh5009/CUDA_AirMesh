#include "Constraint.h"
#include "DeviceManager.cuh"

__global__ void compRestLength_kernel(
	uint *ids, REAL* ns, REAL* oldRs, REAL* newRs, uint istart, uint iend, uint size)
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= size)
		return;

	if (id < istart || id >= iend) {
		REAL length = oldRs[id];
		newRs[id] = length;
	}
	else {
		uint ino = id << 1u;
		uint ino0 = ids[ino + 0u];
		uint ino1 = ids[ino + 1u];

		ino0 <<= 1u; ino1 <<= 1u;
		REAL2 v0, v1;
		v0.x = ns[ino0 + 0u]; v0.y = ns[ino0 + 1u];
		v1.x = ns[ino1 + 0u]; v1.y = ns[ino1 + 1u];
		REAL length = Length(v1 - v0);
		newRs[id] = length;
	}
}
__global__ void project_XPBD_kernel(
	SpringParam springs, REAL* ns, REAL* invMs, uint currColor)
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= springs._size)
		return;

	uint color = springs._colors[id];
	if (color != currColor)
		return;

	uint ino = id << 1u;
	uint ino0 = springs._ids[ino + 0u];
	uint ino1 = springs._ids[ino + 1u];

	REAL w0 = invMs[ino0];
	REAL w1 = invMs[ino1];
	if (w0 == 0. && w1 == 0.)
		return;

	ino0 <<= 1u; ino1 <<= 1u;
	REAL2 v0, v1;
	v0.x = ns[ino0 + 0u]; v0.y = ns[ino0 + 1u];
	v1.x = ns[ino1 + 0u]; v1.y = ns[ino1 + 1u];

	REAL restLength = springs._cs[ino + 0u];
	REAL material = springs._cs[ino + 1u];
	REAL lambda = springs._lambdas[id];

	REAL2 dir = v1 - v0;
	REAL length = Length(dir);
	REAL constraint = length - restLength;
	REAL dt_lambda = (-constraint - material * lambda) / ((w0 + w1) + material);
	dir = dt_lambda / (length + FLT_EPSILON) * dir;
	lambda += dt_lambda;

	v0 -= w0 * dir;
	v1 += w1 * dir;
	springs._lambdas[id] = lambda;
	ns[ino0 + 0u] = v0.x; ns[ino0 + 1u] = v0.y;
	ns[ino1 + 0u] = v1.x; ns[ino1 + 1u] = v1.y;
}