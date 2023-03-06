#include "GraphColoring.h"
#include "DeviceManager.cuh"

__global__ void getNbXs_kernel(
	uint* links, uint2* nbXs, uint numLink, uint numNbXs, uint* pos)
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numNbXs)
		return;

	uint ilink = id / numLink, i;
	uint2 nbX;
	nbX.x = links[id];
	nbX.y = ilink;
	nbXs[id] = nbX;
}
__global__ void initDepth_kernel(
	uint2* nbXs, uint* inbXs, uint* icurrs, uint* isEnds, uint numNodes)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= numNodes)
		return;

	uint istart = inbXs[id];
	uint iend = inbXs[id + 1u];
	uint2 nbX;
	if (istart < iend) {
		nbX = nbXs[istart];
		atomicAdd(isEnds + nbX.y, 1u);
	}
	else istart = 0xffffffff;
	icurrs[id] = istart;
}
__global__ void nextDepth_kernel(
	uint2* nbXs, uint* inbXs, uint* icurrs, uint* isEnds, uint numNodes, uint* isApplied)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id >= numNodes)
		return;

	uint ino = icurrs[id];
	if (ino == 0xffffffff)
		return;

	uint iend = inbXs[id + 1u];
	uint isEnd;
	uint2 nbX;
	nbX = nbXs[ino];
	isEnd = isEnds[nbX.y];
	if (isEnd == 0xffffffff) {
		for (ino++; ino < iend; ino++) {
			nbX = nbXs[ino];
			isEnd = isEnds[nbX.y];
			if (isEnd != 0xffffffff) {
				atomicAdd(isEnds + nbX.y, 1u);
				break;
			}
		}
		if (ino >= iend)
			ino = 0xffffffff;
		icurrs[id] = ino;
	}
	if (ino < iend) *isApplied = 1u;
}
__global__ void getSequence_kernel(
	uint* seqs, uint* isEnds, uint numLink, uint numLinks, uint seq)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= numLinks)
		return;

	uint isEnd = isEnds[id];
	if (isEnd < numLink || isEnd == 0xffffffff)
		return;

	isEnds[id] = 0xffffffff;
	seqs[id] = seq;
}

#define MAX_LINKSIZE		4u

__global__ void getColor_kernel(
	uint* links, uint* isEnds, uint* colors, bool* colorBuffer, uint maxColorSize, 
	uint numLink, uint numLinks)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= numLinks)
		return;

	uint isEnd = isEnds[id];
	if (isEnd < numLink || isEnd == 0xffffffff)
		return;

	/*
	uint offset = id * numLink, ind;
	uint ino, jno;
	bool buffer;
	for (ino = 0u; ino < maxColorSize; ino++) {
		for (jno = 0u; jno < numLink; jno++) {
			ind = links[offset + jno];
			ind *= maxColorSize;
			buffer = colorBuffer[ind + ino];
			if (buffer) break;
		}
		if (jno == numLink)
			break;
	}
	for (jno = 0u; jno < numLink; jno++) {
		ind = links[offset + jno];
		ind *= maxColorSize;
		colorBuffer[ind + ino] = true;
	}
	colors[id] = ino;
	isEnds[id] = 0xffffffff;*/
	uint inos[MAX_LINKSIZE];
	uint offset = id * numLink, ind;
	uint ino, jno;
	for (ino = 0u; ino < numLink; ino++) {
		inos[ino] = links[offset + ino];
		inos[ino] *= maxColorSize;
	}
	bool buffer;
	for (ino = 0u; ino < maxColorSize; ino++) {
		for (jno = 0u; jno < numLink; jno++) {
			buffer = colorBuffer[inos[jno] + ino];
			if (buffer) break;
		}
		if (jno == numLink)
			break;
	}
	for (jno = 0u; jno < numLink; jno++) {
		colorBuffer[inos[jno] + ino] = true;
	}
	colors[id] = ino;
	isEnds[id] = 0xffffffff;
}

__global__ void getMaxNeis_kernel(
	uint2* nbXs, uint* inbXs,
	uint numNodes, uint* maxNeis)
{
	extern __shared__ uint s_maxs[];
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	s_maxs[threadIdx.x] = 0u;
	if (id >= numNodes)
		return;

	uint istart, iend;
	istart = inbXs[id];
	iend = inbXs[id + 1u];
	if (iend > istart)
		s_maxs[threadIdx.x] = iend - istart;

	for (uint s = blockDim.x >> 1u; s > 32u; s >>= 1u) {
		__syncthreads();
		if (threadIdx.x < s)
			if (s_maxs[threadIdx.x] < s_maxs[threadIdx.x + s])
				s_maxs[threadIdx.x] = s_maxs[threadIdx.x + s];
	}
	__syncthreads();
	if (threadIdx.x < 32u) {
		warpMax(s_maxs, threadIdx.x);
		if (threadIdx.x == 0u)
			atomicMax(maxNeis, s_maxs[0]);
	}
}
__global__ void getNumNeis_kernel(
	uint2* nbXs, uint* inbXs, 
	uint numNodes, uint* numNeis)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= numNodes)
		return;

	uint istart, iend;
	uint ino;
	uint2 nbX;

	istart = inbXs[id];
	iend = inbXs[id + 1u];
	uint num = iend - istart;
	if (num > 1u) {
		num = num * (num - 1u) >> 1u;
		atomicAdd(numNeis, num);
	}
}
__global__ void getNeis_kernel(
	uint2* nbXs, uint* inbXs, uint2* neis, 
	uint numLinks, uint* numNeis)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= numLinks)
		return;

	uint istart, iend;
	uint ino, jno, pos;
	uint2 nbX, nei;

	istart = inbXs[id];
	iend = inbXs[id + 1u];
	uint num = iend - istart;
	if (num > 1u) {
		num = num * (num - 1u) >> 1u;
		pos = atomicAdd(numNeis, num);
		for (ino = istart + 1u; ino < iend; ino++) {
			nbX = nbXs[ino];
			nei.x = nbX.y;
			for (jno = istart; jno < ino; jno++) {
				nbX = nbXs[jno];
				nei.y = nbX.y;
				neis[pos++] = nei;
			}
		}
	}
}

__global__ void compColoring_kernel(
	uint2* neis, uint* ineis, 
	uint* icurrs, uint* colors, 
	uint numLinks, uint* isApplied) 
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numLinks)
		return;

	uint pos = icurrs[id];
	if (pos == 0xffffffff)
		return;

	uint istart = ineis[id];
	uint iend = ineis[id + 1u];
	uint ino, icolor;
	uint2 nei;

	uint prev = 0xffffffff;
	for (ino = istart + pos; ino < iend; ino++) {
		nei = neis[ino];
		if (nei.y != prev) {
			icolor = colors[nei.y];
			if (icolor == 0xffffffff)
				break;
			prev = nei.y;
		}
		pos++;
	}

	if (pos == iend - istart) {
		uint color = 0u;
		bool flag;
		do {
			flag = false;
			prev = 0xffffffff;
			for (ino = istart; ino < iend; ino++) {
				nei = neis[ino];
				if (nei.y != prev) {
					icolor = colors[nei.y];
					if (color == icolor) {
						color++;
						flag = true;
						break;
					}
					prev = nei.y;
				}
			}
		} while (flag);
		colors[id] = color;
		icurrs[id] = 0xffffffff;
	}
	else {
		icurrs[id] = pos;
		*isApplied = 1u;
	}
}