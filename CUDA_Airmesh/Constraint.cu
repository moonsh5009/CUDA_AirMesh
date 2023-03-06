#include "Constraint.cuh"

void Constraint::update(vector<uint>& es, vector<REAL>& cs, vector<uint>& bes, vector<REAL>& bcs, uint numNodes) {
	h_ids = es;
	h_cs = cs;
	h_ids.insert(h_ids.end(), bes.begin(), bes.end());
	h_cs.insert(h_cs.end(), bcs.begin(), bcs.end());
	d_ids = h_ids;
	d_cs = h_cs;
	_numSprings = (h_ids.size() >> 1u);

	/*thrust::sort_by_key(thrust::device_ptr<uint2>((uint2*)d_ids.begin()),
		thrust::device_ptr<uint2>((uint2*)d_ids.end()),
		thrust::device_ptr<REAL2>((REAL2*)(d_cs.begin())), uint2_CMP());*/

	GraphColoring::coloring(d_ids, 2u, d_colors, _colorSize, numNodes);

	d_lambdas.resize(_numSprings);
}
void Constraint::projectXPBD(Dvector<REAL>& ns, Dvector<REAL>& invMs) {
	for (int c = 0; c < _colorSize; c++) {
		project_XPBD_kernel << <divup(_numSprings, BLOCKSIZE), BLOCKSIZE >> > (
			param(), ns(), invMs(), c);
		CUDA_CHECK(cudaPeekAtLastError());
	}
}