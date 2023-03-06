#include "AirMesh.cuh"
#include "AirMeshSystem.h"
#include "triangle.h"

void AirMesh::CDT(const Dvector<uint>& segs, Dvector<uint>& fs, Dvector<uint>& es, Dvector<REAL>& ns, vector<REAL>& holes) {
	struct triangulateio in, out;
	in.numberofpoints = ns.size() >> 1u;
	in.numberofpointattributes = 0;
	in.numberofsegments = segs.size() >> 1u;
	in.numberofholes = holes.size() >> 1u;
	in.numberofregions = 0;

	in.pointlist = (REAL*)malloc(ns.size() * sizeof(REAL));
	in.pointmarkerlist = (int*)NULL;
	in.segmentlist = (int*)malloc(segs.size() * sizeof(int));
	in.segmentmarkerlist = (int*)NULL;
	in.regionlist = (REAL*)NULL;
	/*in.numberofpoints = numNodes;
	in.numberofpointattributes = 0;
	in.numberofsegments = numEdges;
	in.numberofholes = 1;
	in.numberofregions = 0;

	in.pointlist = (REAL*)malloc(ns.size() * sizeof(REAL));
	in.pointmarkerlist = (int*)NULL;
	in.segmentlist = (int*)malloc(es.size() * sizeof(int));
	in.segmentmarkerlist = (int*)NULL;
	in.holelist = (REAL*)malloc(2u * sizeof(REAL));
	in.regionlist = (REAL*)NULL;

	in.holelist[0] = 50;
	in.holelist[1] = 50.0;*/
	if (holes.size()) {
		in.holelist = (REAL*)malloc(holes.size() * sizeof(REAL));
		memcpy(in.holelist, &holes[0], holes.size() * sizeof(REAL));
	}
	else {
		in.holelist = (REAL*)NULL;
	}
	CUDA_CHECK(cudaMemcpy(in.pointlist, ns(), ns.size() * sizeof(REAL), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(in.segmentlist, segs(), segs.size() * sizeof(int), cudaMemcpyDeviceToHost));

	// initialize output
	out.pointlist = (REAL*)NULL;
	out.pointmarkerlist = (int*)NULL;
	out.pointattributelist = (REAL*)NULL;
	out.trianglelist = (int*)NULL;
	out.triangleattributelist = (REAL*)NULL;
	out.segmentlist = (int*)NULL;
	out.segmentmarkerlist = (int*)NULL;
	out.holelist = (REAL*)NULL;
	out.regionlist = (REAL*)NULL;
	out.edgelist = (int*)NULL;
	out.edgemarkerlist = (int*)NULL;
	out.neighborlist = (int*)NULL;

	// flags to use: pqzQ
	//triangulate((char*)"pzenQ", &in, &out, (struct triangulateio*)NULL);
	triangulate((char*)"pzQ", &in, &out, (struct triangulateio*)NULL);

	if (out.numberoftriangles) {
		fs.resize(out.numberoftriangles * 3u);
		//es.resize(out.numberofedges << 1u);
		CUDA_CHECK(cudaMemcpy(fs(), out.trianglelist, out.numberoftriangles * 3u * sizeof(uint), cudaMemcpyHostToDevice));
		/*CUDA_CHECK(cudaMemcpy(es(), out.edgelist, (out.numberofedges << 1u) * sizeof(uint), cudaMemcpyHostToDevice));
		sortEdgeIndex << <divup(out.numberofedges, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
			es(), out.numberofedges);
		CUDA_CHECK(cudaPeekAtLastError());
		thrust::sort(thrust::device_ptr<uint2>((uint2*)es.begin()), 
			thrust::device_ptr<uint2>((uint2*)es.end()), uint2_CMP());*/
		/*printf("%d\n", out.numberofedges);
		for (int i = 0; i < out.numberofedges * 2; i += 2)
			printf("edges %d, %d\n", out.edgelist[i], out.edgelist[i + 1]);*/
	}

	free(in.pointlist);
	free(in.pointmarkerlist);
	free(in.segmentlist);
	free(in.segmentmarkerlist);
	free(in.holelist);
	free(in.regionlist);

	free(out.pointlist);
	free(out.pointmarkerlist);
	free(out.pointattributelist);
	free(out.trianglelist);
	free(out.triangleattributelist);
	free(out.segmentlist);
	free(out.segmentmarkerlist);
	free(out.edgelist);
	free(out.edgemarkerlist);
	free(out.neighborlist);
}
void AirMesh::getRigidSpring(Dvector<uint>& fs, Dvector<uint>& es, Dvector<REAL>& ns, Dvector<uint>& nbEFs, Dvector<uint>& nbFEs) {
	vector<REAL> holes(2u, FLT_MAX);
	AirMesh::CDT(es, fs, es, ns, holes);
	reorderElements(fs, es, ns, nbEFs, nbFEs);

	/*uint numEdges = es.size() >> 1u;
	uint numNodes = ns.size() >> 1u;
	DPrefixArray<uint> nbNs;
	uint h_tmp;
	nbNs._index.resize(numNodes + 1u);
	nbNs._index.memset(0);

	uint* d_num, numElems;
	CUDA_CHECK(cudaMalloc((void**)&d_num, sizeof(uint)));

	Dvector<uint> nodePhases(numNodes);
	nodePhases.memset(0);
	uint itr;
	Dvector<FlipElement> elems(numEdges);
	for (itr = 0u; itr < AIR_FLIP_ITERATION; itr++) {
		CUDA_CHECK(cudaMemset(d_num, 0, sizeof(uint)));
		getFlipElements_kernel << <divup(numEdges, BLOCKSIZE), BLOCKSIZE >> > (
			fs(), es(), nbEFs(), nbFEs(),
			ns(), nbNs._array(), nbNs._index(), nodePhases(), 0.0,
			numEdges, elems(), d_num);
		CUDA_CHECK(cudaPeekAtLastError());
		CUDA_CHECK(cudaMemcpy(&numElems, d_num, sizeof(uint), cudaMemcpyDeviceToHost));
		if (!numElems)
			break;

		Flip_kernel << <1, 1 >> > (
			fs(), es(), nbEFs(), nbFEs(),
			ns(), nbNs._array(), nbNs._index(), nodePhases(), 0.0,
			elems(), elems.size());
		CUDA_CHECK(cudaPeekAtLastError());
	}
	CUDA_CHECK(cudaFree(d_num));*/
}
void AirMesh::reorderElements(
	const Dvector<uint>& fs, Dvector<uint>& es,
	Dvector<REAL>& ns, Dvector<uint>& nbEFs, Dvector<uint>& nbFEs)
{
	uint numFaces = fs.size() / 3u;
	uint numNodes = ns.size() >> 1u;
	uint numEdges;
	Dvector<uint> ids;
	Dvector<uint> buffer;
	uint h_tmp;

	//nbFEs.resize(fs.size());
	//nbEFs.resize(es.size());

	ids.resize(numNodes + 1u);
	ids.memset(0);
	getNbFsIds_kernel << <divup(numFaces, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
		fs(), ids(), numFaces);
	CUDA_CHECK(cudaPeekAtLastError());

	thrust::inclusive_scan(thrust::device_ptr<uint>(ids.begin()),
		thrust::device_ptr<uint>(ids.end()), thrust::device_ptr<uint>(ids.begin()));
	CUDA_CHECK(cudaMemcpy(&h_tmp, ids() + numNodes, sizeof(uint), cudaMemcpyDeviceToHost));
	Dvector<uint> idsTmp;
	idsTmp = ids;

	Dvector<uint> nbFs;
	nbFs.resize(h_tmp);
	getNbFs_kernel << <divup(numFaces, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
		fs(), nbFs(), idsTmp(), numFaces);
	CUDA_CHECK(cudaPeekAtLastError());
	idsTmp.free();

	buffer.resize(1u);
	buffer.memset(0);
	getEdgesSize_kernel << <divup(numFaces, BLOCKSIZE), BLOCKSIZE >> > (
		fs(), nbFs(), ids(), buffer(), numFaces);
	CUDA_CHECK(cudaPeekAtLastError());
	CUDA_CHECK(cudaMemcpy(&numEdges, buffer(), sizeof(uint), cudaMemcpyDeviceToHost));

	buffer.memset(0);
	es.resize(numEdges << 1u);
	nbEFs.resize(numEdges << 1u);
	buildEdgeNbFs_kernel << <divup(numFaces, BLOCKSIZE), BLOCKSIZE >> > (
		fs(), es(), nbFs(), ids(), nbEFs(), buffer(), numFaces);
	CUDA_CHECK(cudaPeekAtLastError());

	ids.resize(numFaces);
	ids.memset(0);
	nbFEs.resize(numFaces * 3u);
	buildNbFEs_kernel << <divup(numEdges, BLOCKSIZE), BLOCKSIZE >> > (
		nbEFs(), nbFEs(), ids(), numEdges);
	CUDA_CHECK(cudaPeekAtLastError());

	reorderNbFEs_kernel << <divup(numFaces, BLOCKSIZE), BLOCKSIZE >> > (
		fs(), es(), nbFEs(), numFaces);
	CUDA_CHECK(cudaPeekAtLastError());
}

void AirMesh::init(Dvector<uint>& es, Dvector<REAL>& ns, Dvector<uint>& nodePhases, vector<REAL>& holes) {
	uint numNodes = ns.size() >> 1u;
	AirMesh::CDT(es, d_fs, d_es, ns, holes);
	AirMesh::reorderElements(d_fs, d_es, ns, d_nbEFs, d_nbFEs);

	d_fs.copyToHost(h_fs);
	d_es.copyToHost(h_es);
	
	printf("edge: %d, node: %d, air tri: %d, air edge: %d\n", es.size() >> 1u, ns.size() >> 1u, d_fs.size() / 3u, d_es.size() >> 1u);
}

void AirMesh::getFlipElements(
	Dvector<REAL>& ns, DPrefixArray<uint>& nbNs, DPrefixArray<uint>& nbNs2, Dvector<uint> nodePhases, REAL delta, Dvector<FlipElement>& elems, uint* numElems)
{
	uint numEdges = d_es.size() >> 1u;

	uint* d_num;
	CUDA_CHECK(cudaMalloc((void**)&d_num, sizeof(uint)));

	CUDA_CHECK(cudaMemset(d_num, 0, sizeof(uint)));
	getFlipElements_kernel << <divup(numEdges, BLOCKSIZE), BLOCKSIZE >> > (
		d_fs(), d_es(), d_nbEFs(), d_nbFEs(),
		ns(), nbNs._array(), nbNs._index(), nbNs2._array(), nbNs2._index(), nodePhases(), delta,
		numEdges, elems(), d_num);
	CUDA_CHECK(cudaPeekAtLastError());
	CUDA_CHECK(cudaMemcpy(numElems, d_num, sizeof(uint), cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaFree(d_num));
	/*uint* d_num, h_num;
	CUDA_CHECK(cudaMalloc((void**)&d_num, sizeof(uint)));
	CUDA_CHECK(cudaMemset(d_num, 0, sizeof(uint)));
	getNumFlipElements_kernel << <divup(numEdges, BLOCKSIZE), BLOCKSIZE >> > (
		d_fs(), d_es(), d_nbEFs(), d_nbFEs(),
		ns(), nbNs._array(), nbNs._index(), delta,
		numEdges, d_num);
	CUDA_CHECK(cudaPeekAtLastError());
	CUDA_CHECK(cudaMemcpy(&h_num, d_num, sizeof(uint), cudaMemcpyDeviceToHost));

	elems.resize(h_num);
	if (h_num) {
		CUDA_CHECK(cudaMemset(d_num, 0, sizeof(uint)));
		getFlipElements_kernel << <divup(numEdges, BLOCKSIZE), BLOCKSIZE >> > (
			d_fs(), d_es(), d_nbEFs(), d_nbFEs(),
			ns(), nbNs._array(), nbNs._index(), delta,
			numEdges, elems(), d_num);
		CUDA_CHECK(cudaPeekAtLastError());
	}
	CUDA_CHECK(cudaFree(d_num));*/
}
void AirMesh::getContactElements(
	Dvector<REAL>& ns, Dvector<REAL>& cs, DPrefixArray<uint>& nbNs,REAL delta,
	Dvector<ContactElement>& elems, uint* numElems)
{
	uint numFaces = d_fs.size() / 3u;

	uint* d_num;
	CUDA_CHECK(cudaMalloc((void**)&d_num, sizeof(uint)));

	CUDA_CHECK(cudaMemset(d_num, 0, sizeof(uint)));
	getContactElements_kernel << <divup(numFaces, BLOCKSIZE), BLOCKSIZE >> > (
		d_fs(), ns(), nbNs._array(), nbNs._index(), delta,
		numFaces, elems(), d_num);
	CUDA_CHECK(cudaPeekAtLastError());
	CUDA_CHECK(cudaMemcpy(numElems, d_num, sizeof(uint), cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaFree(d_num));
}
void AirMesh::flip(
	Dvector<REAL>& ns, DPrefixArray<uint>& nbNs, DPrefixArray<uint>& nbNs2, Dvector<uint> nodePhases, REAL delta)
{
	{
		//uint numEdges = d_es.size() >> 1u;
		//bool *d_isApplied, isApplied;
		//CUDA_CHECK(cudaMalloc((void**)&d_isApplied, sizeof(bool)));

		//isApplied = true;
		///*while(isApplied) {
		//	FlipTest_kernel << <1, 1 >> > (
		//		d_fs(), d_es(), d_nbEFs(), d_nbFEs(), ns(), nodePhases(), d_isApplied, numEdges);
		//	CUDA_CHECK(cudaPeekAtLastError());
		//	CUDA_CHECK(cudaMemcpy(&isApplied, d_isApplied, sizeof(bool), cudaMemcpyDeviceToHost));
		//}*/
		//for (uint itr = 0u; itr < 10u; itr++) {
		//	FlipTest_kernel << <1, 1 >> > (
		//		d_fs(), d_es(), d_nbEFs(), d_nbFEs(), 
		//		ns(), nbNs._array(), nbNs._index(), nodePhases(), objs(),
		//		numEdges, d_isApplied);
		//	CUDA_CHECK(cudaPeekAtLastError());
		//	CUDA_CHECK(cudaMemcpy(&isApplied, d_isApplied, sizeof(bool), cudaMemcpyDeviceToHost));
		//	if (!isApplied)
		//		break;
		//}

		//CUDA_CHECK(cudaFree(d_isApplied));
	}
	uint itr;
#ifndef AIR_FLIP_CPU
	Dvector<FlipElement> elems(d_es.size() >> 1u);
	uint numElems = 0u;
	for (itr = 0u; itr < AIR_FLIP_ITERATION; itr++) {
		getFlipElements(ns, nbNs, nbNs2, nodePhases, delta, elems, &numElems);
		//printf("elem %d\n", numElems);
		if (!numElems)
			break;

#if (AIR_FLIP_SORT > 0u)
		thrust::sort(thrust::device_ptr<FlipElement>(elems.begin()),
			thrust::device_ptr<FlipElement>(elems.begin() + numElems), FlipElement_CMP());
#endif
		Flip_kernel << <1, 1 >> > (
			d_fs(), d_es(), d_nbEFs(), d_nbFEs(),
			ns(), nbNs._array(), nbNs._index(), nbNs2._array(), nbNs2._index(), nodePhases(), delta,
			elems(), numElems);
		CUDA_CHECK(cudaPeekAtLastError());
		
		/*vector<uint> test;
		d_nbEFs.copyToHost(test);
		for (int i = 0; i < test.size(); i += 2) {
			if (test[i] == test[i + 1])
				printf("chk0 %d %d\n", test[i], test[i + 1]);
		}
		d_es.copyToHost(test);
		for (int i = 0; i < test.size(); i += 2) {
			if (test[i] == test[i + 1])
				printf("chk1 %d %d\n", test[i], test[i + 1]);
		}
		d_nbFEs.copyToHost(test);
		for (int i = 0; i < test.size(); i += 3) {
			if (test[i] == test[i + 1] || test[i] == test[i + 2] || test[i + 1] == test[i + 2])
				printf("chk2 %d %d %d\n", test[i], test[i + 1], test[i + 2]);
		}
		d_fs.copyToHost(test);
		for (int i = 0; i < test.size(); i += 3) {
			if (test[i] == test[i + 1] || test[i] == test[i + 2] || test[i + 1] == test[i + 2])
				printf("chk3 %d %d %d\n", test[i], test[i + 1], test[i + 2]);
		}*/
	}
	/*uint numEdges = d_es.size() >> 1u;
	uint isApplied;
	uint* d_isApplied;
	CUDA_CHECK(cudaMalloc((void**)&d_isApplied, sizeof(uint)));
	for (itr = 0u; itr < AIR_FLIP_ITERATION; itr++) {
		CUDA_CHECK(cudaMemset(d_isApplied, 0, sizeof(uint)));
		Flip_kernel << <1, 1 >> > (
			d_fs(), d_es(), d_nbEFs(), d_nbFEs(),
			ns(), nbNs._array(), nbNs._index(), nodePhases(),
			numEdges, d_isApplied);
		CUDA_CHECK(cudaPeekAtLastError());
		CUDA_CHECK(cudaMemcpy(&isApplied, d_isApplied, sizeof(uint), cudaMemcpyDeviceToHost));
		if (!isApplied) break;
	}*/
#else
	uint numEdges = h_es.size() >> 1u;
	for (uint id = 0u; id < numEdges; id++) {
		uint ind = id << 1u;
		uint ino;
		uint inos[4];
		inos[0] = h_es[ind + 0u];
		inos[1] = h_es[ind + 1u];
		uint if0 = h_nbEFs[ind + 0u];
		uint if1 = h_nbEFs[ind + 1u];
		if (if1 == 0xffffffff || if0 == 0xffffffff)
			continue;

		uint istart = h_inbNs[inos[0]];
		uint iend = h_inbNs[inos[0] + 1u];
		for (ino = istart; ino < iend; ino++) {
			ind = nbNs[ino];
			if (ind == inos[1])
				break;
		}
		if (ino < iend)
			continue;

		if0 *= 3u; if1 *= 3u;
		{
			uint pos0, tmp;
			for (ino = 0u; ino < 3u; ino++) {
				ind = h_fs[if0 + ino];
				if (ind == inos[0])
					tmp = ino;
				else if (ind == inos[1])
					pos0 = ino;
				else
					inos[2] = ind;
			}
			for (ino = 0u; ino < 3u; ino++) {
				ind = h_fs[if1 + ino];
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

			if (checkFlip_device(h_fs, h_es, h_ns, h_nbEFs, h_nbFEs, h_nbNs, h_inbNs, id, inos, if0, if1))
				continue;

			istart = h_inbNs[inos[2]];
			iend = h_inbNs[inos[2] + 1u];
			for (ino = istart; ino < iend; ino++) {
				ind = nbNs[ino];
				if (ind == inos[3])
					break;
			}
			if (ino < iend)
				continue;
		}

		REAL quality0, quality1;
		computeQuaility(h_ns, h_nbNs, h_inbNs, inos, quality0, quality1);
		if (quality0 < quality1 - AIR_FLIP_EPSILON) {
			//printf("%d\n", id);
			for (ino = 0u; ino < 3u; ino++) {
				ind = h_fs[if0 + ino];
				if (ind == inos[1]) {
					fs[if0 + ino] = inos[3];
					break;
				}
			}
			for (ino = 0u; ino < 3u; ino++) {
				ind = h_fs[if1 + ino];
				if (ind == inos[0]) {
					fs[if1 + ino] = inos[2];
					break;
				}
			}
			uint i0, i1, ie0, ie1, ind0, ind1, jnd0, jnd1;
			for (ino = 0u; ino < 3u; ino++) {
				ind = h_nbFEs[if0 + ino];
				if (ind != id) {
					i0 = h_es[(ind << 1u) + 0u];
					i1 = h_es[(ind << 1u) + 1u];
					if (i0 == inos[1] || i1 == inos[1]) {
						ind0 = ino;
						ie0 = ind;
					}
				}
				else jnd0 = ino;
			}
			for (ino = 0u; ino < 3u; ino++) {
				ind = h_nbFEs[if1 + ino];
				if (ind != id) {
					i0 = h_es[(ind << 1u) + 0u];
					i1 = h_es[(ind << 1u) + 1u];
					if (i0 == inos[0] || i1 == inos[0]) {
						ind1 = ino;
						ie1 = ind;
					}
				}
				else jnd1 = ino;
			}
			h_nbFEs[if0 + jnd0] = ie1;
			h_nbFEs[if0 + ind0] = id;
			h_nbFEs[if1 + jnd1] = ie0;
			h_nbFEs[if1 + ind1] = id;
			ie0 <<= 1u; ie1 <<= 1u;
			if0 /= 3u; if1 /= 3u;
			for (ino = 0u; ino < 2u; ino++) {
				ind = h_nbEFs[ie0 + ino];
				if (ind == if0) {
					h_nbEFs[ie0 + ino] = if1;
					break;
				}
			}
			for (ino = 0u; ino < 2u; ino++) {
				ind = h_nbEFs[ie1 + ino];
				if (ind == if1) {
					h_nbEFs[ie1 + ino] = if0;
					break;
				}
			}

			ind = id << 1u;
			if (inos[2] < inos[3]) {
				h_es[ind + 0u] = inos[2];
				h_es[ind + 1u] = inos[3];
			}
			else {
				h_es[ind + 0u] = inos[3];
				h_es[ind + 1u] = inos[2];
			}
		}
	}
#endif
	printf("Flip iteration: %d\n", itr);

	/*CUDA_CHECK(cudaDeviceSynchronize());
	ctimer timer = CNOW;
	Dvector<uint> fs0;
	Dvector<uint> es0;
	Dvector<uint> nbEFs0;
	Dvector<uint> nbFEs0;
	fs0 = d_fs;
	es0 = d_es;
	nbEFs0 = d_nbEFs;
	nbFEs0 = d_nbFEs;
	for (itr = 0u; itr < 500u; itr++) {
		getFlipElements(ns, nbNs, nodePhases, objs, elems);
		if (!elems.size())
			break;

		thrust::sort(thrust::device_ptr<FlipElement>(elems.begin()),
			thrust::device_ptr<FlipElement>(elems.end()), FlipElement_CMP1());

		Flip_kernel << <1, 1 >> > (
			d_fs(), d_es(), d_nbEFs(), d_nbFEs(),
			ns(), nbNs._array(), nbNs._index(), nodePhases(), objs(),
			elems(), elems.size());
		CUDA_CHECK(cudaPeekAtLastError());
	}

	printf("Flip 1 iteration: %d\n", itr);
	CUDA_CHECK(cudaDeviceSynchronize());
	printf("%f msec\n", (CNOW - timer) * 0.0001);

	d_fs = fs0;
	d_es = es0;
	d_nbEFs = nbEFs0;
	d_nbFEs = nbFEs0;

	CUDA_CHECK(cudaDeviceSynchronize());
	timer = CNOW;

	for (itr = 0u; itr < 500u; itr++) {
		getFlipElements(ns, nbNs, nodePhases, objs, elems);
		if (!elems.size())
			break;

		thrust::sort(thrust::device_ptr<FlipElement>(elems.begin()),
			thrust::device_ptr<FlipElement>(elems.end()), FlipElement_CMP2());

		Flip_kernel << <1, 1 >> > (
			d_fs(), d_es(), d_nbEFs(), d_nbFEs(),
			ns(), nbNs._array(), nbNs._index(), nodePhases(), objs(),
			elems(), elems.size());
		CUDA_CHECK(cudaPeekAtLastError());
	}

	printf("Flip 2 iteration: %d\n", itr);
	CUDA_CHECK(cudaDeviceSynchronize());
	printf("%f msec\n", (CNOW - timer) * 0.0001);

	d_fs = fs0;
	d_es = es0;
	d_nbEFs = nbEFs0;
	d_nbFEs = nbFEs0;

	CUDA_CHECK(cudaDeviceSynchronize());
	timer = CNOW;

	for (itr = 0u; itr < 500u; itr++) {
		getFlipElements(ns, nbNs, nodePhases, objs, elems);
		if (!elems.size())
			break;

		Flip_kernel << <1, 1 >> > (
			d_fs(), d_es(), d_nbEFs(), d_nbFEs(),
			ns(), nbNs._array(), nbNs._index(), nodePhases(), objs(),
			elems(), elems.size());
		CUDA_CHECK(cudaPeekAtLastError());
	}

	printf("Flip 3 iteration: %d\n", itr);
	CUDA_CHECK(cudaDeviceSynchronize());*/
}
void AirMesh::collision(
	Dvector<REAL>& ns, Dvector<REAL>& n0s, Dvector<REAL>& invMs,
	Dvector<REAL>& cs, DPrefixArray<uint>& nbNs,
	REAL delta, REAL friction, uint iteration)
{
	uint numNodes = ns.size() >> 1u;
	uint numFaces = d_fs.size() / 3u;

	Dvector<REAL> ds(ns.size());
	Dvector<uint> invds(numNodes);
	uint isApplied;
	uint* d_isApplied;
	CUDA_CHECK(cudaMalloc((void**)&d_isApplied, sizeof(uint)));

	uint itr;
	{
		/* uint numFaces = d_fs.size() / 3u;
		bool *d_isApplied, isApplied;
		CUDA_CHECK(cudaMalloc((void**)&d_isApplied, sizeof(bool)));

		uint itr;
		for (itr = 0u; itr < iteration; itr++) {
			CollisionSolveTest_kernel << <1, 1 >> > (
				d_fs(), ns(), invMs(), nodePhases(), delta, d_isApplied, numFaces);
			CUDA_CHECK(cudaPeekAtLastError());
			CUDA_CHECK(cudaMemcpy(&isApplied, d_isApplied, sizeof(bool), cudaMemcpyDeviceToHost));
			if (!isApplied)
				break;
		}
		printf("Collision iteration: %d\n", itr);

		CUDA_CHECK(cudaFree(d_isApplied));*/
	}
	{
		/*Dvector<ContactElement> elems(numFaces);
		Dvector<uint> buffer(numFaces);
		Dvector<uint> seqs;
		uint numElems, seqSize;

		for (itr = 0u; itr < iteration; itr++) {
			getContactElements(ns, nbNs, delta, elems, &numElems);
			if (!numElems)
				break;

			thrust::sort(thrust::device_ptr<ContactElement>(elems.begin()),
				thrust::device_ptr<ContactElement>(elems.begin() + numElems), ContactElement_CMP());

			buffer.resize(numElems * 3u);
			getContactElementBuffer_kernel << <divup(numElems, BLOCKSIZE), BLOCKSIZE >> > (
				d_fs(), elems(), numElems, buffer());
			CUDA_CHECK(cudaPeekAtLastError());

			GraphColoring::sequential(buffer, 3u, seqs, seqSize, numNodes);
			buffer.clear();

			for (uint currSeq = 0u; currSeq < seqSize; currSeq++) {
				Collision_kernel << <divup(numElems, BLOCKSIZE), BLOCKSIZE >> > (
					d_fs(), ns(), invMs(), nbNs._array(), nbNs._index(), delta,
					elems(), numElems, seqs(), currSeq);
				CUDA_CHECK(cudaPeekAtLastError());
			}
		}*/
	}

#if 0	// Graph Coloring
	Dvector<uint> colors(d_fs.size());
	uint colorSize;

	GraphColoring::coloring(d_fs, 3u, colors, colorSize, numNodes);

	for (itr = 0u; itr < iteration; itr++) {
		CUDA_CHECK(cudaMemset(d_isApplied, 0, sizeof(uint)));
		for (uint currColor = 0u; currColor < colorSize; currColor++) {
			Collision_kernel << <divup(numFaces, BLOCKSIZE), BLOCKSIZE >> > (
				d_fs(), ns(), invMs(), cs(), nbNs._array(), nbNs._index(), delta,
				numFaces, colors(), currColor, d_isApplied);
			CUDA_CHECK(cudaPeekAtLastError());
		}
		CUDA_CHECK(cudaMemcpy(&isApplied, d_isApplied, sizeof(uint), cudaMemcpyDeviceToHost));
		if (!isApplied) break;
	}
#else	// Jacobi
	for (itr = 0u; itr < iteration; itr++) {
		ds.memset(0); invds.memset(0);
		CUDA_CHECK(cudaMemset(d_isApplied, 0, sizeof(uint)));
		CollisionJacobi_kernel << <divup(numFaces, BLOCKSIZE), BLOCKSIZE >> > (
			d_fs(), ns(), invMs(), cs(), nbNs._array(), nbNs._index(), delta,
			numFaces, ds(), invds(), d_isApplied);
		CUDA_CHECK(cudaPeekAtLastError());
		CUDA_CHECK(cudaMemcpy(&isApplied, d_isApplied, sizeof(uint), cudaMemcpyDeviceToHost));
		if (!isApplied) break;

		REAL omega = 2.0 * (REAL)itr / (REAL)(iteration - 1u);
		ApplyJacobi_kernel << <divup(numNodes, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
			ns(), ds(), invds(), omega, numNodes);
		CUDA_CHECK(cudaPeekAtLastError());
	}
#endif
	// Friction
	{
		/*ds.memset(0); invds.memset(0);
		CUDA_CHECK(cudaMemset(d_isApplied, 0, sizeof(uint)));
		FrictionJacobi_kernel << <divup(numFaces, BLOCKSIZE), BLOCKSIZE >> > (
			d_fs(), ns(), n0s(), invMs(), nbNs._array(), nbNs._index(), delta, friction,
			numFaces, ds(), invds(), d_isApplied);
		CUDA_CHECK(cudaPeekAtLastError());
		CUDA_CHECK(cudaMemcpy(&isApplied, d_isApplied, sizeof(uint), cudaMemcpyDeviceToHost));
		if (isApplied) {
			ApplyJacobi_kernel << <divup(numNodes, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
				ns(), ds(), invds(), 0.0, numNodes);
			CUDA_CHECK(cudaPeekAtLastError());
		}*/
	}
	CUDA_CHECK(cudaFree(d_isApplied));
	printf("Collision iteration: %d\n", itr);
}
void AirMesh::calcPredictPosition(
	Dvector<REAL>& pns, uint iteration) 
{
	uint numFaces = d_fs.size() / 3u;
	uint numNodes = pns.size() >> 1u;

	uint isApplied, itr;
	uint* d_isApplied;
	CUDA_CHECK(cudaMalloc((void**)&d_isApplied, sizeof(uint)));

	Dvector<uint> colors(d_fs.size());
	uint colorSize;

	GraphColoring::coloring(d_fs, 3u, colors, colorSize, numNodes);
	for (itr = 0u; itr < iteration; itr++) {
		CUDA_CHECK(cudaMemset(d_isApplied, 0, sizeof(uint)));
		for (uint currColor = 0u; currColor < colorSize; currColor++) {
			calcPredictPositionGC_kernel << <divup(numFaces, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
				d_fs(), pns(), numFaces, colors(), currColor, d_isApplied);
			CUDA_CHECK(cudaPeekAtLastError());
		}
		CUDA_CHECK(cudaMemcpy(&isApplied, d_isApplied, sizeof(uint), cudaMemcpyDeviceToHost));
		if (!isApplied) break;
	}

	/*Dvector<REAL> ds(pns.size());
	Dvector<uint> invds(numNodes);
	for (itr = 0u; itr < iteration; itr++) {
		ds.memset(0); invds.memset(0);
		CUDA_CHECK(cudaMemset(d_isApplied, 0, sizeof(uint)));
		calcPredictPositionJacobi_kernel << <divup(numFaces, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
			d_fs(), pns(), numFaces, ds(), invds(), d_isApplied);
		CUDA_CHECK(cudaPeekAtLastError());
		CUDA_CHECK(cudaMemcpy(&isApplied, d_isApplied, sizeof(uint), cudaMemcpyDeviceToHost));
		if (!isApplied) break;

		ApplyJacobi_kernel << <divup(numNodes, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
			pns(), ds(), invds(), 0.0, numNodes);
		CUDA_CHECK(cudaPeekAtLastError());
	}*/

	CUDA_CHECK(cudaFree(d_isApplied));
	printf("calcPredictPosition iteration: %d\n", itr);
}
void AirMesh::resolveCollision(
	Dvector<REAL>& ns, Dvector<REAL>& n0s, Dvector<REAL>& invMs,
	Dvector<REAL>& cs, DPrefixArray<uint>& nbNs, DPrefixArray<uint>& nbNs2, Dvector<uint>& nodePhases,
	REAL delta, REAL friction, uint iteration)
{
	if (!d_fs.size())
		return;
	CUDA_CHECK(cudaDeviceSynchronize());
	ctimer timer = CNOW;

	if(_collision)
		collision(ns, n0s, invMs, cs, nbNs, delta, friction, iteration);

	CUDA_CHECK(cudaDeviceSynchronize());
	printf("%f msec\n", (CNOW - timer) * 0.0001);

	timer = CNOW;

	if (_flip) {
		// flip(ns, nbNs, nbNs2, nodePhases, delta);

		Dvector<REAL> pns(ns.size());
		uint numNodes = ns.size() >> 1u;
		calcPredictPosition_kernel << <divup(numNodes, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
			ns(), n0s(), pns(), delta, numNodes);
		CUDA_CHECK(cudaPeekAtLastError());
		flip(pns, nbNs, nbNs2, nodePhases, delta);

		/*Dvector<REAL> pns;
		pns = ns;
		calcPredictPosition(pns, 50u);
		flip(pns, nbNs, nbNs2, nodePhases, delta);*/
	}

	CUDA_CHECK(cudaDeviceSynchronize());
	printf("%f msec\n", (CNOW - timer) * 0.0001);

	/*for (int i = 0; i < iteration; i++) {
		flip(ns, nbNs, nodePhases);
		collision(ns, invMs, nbNs, delta, 1);
	}*/

	d_fs.copyToHost(h_fs);
	d_es.copyToHost(h_es);
}
void AirMesh::testEdge(vector<REAL>& ns, REAL2 mousePos) {
	vector<uint> h_nbEFs;
	d_nbEFs.copyToHost(h_nbEFs);
	bool flag = true;
	while (flag) {
		_testId++;
		if (_testId > h_es.size() / 2) {
			_testId = 0u;
			flag = false;
		}
		int ino0 = h_es[_testId * 2 + 0];
		int ino1 = h_es[_testId * 2 + 1];
		REAL2 p0 = make_REAL2(ns[ino0 * 2 + 0], ns[ino0 * 2 + 1]);
		REAL2 p1 = make_REAL2(ns[ino1 * 2 + 0], ns[ino1 * 2 + 1]);
		if (Length(mousePos - p0) < 1.0 || Length(mousePos - p1) < 1.0)
			flag = false;
	}
}
void AirMesh::draw(vector<REAL>& ns) {
#if 0
	/*_testtimer++;
	if (_testtimer > 70) {
		_testtimer = 0u;
		_testId++;
		if (_testId >= h_es.size() / 2u)
			_testId = 0u;
	}*/
	if (_testId >= h_es.size() / 2u)
		_testId = 0u;
	vector<uint> h_nbEFs;
	d_nbEFs.copyToHost(h_nbEFs);
	glLineWidth(2.0f);
	glColor3f(1.0f, 0.0f, 0.0f);
	glBegin(GL_LINES);
	int ino0 = h_es[_testId * 2 + 0];
	int ino1 = h_es[_testId * 2 + 1];
	glVertex2f(ns[ino0 * 2 + 0], ns[ino0 * 2 + 1]);
	glVertex2f(ns[ino1 * 2 + 0], ns[ino1 * 2 + 1]);
	glEnd();

	glLineWidth(2.0f);
	glColor3f(0.0f, 1.0f, 0.0f);
	uint i = h_nbEFs[_testId * 2 + 0];
	ino0 = h_fs[i * 3 + 0];
	ino1 = h_fs[i * 3 + 1];
	int ino2 = h_fs[i * 3 + 2];
	glBegin(GL_LINE_LOOP);
	glVertex2f(ns[ino0 * 2 + 0], ns[ino0 * 2 + 1]);
	glVertex2f(ns[ino1 * 2 + 0], ns[ino1 * 2 + 1]);
	glVertex2f(ns[ino2 * 2 + 0], ns[ino2 * 2 + 1]);
	glEnd();

	REAL2 p0 = make_REAL2(ns[ino0 * 2 + 0], ns[ino0 * 2 + 1]);
	REAL2 p1 = make_REAL2(ns[ino1 * 2 + 0], ns[ino1 * 2 + 1]);
	REAL2 p2 = make_REAL2(ns[ino2 * 2 + 0], ns[ino2 * 2 + 1]);
	if (Cross(p1 - p0, p2 - p0) < 0) {
		glColor3f(1.0f, 0.0f, 0.0f);
		glBegin(GL_TRIANGLES);
		glVertex2f(ns[ino0 * 2 + 0], ns[ino0 * 2 + 1]);
		glVertex2f(ns[ino1 * 2 + 0], ns[ino1 * 2 + 1]);
		glVertex2f(ns[ino2 * 2 + 0], ns[ino2 * 2 + 1]);
		glEnd();
	}
	i = h_nbEFs[_testId * 2 + 1];
	if (i != 0xffffffff) {
		glLineWidth(2.0f);
		glColor3f(0.0f, 1.0f, 0.0f);
		ino0 = h_fs[i * 3 + 0];
		ino1 = h_fs[i * 3 + 1];
		ino2 = h_fs[i * 3 + 2];
		glBegin(GL_LINE_LOOP);
		glVertex2f(ns[ino0 * 2 + 0], ns[ino0 * 2 + 1]);
		glVertex2f(ns[ino1 * 2 + 0], ns[ino1 * 2 + 1]);
		glVertex2f(ns[ino2 * 2 + 0], ns[ino2 * 2 + 1]);
		glEnd();

		REAL2 p0 = make_REAL2(ns[ino0 * 2 + 0], ns[ino0 * 2 + 1]);
		REAL2 p1 = make_REAL2(ns[ino1 * 2 + 0], ns[ino1 * 2 + 1]);
		REAL2 p2 = make_REAL2(ns[ino2 * 2 + 0], ns[ino2 * 2 + 1]);
		if (Cross(p1 - p0, p2 - p0) < 0) {
			glColor3f(1.0f, 0.0f, 0.0f);
			glBegin(GL_TRIANGLES);
			glVertex2f(ns[ino0 * 2 + 0], ns[ino0 * 2 + 1]);
			glVertex2f(ns[ino1 * 2 + 0], ns[ino1 * 2 + 1]);
			glVertex2f(ns[ino2 * 2 + 0], ns[ino2 * 2 + 1]);
			glEnd();
		}
	}
#endif

	for (int i = 0; i < h_fs.size() / 3; i++) {
		glLineWidth(1.3f);
		//glColor3f(1.f, 1.f, 1.f);
		glColor3f(0.4f, 0.4f, 0.4f);
		int ino0 = h_fs[i * 3 + 0];
		int ino1 = h_fs[i * 3 + 1];
		int ino2 = h_fs[i * 3 + 2];
		REAL2 p0 = make_REAL2(ns[ino0 * 2 + 0], ns[ino0 * 2 + 1]);
		REAL2 p1 = make_REAL2(ns[ino1 * 2 + 0], ns[ino1 * 2 + 1]);
		REAL2 p2 = make_REAL2(ns[ino2 * 2 + 0], ns[ino2 * 2 + 1]);

		glBegin(GL_LINE_LOOP);
		glVertex2f(ns[ino0 * 2 + 0], ns[ino0 * 2 + 1]);
		glVertex2f(ns[ino1 * 2 + 0], ns[ino1 * 2 + 1]);
		glVertex2f(ns[ino2 * 2 + 0], ns[ino2 * 2 + 1]);
		glEnd();
		if (Cross(p1 - p0, p2 - p0) < 1.0e-5) {
			glColor3f(1.0f, 0.0f, 0.0f);
			glBegin(GL_TRIANGLES);
			glVertex2f(ns[ino0 * 2 + 0], ns[ino0 * 2 + 1]);
			glVertex2f(ns[ino1 * 2 + 0], ns[ino1 * 2 + 1]);
			glVertex2f(ns[ino2 * 2 + 0], ns[ino2 * 2 + 1]);
			glEnd();
		}
	}
}