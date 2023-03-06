#include "GraphColoring.cuh"

namespace GraphColoring {
	void getNbs(
		Dvector<uint>& links, uint numLink, DPrefixArray<uint2>& nbXs, uint* d_tmp, uint numNodes)
	{
		uint numNbXs = links.size();
		nbXs._array.resize(numNbXs);

		nbXs._index.resize(numNodes + 1u);
		nbXs._index.memset(0u);

		CUDA_CHECK(cudaMemset(d_tmp, 0, sizeof(uint)));
		getNbXs_kernel << < divup(numNbXs, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
			links(), nbXs._array(), numLink, numNbXs, d_tmp);
		CUDA_CHECK(cudaPeekAtLastError());

		thrust::sort(thrust::device_ptr<uint2>(nbXs._array.begin()),
			thrust::device_ptr<uint2>(nbXs._array.end()), uint2_CMP());

		reorderIdsUint2_kernel << < divup(numNbXs, MAX_BLOCKSIZE), MAX_BLOCKSIZE,
			(MAX_BLOCKSIZE + 1u) * sizeof(uint)>> > (
			nbXs._array(), nbXs._index(), numNbXs);
		CUDA_CHECK(cudaPeekAtLastError());
	}
	void sequential(
		Dvector<uint>& links, uint numLink, Dvector<uint>& seqs, uint& seqSize, uint numNodes)
	{
		DPrefixArray<uint2> nbXs;
		uint* d_tmp;
		CUDA_CHECK(cudaMalloc((void**)&d_tmp, sizeof(uint)));
		getNbs(links, numLink, nbXs, d_tmp, numNodes);

		uint numLinks = links.size() / numLink;
		Dvector<uint> icurrs(numNodes);
		Dvector<uint> isEnds(numLinks);
		seqs.resize(numLinks);
		isEnds.memset(0);

		uint isApplied;
		uint seq = 0u;

		initDepth_kernel << < divup(numNodes, BLOCKSIZE), BLOCKSIZE >> > (
			nbXs._array(), nbXs._index(), icurrs(), isEnds(), numNodes);
		CUDA_CHECK(cudaPeekAtLastError());

		/*vector<uint> test;
		links.copyToHost(test);
		for (auto t : test)printf("%d\n", t);*/

		do {
			getSequence_kernel << < divup(numLinks, BLOCKSIZE), BLOCKSIZE >> > (
				seqs(), isEnds(), numLink, numLinks, seq++);
			CUDA_CHECK(cudaPeekAtLastError());

			CUDA_CHECK(cudaMemset(d_tmp, 0, sizeof(uint)));
			nextDepth_kernel << < divup(numNodes, BLOCKSIZE), BLOCKSIZE >> > (
				nbXs._array(), nbXs._index(), icurrs(), isEnds(), numNodes, d_tmp);
			CUDA_CHECK(cudaPeekAtLastError());
			CUDA_CHECK(cudaMemcpy(&isApplied, d_tmp, sizeof(uint), cudaMemcpyDeviceToHost));
		} while (isApplied);

		seqSize = seq;

		CUDA_CHECK(cudaFree(d_tmp));
	}

	/*void getNeis(
		Dvector<uint>& links, uint numLink, DPrefixArray<uint2>& neis, uint* d_tmp, uint numNodes)
	{
		DPrefixArray<uint2> nbXs;
		getNbs(links, numLink, nbXs, d_tmp, numNodes);

		uint numLinks = links.size() / numLink;

		CUDA_CHECK(cudaMemset(d_tmp, 0, sizeof(uint)));
		getNumNeis_kernel << <divup(numNodes, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
			nbXs._array(), nbXs._index(), numNodes, d_tmp);
		CUDA_CHECK(cudaPeekAtLastError());

		uint numNeis;
		CUDA_CHECK(cudaMemcpy(&numNeis, d_tmp, sizeof(uint), cudaMemcpyDeviceToHost));
		neis._array.resize(numNeis);

		CUDA_CHECK(cudaMemset(d_tmp, 0, sizeof(uint)));
		getNeis_kernel << <divup(numNodes, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
			nbXs._array(), nbXs._index(), neis._array(), numNodes, d_tmp);
		CUDA_CHECK(cudaPeekAtLastError());

		nbXs._array.clear();
		nbXs._index.clear();

		thrust::sort(thrust::device_ptr<uint2>(neis._array.begin()),
			thrust::device_ptr<uint2>(neis._array.end()), uint2_CMP());

		neis._index.resize(numLinks + 1u);
		neis._index.memset(0xffffffff);

		reorderIds_kernel << < divup(numNeis, MAX_BLOCKSIZE), MAX_BLOCKSIZE,
			(MAX_BLOCKSIZE + 1u) * sizeof(uint) >> > (
				neis._array(), neis._index(), numNeis);
		CUDA_CHECK(cudaPeekAtLastError());
	}
	void coloring(
		Dvector<uint>& links, uint numLink, Dvector<uint>& colors, uint& colorSize, uint numNodes)
	{
		DPrefixArray<uint2> neis;
		uint* d_tmp;
		CUDA_CHECK(cudaMalloc((void**)&d_tmp, sizeof(uint)));
		getNeis(links, numLink, neis, d_tmp, numNodes);

		uint numLinks = links.size() / numLink;
		Dvector<uint> icurrs(numLinks);
		colors.resize(numLinks);
		colors.memset(0xffffffff);
		icurrs.memset(0);

		uint flag = 1u;
		while (flag) {
			CUDA_CHECK(cudaMemset(d_tmp, 0, sizeof(uint)));
			compColoring_kernel << <divup(numLinks, BLOCKSIZE), BLOCKSIZE >> > (
				neis._array(), neis._index(), icurrs(), colors(), numLinks, d_tmp);
			CUDA_CHECK(cudaPeekAtLastError());
			CUDA_CHECK(cudaMemcpy(&flag, d_tmp, sizeof(uint), cudaMemcpyDeviceToHost));
		}

		{
			CUDA_CHECK(cudaMemset(d_tmp, 0, sizeof(uint)));
			getDvectorMax(colors, d_tmp);
			CUDA_CHECK(cudaMemcpy(&colorSize, d_tmp, sizeof(uint), cudaMemcpyDeviceToHost));
			colorSize++;
		}

		CUDA_CHECK(cudaFree(d_tmp));
	}*/
	void coloring(
		Dvector<uint>& links, uint numLink, Dvector<uint>& colors, uint& colorSize, uint numNodes)
	{
		DPrefixArray<uint2> nbXs;
		uint* d_tmp;
		CUDA_CHECK(cudaMalloc((void**)&d_tmp, sizeof(uint)));

		getNbs(links, numLink, nbXs, d_tmp, numNodes);

		CUDA_CHECK(cudaMemset(d_tmp, 0, sizeof(uint)));
		getMaxNeis_kernel << <divup(numNodes, MAX_BLOCKSIZE), MAX_BLOCKSIZE,
			MAX_BLOCKSIZE * sizeof(uint) >> > (
				nbXs._array(), nbXs._index(), numNodes, d_tmp);
		CUDA_CHECK(cudaPeekAtLastError());

		CUDA_CHECK(cudaMemcpy(&colorSize, d_tmp, sizeof(uint), cudaMemcpyDeviceToHost));
		colorSize++;

		uint numLinks = links.size() / numLink;
		Dvector<uint> icurrs(numNodes);
		Dvector<uint> isEnds(numLinks);
		Dvector<bool> colorBuffer(numNodes * colorSize);
		colors.resize(numLinks);
		isEnds.memset(0);
		colorBuffer.memset(0);

		uint isApplied;

		initDepth_kernel << < divup(numNodes, BLOCKSIZE), BLOCKSIZE >> > (
			nbXs._array(), nbXs._index(), icurrs(), isEnds(), numNodes);
		CUDA_CHECK(cudaPeekAtLastError());
		do {
			getColor_kernel << < divup(numLinks, BLOCKSIZE), BLOCKSIZE >> > (
				links(), isEnds(), colors(), colorBuffer(), colorSize, numLink, numLinks);
			CUDA_CHECK(cudaPeekAtLastError());

			CUDA_CHECK(cudaMemset(d_tmp, 0, sizeof(uint)));
			nextDepth_kernel << < divup(numNodes, BLOCKSIZE), BLOCKSIZE >> > (
				nbXs._array(), nbXs._index(), icurrs(), isEnds(), numNodes, d_tmp);
			CUDA_CHECK(cudaPeekAtLastError());
			CUDA_CHECK(cudaMemcpy(&isApplied, d_tmp, sizeof(uint), cudaMemcpyDeviceToHost));
		} while (isApplied);

		{
			CUDA_CHECK(cudaMemset(d_tmp, 0, sizeof(uint)));
			getDvectorMax(colors, d_tmp);
			CUDA_CHECK(cudaMemcpy(&colorSize, d_tmp, sizeof(uint), cudaMemcpyDeviceToHost));
			colorSize++;
		}

		CUDA_CHECK(cudaFree(d_tmp));
	}
};