#ifndef __GRAPH_COLORING_H__
#define __GRAPH_COLORING_H__

#pragma once
#include "DeviceManager.h"
#include "PrefixArray.h"
#include "../GL/freeglut.h"

namespace GraphColoring {
	void getNbs(
		Dvector<uint>& links, uint numLink, DPrefixArray<uint2>& nbXs, uint* d_tmp, uint numNodes);
	void sequential(
		Dvector<uint>& links, uint numLink, Dvector<uint>& seqs, uint& seqSize, uint numNodes);
	void getNeis(
		Dvector<uint>& links, uint numLink, DPrefixArray<uint2>& neis, uint* d_tmp, uint numNodes);
	void coloring(
		Dvector<uint>& links, uint numLink, Dvector<uint>& colors, uint& colorSize, uint numNodes);
};

#endif