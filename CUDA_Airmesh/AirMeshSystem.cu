#include "AirMeshSystem.cuh"

void AirMeshSystem::init(REAL width, REAL height) {
	_dt = 0.01;
	_maxIter = 30u;

	_mass = 1.0;
	_invMass = 1.0 / _mass;
	_invdt = 1.0 / _dt;
	_boun._min.x = 3.0;
	_boun._min.y = 3.0;
	_boun._max.x = width - 3.0;
	_boun._max.y = height - 3.0;
	
	_thickness = 0.5;
	_spawnLength = REST_LENGTH;
	_spawnType = OBJ_HAIR_TYPE;
	_spawnColor[0] = 0.5f;
	_spawnColor[1] = 0.5f;
	_spawnColor[2] = 0.8f;
	_spawning = false;

	_moveNodeId = 0xffffffff;

	_drawType = DRAW_MESH | DRAW_AIRMESH;

	initBoundary();
	initNbNs();

	_constraints = new Constraint();
	_constraints->update(h_es, h_cs, h_bes, h_bcs, _numNodes);

	_air = new AirMesh();
	buildAirMesh();
}
void AirMeshSystem::updateNums(void) {
	_numEdges = d_es.size() >> 1u;
	_numNodes = d_ns.size() >> 1u;
}
void AirMeshSystem::initBoundary(void) {
	CUDA_CHECK(cudaDeviceSynchronize());
	ctimer timer = CNOW;

	REAL gap = _spawnLength * 1.0;

	REAL invgap = 1.0 / gap;
	REAL2 box = make_REAL2(_boun._max.x - _boun._min.x, _boun._max.y - _boun._min.y);
	uint wsize = (uint)(box.x * invgap) + 1u;
	uint hsize = (uint)(box.y * invgap) + 1u;
	REAL wgap = box.x / (REAL)(wsize - 1u);
	REAL hgap = box.y / (REAL)(hsize - 1u);
	uint size = wsize + hsize - 2u << 1u;

#if 0
	d_ns.resize(size << 1u);
	d_es.resize(size << 1u);
	initBoundaryVertices_kernel << <divup(max(wsize, hsize - 2u), MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
		d_ns(), _boun, wgap, hgap, wsize, hsize);
	CUDA_CHECK(cudaPeekAtLastError());
	initBoundaryEdges_kernel << <divup(size, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
		d_es(), size);
	CUDA_CHECK(cudaPeekAtLastError());

	d_ns.copyToHost(h_ns);
	d_es.copyToHost(h_es);
#else
	uint ino = 0u;
	h_ns.clear();
	h_es.clear();
	h_cs.clear();
	for (uint i = 0u; i < wsize; i++) {
		h_ns.push_back(_boun._min.x + (REAL)i * wgap);
		h_ns.push_back(_boun._min.y);
		h_es.push_back(ino);
		h_es.push_back(++ino);
		h_cs.push_back(wgap);
		h_cs.push_back(0.0);
	}
	for (uint i = 1u; i < hsize; i++) {
		h_ns.push_back(_boun._max.x);
		h_ns.push_back(_boun._min.y + (REAL)i * hgap);
		h_es.push_back(ino);
		h_es.push_back(++ino);
		h_cs.push_back(hgap);
		h_cs.push_back(0.0);
	}
	for (uint i = 1u; i < wsize; i++) {
		h_ns.push_back(_boun._max.x - (REAL)i * wgap);
		h_ns.push_back(_boun._max.y);
		h_es.push_back(ino);
		h_es.push_back(++ino);
		h_cs.push_back(wgap);
		h_cs.push_back(0.0);
	}
	for (uint i = 1u; i < hsize - 1u; i++) {
		h_ns.push_back(_boun._min.x);
		h_ns.push_back(_boun._max.y - (REAL)i * hgap);
		h_es.push_back(ino);
		h_es.push_back(++ino);
		h_cs.push_back(hgap);
		h_cs.push_back(0.0);
	}
	h_es[h_es.size() - 1u] = h_es[h_es.size() - 2u];
	h_es[h_es.size() - 2u] = 0u;
	
	d_ns = h_ns;
	d_es = h_es;
	d_cs = h_cs;
#endif

	d_nodePhases.resize(size);
	d_nodePhases.memset(0);
	d_nodePhases.copyToHost(h_nodePhases);

	d_vs.resize(size << 1u);
	d_invMs.resize(size);
	d_vs.memset(0);
	d_invMs.memset(0);
	d_vs.copyToHost(h_vs);
	d_invMs.copyToHost(h_invMs);

	Hair* obj = new Hair();
	obj->setColor(0.f, 0.f, 0.f);
	obj->_es = h_es;
	_objs.push_back(obj);

	updateNums();
	_numBounEdges = _numEdges;
	_numBounNodes = _numNodes;

	updateConstraintsInf(0, 0);
	initNbNs();

	CUDA_CHECK(cudaDeviceSynchronize());
	printf("AirMeshSystem::initBoundary: %f msec\n", (CNOW - timer) * 0.0001);
}
void AirMeshSystem::initNbNs(void) {
	d_nbNs._array.resize(_numEdges);
	d_nbNs._index.resize(_numNodes + 1u);
	d_nbNs._index.memset(0);
	reorderIdsUint2_kernel << < divup(_numEdges, MAX_BLOCKSIZE), MAX_BLOCKSIZE,
		(MAX_BLOCKSIZE + 1u) * sizeof(uint) >> > (
			(uint2*)d_es(), d_nbNs._index(), _numEdges);
	CUDA_CHECK(cudaPeekAtLastError());

	getNbNs_kernel << <divup(_numEdges, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
		d_es(), d_nbNs._array(), _numEdges);
	CUDA_CHECK(cudaPeekAtLastError());

	d_nbNs.copyToHost(h_nbNs);

	{
		uint numEdge2 = _numEdges << 1u;
		Dvector<uint> buffer(numEdge2);
		getNbNs2Buffer_kernel << <divup(_numEdges, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
			d_es(), buffer(), _numEdges);
		CUDA_CHECK(cudaPeekAtLastError());

		thrust::sort(thrust::device_ptr<uint2>((uint2*)buffer.begin()),
			thrust::device_ptr<uint2>((uint2*)buffer.end()), uint2_CMP());

		d_nbNs2._array.resize(numEdge2);
		d_nbNs2._index.resize(_numNodes + 1u);
		d_nbNs2._index.memset(0);
		reorderIdsUint2_kernel << < divup(numEdge2, MAX_BLOCKSIZE), MAX_BLOCKSIZE,
			(MAX_BLOCKSIZE + 1u) * sizeof(uint) >> > (
				(uint2*)buffer(), d_nbNs2._index(), numEdge2);
		CUDA_CHECK(cudaPeekAtLastError());

		getNbNs_kernel << <divup(numEdge2, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
			buffer(), d_nbNs2._array(), numEdge2);
		CUDA_CHECK(cudaPeekAtLastError());
	}
}
void AirMeshSystem::buildAirMesh(void) {
	vector<REAL> holes;
	/*for (auto obj: _objs) {
		if (obj->_type == OBJ_RIGID_TYPE)
			((Rigid*)obj)->getHole(holes, h_ns);
	}*/
	_air->init(d_es, d_ns, d_nodePhases, holes);
}
void AirMeshSystem::updateConstraintsInf(REAL stretchMaterial, REAL bendMaterial) {
	uint newSize = d_es.size() >> 1u;
	uint oldSize = d_cs.size() >> 1u;
	Dvector<REAL> newCs;
	if (newSize) {
		newCs.resize(d_es.size());
		compConstraintsInf_kernel << <divup(newSize, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
			d_es(), d_ns(), d_cs(), newCs(), stretchMaterial, oldSize, newSize);
		CUDA_CHECK(cudaPeekAtLastError());
		d_cs.overCopy(newCs);

		thrust::sort_by_key(thrust::device_ptr<uint2>((uint2*)d_es.begin()),
			thrust::device_ptr<uint2>((uint2*)d_es.end()),
			thrust::device_ptr<REAL2>((REAL2*)(d_cs.begin())), uint2_CMP());

		d_es.copyToHost(h_es);
		d_cs.copyToHost(h_cs);
	}
	newSize = d_bes.size() >> 1u;
	oldSize = d_bcs.size() >> 1u;
	if (newSize) {
		newCs.resize(d_bes.size());
		compConstraintsInf_kernel << <divup(newSize, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
			d_bes(), d_ns(), d_bcs(), newCs(), bendMaterial, oldSize, newSize);
		CUDA_CHECK(cudaPeekAtLastError());
		d_bcs.overCopy(newCs);

		thrust::sort_by_key(thrust::device_ptr<uint2>((uint2*)d_bes.begin()),
			thrust::device_ptr<uint2>((uint2*)d_bes.end()),
			thrust::device_ptr<REAL2>((REAL2*)(d_bcs.begin())), uint2_CMP());

		d_bes.copyToHost(h_bes);
		d_bcs.copyToHost(h_bcs);
	}
}
void AirMeshSystem::freeObjs(void) {
	while (_objs.size()) {
		auto obj = _objs.back();
		_objs.pop_back();
		delete obj;
	}
}
void AirMeshSystem::free(void) {
	freeObjs();
	delete _air;
	delete _constraints;
}
void AirMeshSystem::reset(void) {
	d_es.clear();
	d_cs.clear();
	d_bes.clear();
	d_bcs.clear();
	d_ns.clear();
	d_n0s.clear();
	d_vs.clear();
	d_invMs.clear();
	d_nbNs.clear();
	d_nbNs2.clear();
	d_nodePhases.clear();

	h_es.clear();
	h_cs.clear();
	h_bes.clear();
	h_bcs.clear();
	h_ns.clear();
	h_vs.clear();
	h_invMs.clear();
	h_nbNs.clear();
	h_nodePhases.clear();

	freeObjs();

	_constraints->clear();

	initBoundary();
	buildAirMesh();
	_air->clear();
}

void AirMeshSystem::changeSpawnType(void) {
	if (_spawnType == OBJ_HAIR_TYPE) {
		printf("Spawn Type: Rigid\n");
		_spawnType = OBJ_RIGID_TYPE;
	}
	else {
		printf("Spawn Type: Hair\n");
		_spawnType = OBJ_HAIR_TYPE;
	}
}
void AirMeshSystem::spawn(REAL x, REAL y) {
	if (_spawning) {
		if ((_spawnType == OBJ_HAIR_TYPE && _spawnPoints.size() > 6u) ||
			(_spawnType == OBJ_RIGID_TYPE && _spawnPoints.size() > 8u)) {
			_spawnPoints.pop_back();
			_spawnPoints.pop_back();
			_spawnEdges.pop_back();
			_spawnEdges.pop_back();
			Object* obj = nullptr;
			REAL material;
			if (_spawnType == OBJ_HAIR_TYPE) {
				obj = new Hair();
				material = 0.00000016 * _invdt * _invdt;
				//springs = _spawnEdges;
			}
			else {
				obj = new Rigid();

				_spawnEdges.push_back(_spawnEdges[0]);
				_spawnEdges.push_back(_spawnEdges[_spawnEdges.size() - 2u]);
				material = 0.0;

				/*springs = _spawnEdges;
				Dvector<uint> d_faceBuffer, d_springBuffer;
				d_springBuffer = springs;
				_air->getRigidSpring(d_faceBuffer, d_springBuffer, d_ns);
				d_faceBuffer.copyToHost(((Rigid*)obj)->_fs);
				d_springBuffer.copyToHost(springs);
				d_faceBuffer.clear();
				d_springBuffer.clear();*/
			}
			{
				h_ns.insert(h_ns.end(), _spawnPoints.begin(), _spawnPoints.end());
				d_ns = h_ns;

				obj->setColor(0.f, 0.f, 1.f);// 0.3f, 0.3f, 0.9f

				vector<uint> spawnBendingEdges;
				obj->build(_spawnEdges, spawnBendingEdges, d_ns);

				h_es.insert(h_es.end(), _spawnEdges.begin(), _spawnEdges.end());
				h_bes.insert(h_bes.end(), spawnBendingEdges.begin(), spawnBendingEdges.end());
				h_vs.resize(h_ns.size(), 0);
				h_invMs.insert(h_invMs.end(), _spawnInvMs.begin(), _spawnInvMs.end());
				h_nodePhases.resize(h_ns.size() >> 1u, _objs.size());

				d_es = h_es;
				d_bes = h_bes;
				d_vs = h_vs;
				d_invMs = h_invMs;
				d_nodePhases = h_nodePhases;

				updateNums();
				updateConstraintsInf(material, material);
				initNbNs();
				_constraints->update(h_es, h_cs, h_bes, h_bcs, _numNodes);

				_objs.push_back(obj);
				buildAirMesh();
			}
		}

		_spawnPoints.clear();
		_spawnEdges.clear();
		_spawnInvMs.clear();
		_spawning = false;
	}
	else {
		_spawnPoints.push_back(x);
		_spawnPoints.push_back(y);
		_spawnPoints.push_back(x);
		_spawnPoints.push_back(y);
		_spawnEdges.push_back(_numNodes);
		_spawnEdges.push_back(_spawnEdges.back() + 1u);
		_spawnInvMs.push_back(_invMass);
		_spawning = true;
	}
}
void AirMeshSystem::spawnMove(REAL x, REAL y) {
	while (_spawning) {
		REAL2 prev = make_REAL2(_spawnPoints[_spawnPoints.size() - 4u], _spawnPoints[_spawnPoints.size() - 3u]);
		REAL2 curr = make_REAL2(x, y);
		_spawnPoints[_spawnPoints.size() - 2u] = x;
		_spawnPoints[_spawnPoints.size() - 1u] = y;

		REAL2 dir = curr - prev;
		REAL dist = Length(dir);
		if (dist < _spawnLength)
			break;
		dir *= _spawnLength / dist;
		curr = prev + dir;
		_spawnPoints[_spawnPoints.size() - 2u] = curr.x;
		_spawnPoints[_spawnPoints.size() - 1u] = curr.y;
		_spawnPoints.push_back(x);
		_spawnPoints.push_back(y);
		_spawnEdges.push_back(_spawnEdges.back());
		_spawnEdges.push_back(_spawnEdges.back() + 1u);
		_spawnInvMs.push_back(_invMass);
	}
}

void AirMeshSystem::clickNode(REAL2 point) {
	if (_moveNodeId != 0xffffffff) {
		//h_invMs[_moveNodeId] = 1.0;
		//d_invMs = h_invMs;
		_moveNodeId = 0xffffffff;
		return;
	}
	REAL mindist = 5.0;
	uint minIndex = 0xffffffff;
	for (uint i = 0u; i < _numNodes; i++) {
		REAL2 p = make_REAL2(h_ns[i * 2 + 0], h_ns[i * 2 + 1]);

		REAL dist = LengthSquared(p - point);
		if (dist < mindist) {
			mindist = dist;
			minIndex = i;
		}
	}
	if (minIndex != 0xffffffff) {
		_moveNodeId = minIndex;
		_movePos = point;
	}
}
void AirMeshSystem::updateMovePos(REAL2 point){
	if (_moveNodeId == 0xffffffff)
		return;
	_movePos = point;
}
void AirMeshSystem::moveNode(void) {
	if (_moveNodeId == 0xffffffff || _moveNodeId >= _numNodes) {
		_moveNodeId = 0xffffffff;
		return;
	}
	if (h_invMs[_moveNodeId]) {
		REAL2 p;
		p.x = h_ns[_moveNodeId * 2 + 0];
		p.y = h_ns[_moveNodeId * 2 + 1];
		p = (_movePos - p) * _invdt;
		h_vs[_moveNodeId * 2 + 0] = p.x * 0.4;
		h_vs[_moveNodeId * 2 + 1] = p.y * 0.4;
		d_vs = h_vs;
	}
}
void AirMeshSystem::lockNode(void) {
	if (_moveNodeId == 0xffffffff || _moveNodeId >= _numNodes) {
		_moveNodeId = 0xffffffff;
		return;
	}
	if (h_invMs[_moveNodeId]) {
		h_invMs[_moveNodeId] = 0.0;
		h_vs[_moveNodeId * 2 + 0] = 0.0;
		h_vs[_moveNodeId * 2 + 1] = 0.0;
		d_vs = h_vs;
	}
	else 
		h_invMs[_moveNodeId] = _invMass;

	d_invMs = h_invMs;
}

void AirMeshSystem::longHair(void) {
	if (!_thickness)
		return;
	REAL gap = _thickness * 3.0;
	REAL xs[2] = { _boun._min.x + gap,_boun._max.x - gap };
	REAL bottom = _boun._min.y + gap;
	REAL top = _boun._max.y - gap;
	REAL y = bottom;
	uint ix = 0u;

	spawn(xs[ix], y);
	ix ^= 1u;
	y += gap;
	while (y <= top) {
		spawnMove(xs[ix], y);
		ix ^= 1u;
		y += gap;
	}
	spawn(xs[ix], y);
}

void AirMeshSystem::updateVelocities(void) {
	REAL invdt = 1.0 / _dt;
	updateVelocities_kernel << <divup(_numNodes, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
		d_n0s(), d_ns(), d_vs(), invdt, _numNodes);
	CUDA_CHECK(cudaPeekAtLastError());
}
void AirMeshSystem::update(void) {
	REAL2 force = make_REAL2(0.0, GRAVITY * _mass * _invdt);

	compPredictPosition_kernel << <divup(_numNodes, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
		d_ns(), d_vs(), d_invMs(), force, _dt, _numNodes);
	CUDA_CHECK(cudaPeekAtLastError());

	_constraints->d_lambdas.memset(0);
	for (uint itr = 0u; itr < _maxIter; itr++)
		_constraints->projectXPBD(d_ns, d_invMs);

	_air->resolveCollision(d_ns, d_n0s, d_invMs, d_cs, d_nbNs, d_nbNs2, d_nodePhases, _thickness, 0.2, 100u);
}

void AirMeshSystem::simulation(void) {
	d_n0s = d_ns;

	//cudaDeviceSynchronize();
	//ctimer timer = CNOW;

	update();

	//cudaDeviceSynchronize();
	//printf("update %f msec\n", (CNOW - timer) * 0.0001);


	updateVelocities();

	d_ns.copyToHost(h_ns);
	d_vs.copyToHost(h_vs);

	moveNode();
}

void AirMeshSystem::draw(void) {
	drawMove();
	drawSpawn();
	//drawSprings();
	if (_drawType & DRAW_MESH)
		drawObjects();

	drawBoundary();
	if (_drawType & DRAW_AIRMESH)
		_air->draw(h_ns);
}

void AirMeshSystem::drawBoundary(void) {
	glLineWidth(_thickness * 7.5f);
	glColor3f(0.f, 0.f, 0.f);

	glBegin(GL_LINE_LOOP);
	glVertex2f(_boun._min.x, _boun._min.y);
	glVertex2f(_boun._min.x, _boun._max.y);
	glVertex2f(_boun._max.x, _boun._max.y);
	glVertex2f(_boun._max.x, _boun._min.y);
	glEnd();
}
void AirMeshSystem::drawSpawn(void) {
	if (!_spawning)
		return;
	glLineWidth(_thickness * 7.5f);
	glColor3fv(_spawnColor);
	glBegin(GL_LINES);
	for (uint i = 0u; i < _spawnEdges.size(); i += 2u) {
		uint ino0 = (_spawnEdges[i + 0u] - (h_ns.size() >> 1u)) << 1u;
		uint ino1 = (_spawnEdges[i + 1u] - (h_ns.size() >> 1u)) << 1u;
		glVertex2f(_spawnPoints[ino0 + 0u], _spawnPoints[ino0 + 1u]);
		glVertex2f(_spawnPoints[ino1 + 0u], _spawnPoints[ino1 + 1u]);
	}
	glEnd();
	for (uint i = 0u; i < _spawnPoints.size(); i += 2u)
		drawNode(make_REAL2(_spawnPoints[i + 0u], _spawnPoints[i + 1u]), _thickness);
}
void AirMeshSystem::drawMove(void) {
	if (_moveNodeId == 0xffffffff)
		return;

	glColor3f(1.f, 0.f, 0.f);
	REAL2 p = make_REAL2(h_ns[_moveNodeId * 2 + 0], h_ns[_moveNodeId * 2 + 1]);
	//drawNode(p, _thickness);
	glPointSize(_thickness * 18.5f);
	glBegin(GL_POINTS);
	glVertex2f(p.x, p.y);
	glEnd();
}
void AirMeshSystem::drawObjects(void) {
	glLineWidth(_thickness * 7.5f);
	for (int i = 1; i < _objs.size(); i++) {
		auto obj = _objs[i];
		obj->draw(h_ns);
	}
}
void AirMeshSystem::drawSprings(void) {
	glLineWidth(_thickness * 7.5f);
	glColor3f(1.f, 0.f, 0.f);
	glBegin(GL_LINES);
	for (uint i = 0u; i < _constraints->h_ids.size(); i += 2u) {
		uint ino0 = _constraints->h_ids[i + 0u] << 1u;
		uint ino1 = _constraints->h_ids[i + 1u] << 1u;
		glVertex2f(h_ns[ino0 + 0u], h_ns[ino0 + 1u]);
		glVertex2f(h_ns[ino1 + 0u], h_ns[ino1 + 1u]);
	}
	glEnd();
}