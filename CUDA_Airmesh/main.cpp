#include <Windows.h>
#include <stdio.h>

#include "AirMeshSystem.h"

double _width = 1200;
double _height = 800;
int _frame = 0;

unsigned char _buttons[3] = { 0 };
bool _simulation = false;
char _FPS_str[100];

AirMeshSystem* _system = nullptr;

#define SCREEN_CAPTURE

void DrawText(float x, float y, const char* text, void* font = NULL)
{
	glColor3f(0, 0, 0);
	glDisable(GL_DEPTH_TEST);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0.0, (double)_width, 0.0, (double)_height, -1.0, 1.0);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	if (font == NULL) {
		font = GLUT_BITMAP_9_BY_15;
	}

	size_t len = strlen(text);

	glRasterPos2f(x, y);
	for (const char* letter = text; letter < text + len; letter++) {
		if (*letter == '\n') {
			y -= 12.0f;
			glRasterPos2f(x, y);
		}
		glutBitmapCharacter(font, *letter);
	}

	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glEnable(GL_DEPTH_TEST);
}
void Init(void)
{
	glEnable(GL_DEPTH_TEST);
	_system = new AirMeshSystem();
	_system->init((REAL)_width * 0.1, (REAL)_height * 0.1);

	_system->longHair();
}
void FPS(void)
{
	static float framesPerSecond = 0.0f;
	static float lastTime = 0.0f;
	float currentTime = GetTickCount() * 0.001f;
	++framesPerSecond;
	if (currentTime - lastTime > 1.0f) {
		lastTime = currentTime;
		sprintf(_FPS_str, "FPS : %d", (int)framesPerSecond);
		framesPerSecond = 0;
	}
}
void Darw(void)
{
	char text[100];

	if(_system)
		_system->draw();

	//DrawText(10.0f, (float)_height - 20.f, "Air Mesh");

	//if (_system)
	//	sprintf(text, "Number of triangles : %d", _cloth_mesh->_fs.size() / 3u);

	//DrawText(10.0f, (float)_height - 40.f, text);

	/*DrawText(10.0f, (float)_height - 60.f, _FPS_str);
	sprintf(text, "Frame : %d", _frame);
	DrawText(10.0f, (float)_height - 80.f, text);*/
}
void Capture(int endFrame)
{
	if (_frame == 0 || _frame % 2 == 0) {
		static int index = 0;
		char filename[100];
		sprintf_s(filename, "capture\\capture-%d.bmp", index);
		BITMAPFILEHEADER bf;
		BITMAPINFOHEADER bi;
		unsigned char* image = (unsigned char*)malloc(sizeof(unsigned char) * _width * _height * 3);
		FILE* file;
		fopen_s(&file, filename, "wb");
		if (image != NULL) {
			if (file != NULL) {
				glReadPixels(0, 0, _width, _height, 0x80E0, GL_UNSIGNED_BYTE, image);
				memset(&bf, 0, sizeof(bf));
				memset(&bi, 0, sizeof(bi));
				bf.bfType = 'MB';
				bf.bfSize = sizeof(bf) + sizeof(bi) + _width * _height * 3;
				bf.bfOffBits = sizeof(bf) + sizeof(bi);
				bi.biSize = sizeof(bi);
				bi.biWidth = _width;
				bi.biHeight = _height;
				bi.biPlanes = 1;
				bi.biBitCount = 24;
				bi.biSizeImage = _width * _height * 3;
				fwrite(&bf, sizeof(bf), 1, file);
				fwrite(&bi, sizeof(bi), 1, file);
				fwrite(image, sizeof(unsigned char), _height * _width * 3, file);
				fclose(file);
			}
			free(image);
		}
		index++;
		if (index == endFrame) {
			exit(0);
		}
	}
}

void Update(void)
{
	if (_simulation) {
#ifdef SCREEN_CAPTURE
		Capture(1000);
#endif
		if (_system)
			_system->simulation();
		if (_frame == 1000) {
			//	exit(0);
		}
		_frame++;
	}
	::glutPostRedisplay();
}

#define QUALITY_TEST	0
void Display(void)
{
	//glClearColor(0.65, 0.65, 0.65, 0.8f);
	glClearColor(1.f, 1.f, 1.f, 1.f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_POLYGON_OFFSET_FILL);
	glPolygonOffset(1.1f, 4.0f);

#if QUALITY_TEST == 0
	Darw();
#else 
	{
		REAL offset = 10.0;
		REAL2 p0 = make_REAL2(50. / offset, 40. / offset);
		REAL2 p1 = make_REAL2(70. / offset, 40. / offset);
		glColor3f(1.f, 1.f, 1.f);
		glLineWidth(5.f);
		glBegin(GL_LINES);
		glColor3f(1.f, 1.f, 1.f);
		glVertex2f(p0.x * offset, p0.y * offset);
		glVertex2f(p1.x * offset, p1.y * offset);
		glEnd();
		glPointSize(10);

		REAL maxQ = -DBL_MAX;
		REAL minQ = DBL_MAX;
		for (int i = 0; i < _width / 10; i++) {
			for (int j = 0; j < _height / 10; j++) {
				REAL2 v = make_REAL2((REAL)i / offset, (REAL)j / offset);
				REAL2 dirs[3];
				dirs[0] = p1 - p0;
				dirs[1] = v - p0;
				dirs[2] = v - p1;

				REAL invLs[3];
				invLs[0] = Length(dirs[0]);
				invLs[1] = Length(dirs[1]);
				invLs[2] = Length(dirs[2]);
				if (invLs[0]) invLs[0] = 1.0 / invLs[0];
				if (invLs[1]) invLs[1] = 1.0 / invLs[1];
				if (invLs[2]) invLs[2] = 1.0 / invLs[2];
				dirs[0] *= invLs[0];
				dirs[1] *= invLs[1];
				dirs[2] *= invLs[2];
				
				REAL quality;
				//quality = Dot(dirs[1], dirs[2]); // -1~1
				//quality = Dot(dirs[1], dirs[2]) + Dot(dirs[0], dirs[1]); // 0~2
				//quality = Dot(dirs[1], dirs[2]) - Dot(dirs[0], dirs[2]); //  0~2
				//quality = min(Dot(dirs[1], dirs[2]) + Dot(dirs[0], dirs[1]), Dot(dirs[1], dirs[2]) - Dot(dirs[0], dirs[2])); // 0~2
				//quality = Dot(dirs[1], dirs[2]) + Dot(dirs[0], dirs[1]) - Dot(dirs[0], dirs[2]); // 0~2
				//quality = Dot(dirs[0], dirs[1]) - Dot(dirs[0], dirs[2]); // 0~2
				//quality = Dot(v - p0, v - p1) / max(Length(v - p0), Length(v - p1));
				quality = Cross(dirs[2], dirs[1]) * min(Length(v - p0), Length(v - p1)) * (Dot(dirs[1], dirs[2]) + 1.0);
				if (maxQ < quality) maxQ = quality;
				if (minQ > quality) minQ = quality;

				quality += 3.652301;
				quality /= 3.652301 * 2;
				if (quality < 0.) quality = 0.;
				if (quality > 1.) quality = 1.;

				float r, g, b;
				r = g = 0.0;
				b = quality;
				/*b = quality * 2.f;
				g = (quality - 0.5) * 2.5f;
				r = (quality - 0.9) * 10.f;*/
				/*b = quality * 3.f;
				g = (quality - 1.f / 3.f) * 3.f;
				r = (quality - 2.f / 3.f) * 3.f;*/
				if (r < 0.f) r = 0.f;
				else if (r > 1.f) r = 1.f;
				if (g < 0.f) g = 0.f;
				else if (g > 1.f) g = 1.f;
				if (b < 0.f) b = 0.f;
				else if (b > 1.f) b = 1.f;
				glColor3f(r, max(g - r, 0.0), max(b - g, 0.0));

				if (quality > 1.0-0.00001)
					glColor3f(1.f, 0.f, 0.f);

				glBegin(GL_POINTS);
				glVertex2f(v.x * offset, v.y * offset);
				glEnd();
			}
		}
		printf("%f, %f\n", maxQ, minQ);
	}
#endif

	FPS();

	glutSwapBuffers();
}

void Reshape(int w, int h)
{
	glLoadIdentity();
	glViewport(0, 0, w, h);
	gluOrtho2D(0.0, (double)_width * 0.1, (double)_height* 0.1, 0.0);
	//glLoadIdentity();
}

void Motion(int x, int y)
{
	_system->spawnMove((REAL)(x + 1) * 0.1, (REAL)(y + 1) * 0.1);
	glutPostRedisplay();
}
void PassiveMotion(int x, int y)
{
	_system->updateMovePos(make_REAL2((REAL)(x + 1) * 0.1, (REAL)(y + 1) * 0.1));
	glutPostRedisplay();
}

void Mouse(int button, int state, int x, int y)
{
	switch (button)
	{
	case GLUT_LEFT_BUTTON:
		if (state == 0) {
			_system->clickNode(make_REAL2((REAL)(x + 1) * 0.1, (REAL)(y + 1) * 0.1));
			_system->spawnMove((REAL)(x + 1) * 0.1, (REAL)(y + 1) * 0.1);
		}
		break;
	case GLUT_MIDDLE_BUTTON:
		break;
	case GLUT_RIGHT_BUTTON:
		break;
	default:
		break;
	}
	glutPostRedisplay();
}

void Keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 'q':
	case 'Q':
		exit(0);
	case ' ':
		_simulation = !_simulation;
		break;
	case 'a':
	case 'A':
		_system->_drawType ^= DRAW_AIRMESH;
		break;
	case 'p':
	case 'P':
		break;
	case 'r':
	case 'R':
		_system->reset();
		break;
	case 'l':
	case 'L':
		_system->lockNode();
		break;
	case 't':
	case 'T':
		_system->changeSpawnType();
		break;
	case 'd':
	case 'D':
		_system->_air->testEdge(_system->h_ns, _system->_movePos);
		break;
	case 'f':
	case 'F':
		/*_system->_air->flip(_system->d_ns, _system->d_nbNs, _system->d_nodePhases, _system->_thickness);
		_system->_air->d_fs.copyToHost(_system->_air->h_fs);
		_system->_air->d_es.copyToHost(_system->_air->h_es);*/
		_system->_air->_flip = !_system->_air->_flip;
		break;
	case 's':
	case 'S':
		_system->spawn((REAL)x * 0.1, (REAL)y * 0.1);
		break;
	case 'm':
	case 'M':
		_system->_drawType ^= DRAW_MESH;
		break;
	case 'c':
	case 'C':
		_system->_air->_collision = !_system->_air->_collision;
		break;
	}
	glutPostRedisplay();
}
int main(int argc, char** argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowSize(_width, _height);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("AirMesh");
	glutDisplayFunc(Display);
	glutReshapeFunc(Reshape);
	glutIdleFunc(Update);
	glutMouseFunc(Mouse);
	glutMotionFunc(Motion);
	glutPassiveMotionFunc(PassiveMotion);
	glutKeyboardFunc(Keyboard);

	Init();

	glutMainLoop();
}