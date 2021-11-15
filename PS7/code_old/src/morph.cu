#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <string>
#include <cmath>
#include <pthread.h>

#include <chrono> // For timing

#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL/glut.h>
#endif

using namespace std;

//the pixel
typedef struct pix{
  unsigned char r,g,b,a;
} pixel;

typedef struct SimplePoint_struct {
        float x, y;

} SimplePoint;


typedef struct SimpleFeatureLine_struct {
      SimplePoint startPoint;
        SimplePoint endPoint;

} SimpleFeatureLine;


template <typename T>
__host__ __device__ T CLAMP(T value, T low, T high)
{
        return (value < low) ? low : ((value > high) ? high : value);
}

#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) {
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


int imgWidthOrig, imgHeightOrig, imgWidthDest, imgHeightDest;
int steps;
float p,a,b,t;
pixel* hSrcImgMap;
pixel* hDstImgMap;

//The name of input and output files
string inputFileSrc;
string inputFileDest;
string inputFileLines;
string outputPath;
string tempFile;
string dataPath;
string stepsStr;
string pStr, aStr, bStr, tStr;

void imgRead(string filename, pixel * &map, int &imgW, int &imgH){
    stbi_set_flip_vertically_on_load(true);

    int x, y, componentsPerPixel;
    if( ! filename.empty()  ){
        map = (pixel *) stbi_load(filename.c_str(), &x, &y, &componentsPerPixel, STBI_rgb_alpha);
    } else{
        cout<<"The input file name cannot be empty"<<endl;
        exit(1);
    }

    // get the current image columns and rows
    imgW = x;
    imgH = y;

    cout<<"Read the image file \""<<filename<<"\" successfully ."<<endl;
}

void imgWrite(string filename, pixel * map, int imgW, int imgH){
    if(filename.empty()){
        cout<<"The output file name cannot be empty"<<endl;
        exit(1);
    }
    stbi_flip_vertically_on_write(true);
    stbi_write_png(filename.c_str(), imgW, imgH, STBI_rgb_alpha, map, sizeof(pixel) * imgW);
    cout<<"Write the image into \""<<filename<<"\" file successfully."<<endl;
}

void loadLines(SimpleFeatureLine** linesSrc, SimpleFeatureLine** linesDst, int* linesLen, const char* name) {
	FILE *f = fopen(name, "r");
	if (f == NULL)
	{
		printf("Error opening file %s! \n", name);
		exit(1);
	}
	fscanf(f, "%d", linesLen);
	SimpleFeatureLine* srcLines = (SimpleFeatureLine*) malloc(sizeof(SimpleFeatureLine)*(*linesLen));
	SimpleFeatureLine* dstLines = (SimpleFeatureLine*) malloc(sizeof(SimpleFeatureLine)*(*linesLen));
	SimpleFeatureLine* line;
	int fac = 2;

	for(int i = 0; i < (*linesLen)*fac; i++) {
		line = (i % fac) ? &dstLines[(i-1)/fac] : &srcLines[i/fac];
		fscanf(f, "%f,%f,%f,%f",
				&(line->startPoint.x), &(line->startPoint.y),
				&(line->endPoint.x), &(line->endPoint.y));
	}

	*linesSrc = srcLines;
	*linesDst = dstLines;
}


// Parse commandline arguments
void parse(int argc, char *argv[]) {
    p = 0;
    a = 1;
    b = 2;
    t = 0.5;
    steps = 90;
    switch(argc){
        case 2:
            dataPath = argv[1];
        case 6:
            inputFileSrc = argv[1];
            inputFileDest = argv[2];
            inputFileLines = argv[3];
            outputPath = argv[4];
            stepsStr = argv[5];
            istringstream ( stepsStr ) >> steps;
            imgRead(inputFileSrc, hSrcImgMap, imgWidthOrig, imgHeightOrig);
            imgRead(inputFileDest, hDstImgMap, imgWidthDest, imgHeightDest);
            break;

        case 9:
            inputFileSrc = argv[1];
            inputFileDest = argv[2];
            inputFileLines = argv[3];
            outputPath = argv[4];
            stepsStr = argv[5];
            pStr = argv[6];
            aStr = argv[7];
            bStr = argv[8];
            istringstream ( stepsStr ) >> steps;
            istringstream ( pStr ) >> p;
            istringstream ( aStr ) >> a;
            istringstream ( bStr ) >> b;
            imgRead(inputFileSrc, hSrcImgMap, imgWidthOrig, imgHeightOrig);
            imgRead(inputFileDest, hDstImgMap, imgWidthDest, imgHeightDest);
            break;

        default:
            cout<<"Usage:"<<endl;
            cout<<"./morph srcImg.png destImg.png lines.txt outputPath steps [p] [a] [b]"<<endl;
            exit(1);
    }

}


void simpleLineInterpolate(SimpleFeatureLine* sourceLines,
                     SimpleFeatureLine* destLines , SimpleFeatureLine** morphLines, int linesLen, float t)
{
	SimpleFeatureLine* interLines = (SimpleFeatureLine*) malloc(sizeof(SimpleFeatureLine)*linesLen);
	for(int i=0; i<linesLen; i++){
		interLines[i].startPoint.x = (1-t)*(sourceLines[i].startPoint.x) + t*(destLines[i].startPoint.x);
		interLines[i].startPoint.y = (1-t)*(sourceLines[i].startPoint.y) + t*(destLines[i].startPoint.y);
		interLines[i].endPoint.x = (1-t)*(sourceLines[i].endPoint.x) + t*(destLines[i].endPoint.x);
		interLines[i].endPoint.y = (1-t)*(sourceLines[i].endPoint.y) + t*(destLines[i].endPoint.y);
	}
	*morphLines = interLines;
}




/* warping function (backward mapping)
   input:
   interPt = the point in the intermediary image
   interLines = given line in the intermediary image
   srcLines = given line in the source image
   p, a, b = parameters of the weight function
   output:
   src = the corresponding point */
__host__ __device__ void warp(const SimplePoint* interPt, SimpleFeatureLine* interLines,
          SimpleFeatureLine* sourceLines, const int sourceLinesSize, SimplePoint* src)
{
	int i;
	float interLength, srcLength;
	float weight, weightSum, dist;
	float sum_x, sum_y; // weighted sum of the coordination of the point "src"
	float u, v;
	SimplePoint pd, pq, qd;
	float X, Y;

	sum_x = 0;
	sum_y = 0;
	weightSum = 0;

	for (i=0; i<sourceLinesSize; i++) {
		pd.x = interPt->x - interLines[i].startPoint.x;
		pd.y = interPt->y - interLines[i].startPoint.y;
		pq.x = interLines[i].endPoint.x - interLines[i].startPoint.x;
		pq.y = interLines[i].endPoint.y - interLines[i].startPoint.y;
		interLength = pq.x * pq.x + pq.y * pq.y;
		u = (pd.x * pq.x + pd.y * pq.y) / interLength;

		interLength = sqrt(interLength); // length of the vector PQ

		v = (pd.x * pq.y - pd.y * pq.x) / interLength;

		pq.x = sourceLines[i].endPoint.x - sourceLines[i].startPoint.x;
		pq.y = sourceLines[i].endPoint.y - sourceLines[i].startPoint.y;

		srcLength = sqrt(pq.x * pq.x + pq.y * pq.y); // length of the vector P'Q'
		// corresponding point based on the ith line
		X = sourceLines[i].startPoint.x + u * pq.x + v * pq.y / srcLength;
		Y = sourceLines[i].startPoint.y + u * pq.y - v * pq.x / srcLength;

		// the distance from the corresponding point to the line P'Q'
		if (u < 0)
			dist = sqrt(pd.x * pd.x + pd.y * pd.y);
		else if (u > 1) {
			qd.x = interPt->x - interLines[i].endPoint.x;
			qd.y = interPt->y - interLines[i].endPoint.y;
			dist = sqrt(qd.x * qd.x + qd.y * qd.y);
		}else{
			dist = abs(v);
		}

		weight = pow(1.0f / (1.0f + dist), 2.0f);
		sum_x += X * weight;
		sum_y += Y * weight;
		weightSum += weight;
	}

	src->x = sum_x / weightSum;
	src->y = sum_y / weightSum;
}

__host__ __device__ void bilinear(pixel* Im, float row, float col, pixel* pix, int dImgWidth)
{
	int cm, cn, fm, fn;
	double alpha, beta;

	cm = (int)ceil(row);
	fm = (int)floor(row);
	cn = (int)ceil(col);
	fn = (int)floor(col);
	alpha = ceil(row) - row;
	beta = ceil(col) - col;

	pix->r = (unsigned int)( alpha*beta*Im[fm*dImgWidth+fn].r
			+ (1-alpha)*beta*Im[cm*dImgWidth+fn].r
			+ alpha*(1-beta)*Im[fm*dImgWidth+cn].r
			+ (1-alpha)*(1-beta)*Im[cm*dImgWidth+cn].r );
	pix->g = (unsigned int)( alpha*beta*Im[fm*dImgWidth+fn].g
			+ (1-alpha)*beta*Im[cm*dImgWidth+fn].g
			+ alpha*(1-beta)*Im[fm*dImgWidth+cn].g
			+ (1-alpha)*(1-beta)*Im[cm*dImgWidth+cn].g );
	pix->b = (unsigned int)( alpha*beta*Im[fm*dImgWidth+fn].b
			+ (1-alpha)*beta*Im[cm*dImgWidth+fn].b
			+ alpha*(1-beta)*Im[fm*dImgWidth+cn].b
			+ (1-alpha)*(1-beta)*Im[cm*dImgWidth+cn].b );
	pix->a = 255;
}

__host__ __device__ void ColorInterPolate(const SimplePoint* Src_P,
                      const SimplePoint* Dest_P, float t,
                      pixel* imgSrc, pixel* imgDest, pixel* rgb, int dImgWidth)
{
    pixel srcColor, destColor;

    bilinear(imgSrc, Src_P->y, Src_P->x, &srcColor, dImgWidth);
    bilinear(imgDest, Dest_P->y, Dest_P->x, &destColor, dImgWidth);

    rgb->b = srcColor.b*(1-t)+ destColor.b*t;
    rgb->g = srcColor.g*(1-t)+ destColor.g*t;
    rgb->r = srcColor.r*(1-t)+ destColor.r*t;
    rgb->a = 255;
}


///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
///// DO NOT TOUCH CODE ABOVE. YOU ONLY NEED TO CHANGE THE FUNCTIONS BELOW ////////
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

void morphKernel(SimpleFeatureLine* dSrcLines, SimpleFeatureLine* dDstLines, SimpleFeatureLine* dMorphLines,
		pixel* dSrcImgMap, pixel* dDstImgMap,  pixel* dMorphMap,
		int linesLen, int dImgWidth, int dImgHeight, float dT) {

	for (int i = 0; i < dImgHeight; i++) {
		for (int j = 0; j < dImgWidth; j++) {
			pixel interColor;
			SimplePoint dest;
			SimplePoint src;
			SimplePoint q;
			q.x = j;
			q.y = i;

			// warping
			warp(&q, dMorphLines, dSrcLines, linesLen, &src);
			warp(&q, dMorphLines, dDstLines, linesLen, &dest);

			src.x = CLAMP<double>(src.x, 0, dImgWidth-1);
			src.y = CLAMP<double>(src.y, 0, dImgHeight-1);
			dest.x = CLAMP<double>(dest.x, 0, dImgWidth-1);
			dest.y = CLAMP<double>(dest.y, 0, dImgHeight-1);

			// color interpolation
			ColorInterPolate(&src, &dest, dT, dSrcImgMap, dDstImgMap, &interColor, dImgWidth);

			dMorphMap[i*dImgWidth+j].r = interColor.r;
			dMorphMap[i*dImgWidth+j].g = interColor.g;
			dMorphMap[i*dImgWidth+j].b = interColor.b;
			dMorphMap[i*dImgWidth+j].a = interColor.a;
		}
	}
}

typedef struct Args_struct {
	int i;
	pixel* hMorphMap;
} Args;

void* pthread_imgWrite(void* args);
float stepSize;

int main(int argc,char *argv[]){

    // Timing (https://stackoverflow.com/questions/12231166/timing-algorithm-clock-vs-time-in-c)
    auto start_time_tot = chrono::high_resolution_clock::now();

	// Setup //////////////////

    // Timing
    auto start_time_load = chrono::high_resolution_clock::now();

	parse(argc, argv);
	tempFile = outputPath;
	stepSize = 1.0/steps;

	int linesLen;
	SimpleFeatureLine *hSrcLines, *hDstLines;
	loadLines(&hSrcLines, &hDstLines, &linesLen, inputFileLines.c_str());
	printf("Loaded %d lines\n", linesLen);

	pixel** hMorphMapArr = (pixel**) malloc(sizeof(pixel*) * (steps+1));
	SimpleFeatureLine** hMorphLinesArr = (SimpleFeatureLine**) malloc(sizeof(SimpleFeatureLine*)*(steps+1));

	for (int i = 0; i < steps+1; i++) {
		hMorphMapArr[i] = (pixel*) malloc(sizeof(pixel)*imgHeightOrig*imgWidthOrig);
		simpleLineInterpolate(hSrcLines, hDstLines, &(hMorphLinesArr[i]), linesLen, t);
	}

    // Timing
    auto stop_time_load = chrono::high_resolution_clock::now();
    printf("Time spent on loading images & lines: %ld ms\n", chrono::duration_cast<chrono::milliseconds>(stop_time_load-start_time_load).count() );

	///////////////////////////



    // Image morphing /////////

    // Timing
    auto start_time_morph = chrono::high_resolution_clock::now();

	int dImgWidth = imgHeightOrig; // 1024
	int dImgHeight = imgWidthOrig; // 1024

	// Computes a morphed image for each step based on hMorphLinesArr[i].
	// The morphed image is saved in hMorphMapArr[i];
	for (int i = 0; i < steps+1; i++) {
		t = stepSize*i;
		SimpleFeatureLine* hMorphLines = hMorphLinesArr[i];
		pixel* hMorphMap = hMorphMapArr[i];
		float dT = t;

		// Delete these lines and replace with CUDA variables
		SimpleFeatureLine* dSrcLines = hSrcLines;
		SimpleFeatureLine* dDstLines = hDstLines;
		SimpleFeatureLine* dMorphLines = hMorphLines;
		pixel* dSrcImgMap = hSrcImgMap;
		pixel* dDstImgMap = hDstImgMap;
		pixel* dMorphMap = hMorphMap;

		morphKernel(dSrcLines, dDstLines, dMorphLines, dSrcImgMap, dDstImgMap, dMorphMap, linesLen, dImgWidth, dImgHeight, dT);
	}

    // Timing
    auto stop_time_morph = chrono::high_resolution_clock::now();
    printf("Time spent on morphing: %ld ms\n", chrono::duration_cast<chrono::milliseconds>(stop_time_morph-start_time_morph).count() );

    ///////////////////////////



	// Saving image ///////////

    // Timing
    auto start_time_save = chrono::high_resolution_clock::now();

	for (int i = 0; i < steps+1; i++) {
        float t_i = stepSize*i;
        string path = tempFile + "output-" + to_string(t_i) + ".png";
        imgWrite(path, hMorphMapArr[i], imgWidthOrig, imgHeightOrig);
        free(hMorphMapArr[i]);
    	free(hMorphLinesArr[i]);
	}

    // Timing
    auto stop_time_save = chrono::high_resolution_clock::now();
    printf("Time spent on saving files: %ld ms\n", chrono::duration_cast<chrono::milliseconds>(stop_time_save-start_time_save).count() );

    ///////////////////////////



    // Free host side heap-allocated memory
    free(hMorphMapArr);
	free(hMorphLinesArr);
    free(args_arr);
    free(threads);

	// Free the device side heap-allocated memory
    cudaFree(dSrcLines);
    cudaFree(dDstLines);
    cudaFree(dMorphLines);
    cudaFree(dSrcImgMap);
    cudaFree(dDstImgMap);
    cudaFree(dMorphMap);

    // Timing
    auto stop_time_tot = chrono::high_resolution_clock::now();
    printf("Time spent in total: %ld ms\n", chrono::duration_cast<chrono::milliseconds>(stop_time_tot-start_time_tot).count() );

	return 0;
}


void* pthread_imgWrite(void* args) {
    // TODO: Fill in parallelized code
    return NULL;
}
