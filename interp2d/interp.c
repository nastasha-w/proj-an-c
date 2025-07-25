#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <time.h>

// determines which relative difference is tolerable in judging whether the grid points
// in a table are equally spaced
// relatively large value is to allow equal-spacing approximation for logT values of
// Rob Wiersma's cooling tables, clearly meant to be equally spaced
#define EQTOL_TABLE 2e-3



/* 
--------------
| prototypes |
--------------
*/

float bilinear_interpolation(float a, float b, 
		float a0, float a1, float b0, float b1, 
		float f00, float f01, float f10, float f11);

float linear_interpolation(float a, float a0, float a1, float f0, float f1);

int hasequalspacing(float* arr, int lenarr);
void findindices_eqspacing(float val, float x0, float dx, int lenarr, int incr, int* liadr, int* uiadr);
void findindices_neqspacing(float val, float* arr, int lenarr, int incr, int* liadr, int* uiadr, float* invspacingadr);

int interpolate_3d(float* tointerp1, float* tointerp2, float* tointerp3,
                   long long int NumPart, 
                   float* table, float* grid1, int lengrid1, 
                                 float* grid2, int lengrid2,
                                 float* grid3, int lengrid3,
                   float* out);
                   
int interpolate_2d(float* tointerp1, float* tointerp2,
                   long long int NumPart, 
                   float* table, float* grid1, int lengrid1, 
                                 float* grid2, int lengrid2,
                   float* out);

/* for tests */
int interpolate3d_test();
int interpolate2d_test();
int interpolate2d_pw_test();
int interpolate3d_pw_test();

float testfunc(float x, float y){return 1.;}
float testfunc1(float x, float y){return 0.1 + 3.0*x + 0.5*x*y;}
float testfunc2(float x, float y){return -5. + 2.4*y + 1.3*x*y;}
float testfunc3(float x, float y){return x*y;}
float test3func0(float x, float y, float z){return 0.;}
float test3func1(float x, float y, float z){return 1. + 0.5*x + 3.*y + 5*y*z;}
float test3func2(float x, float y, float z){return 1. + 0.5*x*z + 3.*z*y + 5*x*y*z;}
// prototypes for piecewise linear functions (to test use of correct table indices)
float testpw1(float x);
float testpw2(float x);
float testpw3(float x);
float testpw4(float x);
float testfunc1_pw(float x, float y);
float testfunc2_pw(float x, float y);
float test3func1_pw(float x, float y, float z);
float test3func1_pw(float x, float y, float z);

void printarray(float* arr, int len){int i; printf("{ ");for(i=0;i<len;i++){printf("%f, " ,arr[i]);}printf("}");}
void print3arr(float* arr, int* shape) {
	 int i, j ,k;
    for(i=0; i < shape[0]; i++){
        for(j=0; j < shape[1]; j++){
            for(k=0; k < shape[2]; k++){
                printf("%f, ", arr[shape[2]*shape[1]*i + shape[2]*j + k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

int isequal(float* arr1, float* arr2, int len) {
	 int i;
    for(i=0; i<len; i++){
        if(arr1[i] != arr2[i]){
            return 0;
        }
    }
    return 1; // all elements must have been equal
}

int iscloseto(float* arr1, float* arr2, int len, float maxdiff) {
	 int i;
    for(i=0; i<len; i++){
        //printf("%g\t",fabs(arr1[i] - arr2[i]));
        if(arr1[i]==0.)
          {
          if(fabs(arr2[i]) > maxdiff){return 0.;}
          }        
        else if(fabs((arr1[i] - arr2[i])/arr1[i]) > maxdiff){
            //printf("\n");
            return 0;
        }
    }
    //printf("\n");
    return 1; // all elements must have been close together
}



/*
--------
| main |
--------
Contains only test cases
*/



int main(void) 
{
  printf("-----------------------------------------------------------------------------------------\n-----------------------------------------------------------------------------------------\n                                    NEW TEST ROUND                           \n-----------------------------------------------------------------------------------------\n-----------------------------------------------------------------------------------------\n");
  int succes_interp2d = interpolate2d_test(); 
  int succes_interp3d = interpolate3d_test();
  int succes_interp2d_pw = interpolate2d_pw_test();
  int succes_interp3d_pw = interpolate3d_pw_test();
  if(succes_interp3d && succes_interp2d && succes_interp3d_pw && succes_interp2d_pw)
   {printf("\n\n2D and 3D interpolation passed their test series\n\n");}
  else
   {printf("\n\n--------------------------------------------------------------------------------\n|      WARNING! interpolate_2d or interpolate_3d did not pass their tests!     |\n--------------------------------------------------------------------------------\n\n");}
  return 0;
}





/* left in but not used*/

float linear_interpolation(float a, float a0, float a1, float f0, float f1)
{
  return 1./(a1-a0) * ( (a1-a)*f0 + (a-a0)*f1 ); 
}

/* bilinear interpolation. 
Arguments: 
a,b: where you want to know the function
a0, a1, b0, b1: a,b values where the function is known
f00, f01, f10, f11: function values at (a0, b0), (a0, b1), (a1, b0), (a1, b1) */

float bilinear_interpolation(float a, float b, 
		float a0, float a1, float b0, float b1, 
		float f00, float f01, float f10, float f11)
{
  float da1 = a1 - a;
  float da0 = a  - a0;
  float db1 = b1 - b;
  float db0 = b- b0; 
   
  return ( (da1)*(db1)*f00 + (da1)*(db0)*f01 + (da0)*(db1)*f10 + (da0)*(db0)*f11 )/((a1-a0)*(b1-b0));
}



/* Used in the interpolation functions: check grid spacing, find table and grid indices for a given value*/


/* returns 0 (False) or 1 (True) for whether the the input array has equal spacing*/
int hasequalspacing(float* arr, int lenarr)
{
  int eq = 1;
  if(lenarr>2){
    float diff = arr[1] - arr[0];
    float tol = diff*EQTOL_TABLE;	
    int i = 2; 
    while(i<lenarr && eq)
    {
      if(fabs(arr[i] - arr[i-1] -diff)  > tol){eq = 0;}	
      i++;
    }
  }
  return eq;
}



/* returns lower and upper indices of array elements bracketing val 
for arrays of equal spacing dx, with first element x0
elements outside array range: lower and upper indices are the extreme
edge indices */

void findindices_eqspacing(float val, float x0, float dx, int lenarr, int incr, int* liadr, int* uiadr)
{
  if( (incr && val < x0) || (1-incr && val > x0) ) //edge case: return index 0 for lower and upper
  { 
   #ifdef BOUNDSWARNINGS    
	printf("In findindices_eqspacing: value %f outside array range %f to %f \n",val,x0,x0+dx*(lenarr-1.)); 
	#endif
	*liadr = 0;
	*uiadr = 0;
  }
  else if( (incr && val > x0+dx*(lenarr-1.)) || (1-incr && val < x0+dx*(lenarr-1.)) ) // edge case: return last index for lower and upper
  {
   #ifdef BOUNDSWARNINGS    
	printf("In findindices_eqspacing: value %f outside array range %f to %f \n",val,x0,x0+dx*(lenarr-1.)); 
	#endif
	*liadr = lenarr-1;
	*uiadr = lenarr-1;
  }
  else if(incr)//normal index finding
  {
  	*liadr = floor((val-x0)/dx);
  	*uiadr = *liadr+1;
  }
  else{
  /* error has occurred */
  *liadr = -1;
  *uiadr = -1;
  printf("In findindices_eqspacing: error for value %f and array range %f to %f \n",val,x0,x0+lenarr*dx);
  }
}


/* returns lower index of array elements bracketing val 
assumes the array is monotonic, and has length >=2 
not the most efficient search, but the target tables have equal spacing
anyway 
elements outside array range: lower and upper indices are the extreme
edge indices */

void findindices_neqspacing(float val, float* arr, int lenarr, int incr, int* liadr, int* uiadr, float* invspacingadr)
{
  if( (incr && val < arr[0]) || (1-incr && val > arr[0]) )
  { 
    #ifdef BOUNDSWARNINGS  
	 printf("In findindices_neqspacing: value %f outside array range %f to %f \n",val,arr[0],arr[lenarr-1]);  
    #endif    
    *liadr = 0;
	 *uiadr = 0; 
	 *invspacingadr = 1;
  }
  else if( (incr && val > arr[lenarr-1]) || (1-incr && val < arr[lenarr-1]) )
  { 
    #ifdef BOUNDSWARNINGS  
	 printf("In findindices_neqspacing: value %f outside array range %f to %f \n",val,arr[0],arr[lenarr-1]);  
    #endif    
    *liadr = lenarr-1;
	 *uiadr = lenarr-1;
	 *invspacingadr = 1;
  }  
  else if(incr){  
    int i = 0;
    while(arr[i]<val){i++;}
    *liadr = i-1;
    *uiadr = i;
    *invspacingadr = 1./(arr[*uiadr] - arr[*liadr]);
  }
  else if(1-incr){
  	 int i = lenarr-1;  
    while(arr[i]<val){i--;}
    *liadr = i;
    *uiadr = i + 1;
    *invspacingadr = 1./(arr[*uiadr] - arr[*liadr]);
  }
  else{
  /* error has occurred */
  *liadr = -1;
  *uiadr = -1;
  printf("In findindices_neqspacing: error for value %f and array range %f to %f \n",val,arr[0],arr[lenarr-1]);
  }
}


int interpolate_2d(float* tointerp1, float* tointerp2, 
                   long long int NumPart , 
                   float* table, float* grid1, int lengrid1, 
                                 float* grid2, int lengrid2,
                   float* out)
{
  /* tointerp1 (2): particle properties to use in the interpolation
     NumPart: length of the tointerp# arrays (should all be the same)
     table: values to interpolate, on a grid
     grid#, lengrid#: grid points along each dimension, and the size 
                      of the grids
     grid1 is for the slowest-varying index, grid2 for the fastest
     out: where to put the output. memory should be allocated, length 
          NumPart 
     (pre-existing values will be overwritten)  
   */
	
  /* Deal with values outside the table range: 
  just use the highest/lowest values that are included */ 

  // timing (Could be important if we're doing an interpolation for each element)
  #ifdef _OPENMP
  double starttime_omp = omp_get_wtime();
  #endif
  clock_t starttime = clock();
      
  #ifdef _OPENMP
  // small isolated parallel region   
  #pragma omp parallel 
  {
  int size = omp_get_num_threads();
  int rank = omp_get_thread_num();
  if(rank==0){printf("interpolate_2d called with OpenMP on %i cores\n",size);}
  }
  #else
  printf("interpolate_2d called (serial version)\n");
  #endif 
 
 
  // are grids equally spaced? determines grid index calculation
  const int iseqgrid1 = hasequalspacing(grid1, lengrid1);
  const int iseqgrid2 = hasequalspacing(grid2, lengrid2);
  // are grids increasing or decreasing? 
  const int isincg1 = grid1[1] > grid1[0];
  const int isincg2 = grid2[1] > grid2[0];
  
  
  // get the table spacings for later (for unequal spacing: only idg# values are used; overwritten before inital values are used)
  const float dg1 = (grid1[lengrid1-1] - grid1[0])/(lengrid1-1.);
  float idg1 = 1./dg1;
  const float dg2 = (grid2[lengrid2-1] - grid2[0])/(lengrid2-1.);
  float idg2 = 1./dg2;

  #ifdef DEBUG2
  printf("idg1:  %f\n",idg1);
  printf("idg2:  %f\n",idg2);
  #endif


  // particle number counter (main loop)
  long long int i;
  // private: inverse grid spacings (could be shared in the equal spacing case, but that is unknown at compile time)
  #ifdef _OPENMP  
  #pragma omp parallel for firstprivate(idg1, idg2)
  #endif
  for(i=0;i<NumPart;i++)
  {
    // get upper and lower indices; edge cases handled in function
    int li1, li2, ui1, ui2;
	 if(iseqgrid1){findindices_eqspacing(tointerp1[i], grid1[0], dg1, lengrid1, isincg1, &li1, &ui1);}
    else{findindices_neqspacing(tointerp1[i], grid1, lengrid1, isincg1, &li1, &ui1, &idg1);} 
    if(iseqgrid2){findindices_eqspacing(tointerp2[i], grid2[0], dg2, lengrid2, isincg2, &li2, &ui2);}
    else{findindices_neqspacing(tointerp2[i], grid2, lengrid2, isincg2, &li2, &ui2, &idg2);} 	  	     
	     
	// do the actual interpolation
	float v00 = table[li1 * lengrid2 + li2];
	float v01 = table[li1 * lengrid2 + ui2];
	float v10 = table[ui1 * lengrid2 + li2];
	float v11 = table[ui1 * lengrid2 + ui2];
	     
	float w = (tointerp2[i] - grid2[li2]) * idg2; 
	#ifdef DEBUG
	printf("round 1 w: %f, idg2: %f\n", w, idg2);
	#endif

	float v0 = (1.-w)*v00 + w*v01; 
	float v1 = (1.-w)*v10 + w*v11; 
     
	w = (tointerp1[i] - grid1[li1]) * idg1;
	#ifdef DEBUG
	printf("round 2 w: %f, idg1: %f\n", w, idg1);
	#endif
	out[i] = (1.-w)*v0 + w*v1;

	#ifdef DEBUG
	printf("For particle %lli:\n", i); 
   printf("  values: %f, %f\n", tointerp1[i], tointerp2[i]); 
	printf("  grid values: (%f, %f), (%f, %f)\n", grid1[li1], grid1[ui1], grid2[li2], grid2[ui2]);
	printf("  indices:     (%i, %i), (%i, %i)\n",li1,ui1,li2,ui2);
	printf("  table values: %f %f\n                %f %f\n", v00, v01, v10, v11);
	printf("  round 2 values: %f %f\n",  v0, v1);
	printf("final value: %f\n\n", out[i]);     
   #endif
	     
  } // end of loop over values to interpolate (and parallel region if compiled with OpenMP)
  
  #ifdef _OPENMP
  double endtime_omp = omp_get_wtime();
  printf("Wall time for 2d interpolation (omp_get_wtime): %g s\n", endtime_omp-starttime_omp);
  #endif  
  clock_t endtime = clock();  
  printf("CPU time for 2d interpolation (clock): %g s\n",((float)(endtime-starttime))/CLOCKS_PER_SEC);
  
  return 0;
 }




int interpolate_3d(float* tointerp1, float* tointerp2, float* tointerp3, 
                   long long int NumPart , 
                   float* table, float* grid1, int lengrid1, 
                                 float* grid2, int lengrid2,
                                 float* grid3, int lengrid3,
                   float* out)
{
  /* tointerp1 (2,3): particle properties to use in the interpolation
     NumPart: length of the tointerp# arrays (should all be the same)
     table: values to interpolate, on a grid
     grid#, lengrid#: grid points along each dimension, and the size 
                      of the grids
     grid1 is for the slowest-varying index, grid3 for the fastest
     out: where to put the output. memory should be allocated, length 
          NumPart 
     (pre-existing values will be overwritten)  
   */
	
  /* Deal with values outside the table range: 
  just use the highest/lowest values that are included */ 

  // timing (Could be important if we're doing an interpolation for each element)
  #ifdef _OPENMP
  double starttime_omp = omp_get_wtime();
  #endif
  clock_t starttime = clock();
      
  #ifdef _OPENMP
  // small isolated parallel region   
  #pragma omp parallel 
  {
  int size = omp_get_num_threads();
  int rank = omp_get_thread_num();
  if(rank==0){printf("interpolate_3d called with OpenMP on %i cores\n",size);}
  }
  #else
  printf("interpolate_3d called (serial version)\n");
  #endif 
 
 
  // are grids equally spaced? determines grid index calculation
  const int iseqgrid1 = hasequalspacing(grid1, lengrid1);
  const int iseqgrid2 = hasequalspacing(grid2, lengrid2);
  const int iseqgrid3 = hasequalspacing(grid3, lengrid3);
  // are grids increasing or decreasing? 
  const int isincg1 = grid1[1] > grid1[0];
  const int isincg2 = grid2[1] > grid2[0];
  const int isincg3 = grid3[1] > grid3[0];
  
  
  // get the table spacings for later (for unequal spacing: only idg# values are used; overwritten before inital values are used)
  const float dg1 = (grid1[lengrid1-1] - grid1[0])/(lengrid1-1.);
  float idg1 = 1./dg1;
  const float dg2 = (grid2[lengrid2-1] - grid2[0])/(lengrid2-1.);
  float idg2 = 1./dg2;
  const float dg3 = (grid3[lengrid3-1] - grid3[0])/(lengrid3-1.);
  float idg3 = 1./dg3;

  #ifdef DEBUG2
  printf("idg1:  %f\n",idg1);
  printf("idg2:  %f\n",idg2);
  printf("idg3:  %f\n",idg3);
  #endif


  // particle number counter (main loop)
  long long int i;
  // private: inverse grid spacings (could be shared in the equal spacing case, but that is unknown at compile time)
  #ifdef _OPENMP  
  #pragma omp parallel for firstprivate(idg1, idg2, idg3)
  #endif
  for(i=0;i<NumPart;i++)
  {
    // get upper and lower indices; edge cases handled in function
    int li1, li2, li3, ui1, ui2, ui3;
	 if(iseqgrid1){findindices_eqspacing(tointerp1[i], grid1[0], dg1, lengrid1, isincg1, &li1, &ui1);}
    else{findindices_neqspacing(tointerp1[i], grid1, lengrid1, isincg1, &li1, &ui1, &idg1);} 
    if(iseqgrid2){findindices_eqspacing(tointerp2[i], grid2[0], dg2, lengrid2, isincg2, &li2, &ui2);}
    else{findindices_neqspacing(tointerp2[i], grid2, lengrid2, isincg2, &li2, &ui2, &idg2);} 
    if(iseqgrid3){findindices_eqspacing(tointerp3[i], grid3[0], dg3, lengrid3, isincg3, &li3, &ui3);}
    else{findindices_neqspacing(tointerp3[i], grid3, lengrid3, isincg3, &li3, &ui3, &idg3);} 	  	     
	     
	// do the actual interpolation

	float v000 = table[li1 * lengrid2*lengrid3 + li2 * lengrid3 + li3];
	float v001 = table[li1 * lengrid2*lengrid3 + li2 * lengrid3 + ui3];
	float v010 = table[li1 * lengrid2*lengrid3 + ui2 * lengrid3 + li3];
	float v011 = table[li1 * lengrid2*lengrid3 + ui2 * lengrid3 + ui3];
	float v100 = table[ui1 * lengrid2*lengrid3 + li2 * lengrid3 + li3];
	float v101 = table[ui1 * lengrid2*lengrid3 + li2 * lengrid3 + ui3];
	float v110 = table[ui1 * lengrid2*lengrid3 + ui2 * lengrid3 + li3];
	float v111 = table[ui1 * lengrid2*lengrid3 + ui2 * lengrid3 + ui3];
	     
	float w = (tointerp3[i] - grid3[li3]) * idg3; 
	#ifdef DEBUG
	printf("round 1 w: %f, idg3: %f\n", w, idg3);
	#endif
	float v00 = (1.-w)*v000 + w*v001;
	float v01 = (1.-w)*v010 + w*v011;
	float v10 = (1.-w)*v100 + w*v101;
	float v11 = (1.-w)*v110 + w*v111;
	     
	w = (tointerp2[i] - grid2[li2]) * idg2;
	#ifdef DEBUG
	printf("round 2 w: %f, idg2: %f\n", w, idg2);
	#endif
	float v0 = (1.-w)*v00 + w*v01; 
	float v1 = (1.-w)*v10 + w*v11; 

	     
	w = (tointerp1[i] - grid1[li1]) * idg1;
	#ifdef DEBUG
	printf("round 3 w: %f, idg1: %f\n", w, idg1);
	#endif
	out[i] = (1.-w)*v0 + w*v1;

	#ifdef DEBUG
	printf("For particle %lli:\n", i); 
    printf("  values: %f, %f, %f\n", tointerp1[i], tointerp2[i], tointerp3[i]); 
	printf("  grid values: (%f, %f), (%f, %f), (%f, %f)\n", grid1[li1], grid1[ui1], grid2[li2], grid2[ui2], grid3[li3], grid3[ui3]);
	printf("  indices:     (%i, %i), (%i, %i), (%i,%i)\n",li1,ui1,li2,ui2,li3,ui3);
	printf("  table values: %f %f\t %f %f\n                %f %f\t %f %f\n", v000, v001, v010, v011, v100, v101, v110, v111);
	printf("  round 2 values: %f %f\n                  %f %f\n",  v00, v01, v10, v11);
	printf("  round 3 values: %f %f\n",  v0, v1);
	printf("final value: %f\n\n", out[i]);     
    #endif
	     
  } // end of loop over values to interpolate (and parallel region if compiled with OpenMP)
  
  #ifdef _OPENMP
  double endtime_omp = omp_get_wtime();
  printf("Wall time for 3d interpolation (omp_get_wtime): %g s\n", endtime_omp-starttime_omp);
  #endif  
  clock_t endtime = clock();  
  printf("CPU time for 3d interpolation (clock): %g s\n",((float)(endtime-starttime))/CLOCKS_PER_SEC);
  
  return 0;
 }
 
 
 
int interpolate3d_test()
 {
  // ---------------------------------------------------------
  //               TESTS FOR 3D INTERPOLATION
  // ---------------------------------------------------------
  // positive/negative numbers, increasing/decreasing, equal/non-equal spacing
  // end points are fixed to keep edge case outcomes predictable
  // outcomes should be the same for all grid1,2,3 combinations
  
  // equal spacing, increasing
  float pgrid3[3] = {0.5, 1.6, 2.7};
  float pgrid2[5] = {-4.4, -3.2, -2., -0.8, 0.4};
  float pgrid1[3] = {-5.6, -5.3, -5.0};
  
  // equal spacing, decreasing
  float pdgrid3[3] = {2.7, 1.6, 0.5 };
  float pdgrid2[5] = {0.4, -0.8, -2., -3.2, -4.4};
  float pdgrid1[3] = {-5.0, -5.3, -5.6};

  // unequal spacing, increasing
  float pngrid3[3] = {0.5, 1.8, 2.7};
  float pngrid2[5] = {-4.4, -3.7, -2., -0.8, 0.4};
  float pngrid1[3] = {-5.6, -5.1, -5.0};

  // unequal spacing, decreasing
  float pndgrid3[3] = {2.7, 1.8, 0.5 };
  float pndgrid2[5] = {0.4, -0.8, -2., -3.7, -4.4};
  float pndgrid1[3] = {-5.0, -5.1, -5.6};

  int lengrid1 = 3;
  int lengrid2 = 5;
  int lengrid3 = 3;

  float vals3[27] = { 0.2, 1.3, 3.1,
	                0.2, 1.3, 3.1,
	                0.2, 1.3, 3.1,
	                
	                0.2, 1.3, 3.1,
	                0.2, 1.3, 3.1,
	                0.2, 1.3, 3.1,
	                
	                0.2, 1.3, 3.1,
	                0.2, 1.3, 3.1,
	                0.2, 1.3, 3.1};

  float vals3e[27]= { 0.5, 1.3, 2.7,
	                0.5, 1.3, 2.7,
	                0.5, 1.3, 2.7,
	                
	                0.5, 1.3, 2.7,
	                0.5, 1.3, 2.7,
	                0.5, 1.3, 2.7,
	                
	                0.5, 1.3, 2.7,
	                0.5, 1.3, 2.7,
	                0.5, 1.3, 2.7};  

  float vals2[27] = { 6.0,   6.0,   6.0,
	               -2.3,  -2.3,  -2.3,
	               -4.41, -4.41, -4.41,
	               
	                6.0,   6.0,   6.0,
	               -2.3,  -2.3,  -2.3,
	               -4.41, -4.41, -4.41,
	               
	                6.0,   6.0,   6.0,
	               -2.3,  -2.3,  -2.3,
	               -4.41, -4.41, -4.41};
	              
  float vals2e[27]= { 0.4,   0.4,   0.4,
	               -2.3,  -2.3,  -2.3,
	               -4.4,  -4.4,  -4.4,
	               
	                0.4,   0.4,   0.4,
	               -2.3,  -2.3,  -2.3,
	               -4.4,  -4.4,  -4.4,
	               
	                0.4,   0.4,   0.4,
	               -2.3,  -2.3,  -2.3,
	               -4.4,  -4.4,  -4.4};
	               
  float vals1[27] = { -8.,  -8.,  -8.,
	                -8.,  -8.,  -8.,
	                -8.,  -8.,  -8.,
	                
	                -5.2, -5.2, -5.2, 
	                -5.2, -5.2, -5.2, 
	                -5.2, -5.2, -5.2, 
	                 
	                 3.,   3.,   3.,
	                 3.,   3.,   3.,
	                 3.,   3.,   3.};

  float vals1e[27]= { -5.6, -5.6, -5.6,
	                -5.6, -5.6, -5.6,
	                -5.6, -5.6, -5.6,
	                
	                -5.2, -5.2, -5.2, 
	                -5.2, -5.2, -5.2, 
	                -5.2, -5.2, -5.2, 
	                
	                -5.0, -5.0, -5.0, 
	                -5.0, -5.0, -5.0, 
	                -5.0, -5.0, -5.0};
	                
  float output_test[27], output_check_f1[27], output_check_f2[27];	
  float table_f1[3*5*3], table_f2[3*5*3]; 
  int tableshape[3] = {3,5,3};
  float *grid1;
  float *grid2;
  float *grid3;      
  int shape[3]  = {3,3,3};        
  int gind1, gind2, gind3;
  int allpassed = 1;

  // should be the same for each test
  int i;
  for(i=0; i<27;i++)
  {
    output_check_f1[i] = test3func1(vals1e[i], vals2e[i], vals3e[i]);
    output_check_f2[i] = test3func2(vals1e[i], vals2e[i], vals3e[i]);
  }  
  
  // loop over the different grid combinations
  printf("\n-----------------------------------------------------------------------------------------\n                             interpolate_3d part 1                          \n-----------------------------------------------------------------------------------------\n");
  int gind;
  for(gind=0;gind<4*4*4;gind++)
  { 
    gind1 = gind/(4*4); // slowest index
    gind2 = gind/4 - gind1*4;    
    gind3 = gind%4;    
    printf("Test %i: grid1 %i, grid2 %i, grid3 %i\n",gind, gind1,gind2,gind3);

    if(gind1 ==0){grid1 = pgrid1;}
    else if(gind1 == 1){grid1 = pdgrid1;}
    else if(gind1 == 2){grid1 = pngrid1;}
    else if(gind1 == 3){grid1 = pndgrid1;}
    if(gind2 ==0){grid2 = pgrid2;}
    else if(gind2 == 1){grid2 = pdgrid2;}
    else if(gind2 == 2){grid2 = pngrid2;}
    else if(gind2 == 3){grid2 = pndgrid2;}
    if(gind3 ==0){grid3 = pgrid3;}
    else if(gind3 == 1){grid3 = pdgrid3;}
    else if(gind3 == 2){grid3 = pngrid3;}
    else if(gind3 == 3){grid3 = pndgrid3;}
    
  
    // run a test
    
    // generate the interpolation table for the given grids
    int i0,i1,i2, genind;
    for(genind = 0; genind < lengrid1*lengrid2*lengrid3; genind++)
        {
        i0 = genind/(lengrid2*lengrid3);
        i1 = genind/lengrid3 - lengrid2*i0;
        i2 = genind%lengrid3;
        table_f1[genind] = test3func1(grid1[i0], grid2[i1], grid3[i2]);
        table_f2[genind] = test3func2(grid1[i0], grid2[i1], grid3[i2]);
        }

     interpolate_3d(vals1, vals2, vals3, 
                   27, 
                   table_f1, grid1, lengrid1, 
                          grid2, lengrid2,
                          grid3, lengrid3,
                   output_test);

     if(iscloseto(output_test, output_check_f1, 27, 1e-5))
       {printf("passed test %i, function 1\n",gind);}
     else
       {
       printf("failed test %i, function 1\n",gind); 
       print3arr(output_test, shape); 
       print3arr(output_check_f1, shape);
       printf("Table1:\n");
       print3arr(table_f1, tableshape);
       allpassed=0;
       }
    
    interpolate_3d(vals1, vals2, vals3, 
                   27, 
                   table_f2, grid1, lengrid1, 
                          grid2, lengrid2,
                          grid3, lengrid3,
                   output_test);

    if(iscloseto(output_test, output_check_f2, 27, 1e-5))
      {printf("passed test %i, function 2\n",gind);}
    else
      {
      printf("failed test %i, function 2\n",gind);
      print3arr(output_test, shape); 
      print3arr(output_check_f2, shape); 
      allpassed=0;
      }
    
   }// end of loop over different grids
 if(allpassed){printf("\n All tests passed! (interpolate_3d, part 1)\n");}  
 return allpassed;   
 }
 
 
 int interpolate2d_test()
 {
  // ---------------------------------------------------------
  //               TESTS FOR 2D INTERPOLATION
  // ---------------------------------------------------------
  // positive/negative numbers, increasing/decreasing, equal/non-equal spacing
  // end points are fixed to keep edge case outcomes predictable
  // outcomes should be the same for all grid1,2 combinations
  
  // equal spacing, increasing
  float pgrid1[3] = {0.5, 1.6, 2.7};
  float pgrid2[5] = {-4.4, -3.2, -2., -0.8, 0.4};
  
  // equal spacing, decreasing
  float pdgrid1[3] = {2.7, 1.6, 0.5 };
  float pdgrid2[5] = {0.4, -0.8, -2., -3.2, -4.4};

  // unequal spacing, increasing
  float pngrid1[3] = {0.5, 1.8, 2.7};
  float pngrid2[5] = {-4.4, -3.7, -2., -0.8, 0.4};

  // unequal spacing, decreasing
  float pndgrid1[3] = {2.7, 1.8, 0.5 };
  float pndgrid2[5] = {0.4, -0.8, -2., -3.7, -4.4};

  int lengrid1 = 3;
  int lengrid2 = 5;

  float vals1[9] = { 0.2, 1.3, 3.1,
	                0.2, 1.3, 3.1,
	                0.2, 1.3, 3.1};

  float vals1e[9]= { 0.5, 1.3, 2.7,
	                0.5, 1.3, 2.7,
	                0.5, 1.3, 2.7};  

  float vals2[9] = { 6.0,   6.0,   6.0,
	               -2.3,  -2.3,  -2.3,
	               -4.41, -4.41, -4.41};
	              
  float vals2e[9]= { 0.4,   0.4,   0.4,
	               -2.3,  -2.3,  -2.3,
	               -4.4,  -4.4,  -4.4};
	               
	                
  float output_test[9], output_check_f1[9], output_check_f2[9];	
  float table_f1[3*5], table_f2[3*5]; 
  int tableshape[2] = {3,5};
  float *grid1;
  float *grid2;      
  int shape[3]  = {3,3,1};        
  int gind1, gind2;
  int allpassed = 1;

  // should be the same for each test
  int i;
  for(i=0; i<9;i++)
  {
    output_check_f1[i] = testfunc1(vals1e[i], vals2e[i]);
    output_check_f2[i] = testfunc2(vals1e[i], vals2e[i]);
  }  
  
  // loop over the different grid combinations
  printf("\n-----------------------------------------------------------------------------------------\n                             interpolate_2d part 1                         \n-----------------------------------------------------------------------------------------\n");
  int gind;
  for(gind=0;gind<4*4;gind++)
  { 
    gind1 = gind/4; // slowest index 
    gind2 = gind%4;    
    printf("Test %i: grid1 %i, grid2 %i\n",gind, gind1,gind2);

    if(gind1 ==0){grid1 = pgrid1;}
    else if(gind1 == 1){grid1 = pdgrid1;}
    else if(gind1 == 2){grid1 = pngrid1;}
    else if(gind1 == 3){grid1 = pndgrid1;}
    if(gind2 ==0){grid2 = pgrid2;}
    else if(gind2 == 1){grid2 = pdgrid2;}
    else if(gind2 == 2){grid2 = pngrid2;}
    else if(gind2 == 3){grid2 = pndgrid2;}
    
  
    // run a test
    
    // generate the interpolation table for the given grids
    int i0,i1, genind;
    for(genind = 0; genind < lengrid1*lengrid2; genind++)
        {
        i0 = genind/(lengrid2);
        i1 = genind%lengrid2;
        table_f1[genind] = testfunc1(grid1[i0], grid2[i1]);
        table_f2[genind] = testfunc2(grid1[i0], grid2[i1]);
        }

     interpolate_2d(vals1, vals2, 
                   9, 
                   table_f1, grid1, lengrid1, 
                          grid2, lengrid2,
                   output_test);

     if(iscloseto(output_test, output_check_f1, 9, 1e-5))
       {printf("passed test %i, function 1\n",gind);}
     else
       {
       printf("failed test %i, function 1\n",gind); 
       print3arr(output_test, shape); 
       print3arr(output_check_f1, shape);
       printf("Table1:\n");
       print3arr(table_f1, tableshape);
       allpassed=0;
       }
    
    interpolate_2d(vals1, vals2, 
                   9, 
                   table_f2, grid1, lengrid1, 
                          grid2, lengrid2,
                   output_test);

    if(iscloseto(output_test, output_check_f2, 9, 1e-5))
      {printf("passed test %i, function 2\n",gind);}
    else
      {
      printf("failed test %i, function 2\n",gind);
      print3arr(output_test, shape); 
      print3arr(output_check_f2, shape); 
      allpassed=0;
      }
    
   }// end of loop over different grids
 if(allpassed){printf("\n All tests passed! (interpolate_2d, part 1)\n");} 
 
 return allpassed;    
 }
 

// 1.2, 1.7, 2.2, d=0.5 
float testpw1(float x)
{
  if(x<1.2){return 0.5 + x;}
  else if(x<1.7){return 1.7 + 0.5*(x-1.2);}
  else if(x<2.2){return 1.95 + 3.*(x-1.7);}
  else{return 3.45 + 1.7*(x-2.2);}
}
// -1.6, -0.3, 1., d=1.3
float testpw2(float x)
{
  if(x<-1.6){return -2.3-x;}
  else if(x<-0.3){return -0.7+ 0.5*(x+1.6);}
  else if(x<1.){return -0.05 + 3*(x+0.3);}
  else{return 3.85 - 0.2*(x-1.);}
}

// -3.5, -2.1, -0.7, d=1.4
float testpw3(float x)
{
  if(x<-3.5){return 2.-0.5*(x+3.5);}
  else if(x<-2.1){return 2.-1.4*(x+3.5);}
  else if(x<-0.7){return 0.04 - 0.2*(x+2.1);}
  else{return -0.24+3.*(x+0.7);}
}

// -0.3, 0.5, 1.3, d=0.8
float testpw4(float x)
{
  if(x<-0.3){return 3.*(x+0.3);}
  else if(x<0.5){return -2.*(x+0.3);}
  else if(x<1.3){return -1.6 + 2.5*(x-0.5);}
  else{return 0.4-0.3*(x-1.3);}
}

float testfunc1_pw(float x, float y){return testpw1(x)*testpw2(y);} 
float testfunc2_pw(float x, float y){return testpw3(x)*testpw4(y);}
float test3func1_pw(float x, float y, float z){return testpw1(x)*testpw2(y)*testpw3(z);} 
float test3func2_pw(float x, float y, float z){return testpw3(x)*testpw4(y)*testpw1(z);}

 int interpolate2d_pw_test()
 {
  // ---------------------------------------------------------
  //          TESTS FOR 2D INTERPOLATION: Part 2
  // ---------------------------------------------------------
  // positive/negative numbers, increasing/decreasing, equal/non-equal spacing
  // end points are fixed to keep edge case outcomes predictable
  // outcomes should be the same for all grid1,2 combinations
  // uses piecewise functions for the tests, to check correct indices use in non-edge cases
  
  printf("\n-----------------------------------------------------------------------------------------\n                             interpolate_2d part 2                         \n-----------------------------------------------------------------------------------------\n");

  // equal spacing, increasing
  float pgrid_f1[5] = { 0.7,  1.2,  1.7,  2.2, 2.7};
  float pgrid_f2[5] = {-2.9, -1.6, -0.3,  1.0, 2.3};
  float pgrid_f3[5] = {-4.9, -3.5, -2.1, -0.7, 1.1};
  float pgrid_f4[5] = {-1.1, -0.3,  0.5,  1.3, 2.1};
  
  // equal spacing, decreasing
  float pdgrid_f1[5] = {2.7,  2.2,  1.7,  1.2,  0.7};
  float pdgrid_f2[5] = {2.3,  1.0, -0.3, -1.6, -2.9};
  float pdgrid_f3[5] = {1.1, -0.7, -2.1, -3.5, -4.9};
  float pdgrid_f4[5] = {2.1,  1.3,  0.5, -0.3, -1.1};

  // unequal spacing, increasing
  float pngrid_f1[5] = { 0.6,  1.2,  1.7,  2.2, 2.7};
  float pngrid_f2[5] = {-3.0, -1.6, -0.3,  1.0, 2.3};
  float pngrid_f3[5] = {-5.0, -3.5, -2.1, -0.7, 1.1};
  float pngrid_f4[5] = {-1.2, -0.3,  0.5,  1.3, 2.1};

  // unequal spacing, decreasing
  float pndgrid_f1[5] = {2.7,  2.2,  1.7,  1.2,  0.6};
  float pndgrid_f2[5] = {2.3,  1.0, -0.3, -1.6, -3.0};
  float pndgrid_f3[5] = {1.1, -0.7, -2.1, -3.5, -5.0};
  float pndgrid_f4[5] = {2.1,  1.3,  0.5, -0.3, -1.2};

  int lengrid_f1 = 5;
  int lengrid_f2 = 5;
  int lengrid_f3 = 5;
  int lengrid_f4 = 5;

  // since values depend on the function being called: 
  // set test values along with grid values for each 
  // test set
  
  // for pw1
  float vals_f1[3]  = { 0.2, 1.3, 3.1};
  float vals_f1e[3] = { 0.7, 1.3, 2.7};  
  float vals_f1ne[3]= { 0.6, 1.3, 2.7}; 
	                    
  // for pw2	                    
  float vals_f2[3]  = { -4.5, -0.7, 3.1};
  float vals_f2e[3] = { -2.9, -0.7, 2.3};  
  float vals_f2ne[3]= { -3.0, -0.7, 2.3}; 
  
  // for pw3
  float vals_f3[3]  = { -5.6, -0.9, 3.1};
  float vals_f3e[3] = { -4.9, -0.9, 1.1};  
  float vals_f3ne[3]= { -5.0, -0.9, 1.1}; 
	                    
  // for pw4	                    
  float vals_f4[3]  = { -5.0, 0.0, 5.0};
  float vals_f4e[3] = { -1.1, 0.0, 2.1};  
  float vals_f4ne[3]= { -1.2, 0.0, 2.1}; 

	                
  float output_test[9], output_check[9];	
  float table[5*5]; 
  int tableshape[3] = {1,5,5};
  float *grid1, *grid2;
  float *pvals1e, *pvals2e;
  float vals1[9], vals2[9];     
  int shape[3]  = {1,3,3};        
  int gind1, gind2;
  int allpassed = 1;

  
  // loop over the different grid combinations (function 1)
  printf("\n\ninterpolate_2d, piecewise function 1\n\n");
  int gind;
  for(gind=0;gind<4*4;gind++)
  { 
    // retrieve the values to input for piecewise function 1
    gind1 = gind/4; // slowest index 
    gind2 = gind%4;    
    printf("Test %i: grid1 %i, grid2 %i\n",gind, gind1,gind2);

    if(gind1 ==0){grid1 = pgrid_f1; pvals1e = vals_f1e;}
    else if(gind1 == 1){grid1 = pdgrid_f1; pvals1e = vals_f1e;}
    else if(gind1 == 2){grid1 = pngrid_f1; pvals1e = vals_f1ne;}
    else if(gind1 == 3){grid1 = pndgrid_f1; pvals1e = vals_f1ne;}
    if(gind2 ==0){grid2 = pgrid_f2; pvals2e = vals_f2e;}
    else if(gind2 == 1){grid2 = pdgrid_f2; pvals2e = vals_f2e;}
    else if(gind2 == 2){grid2 = pngrid_f2; pvals2e = vals_f2ne;}
    else if(gind2 == 3){grid2 = pndgrid_f2; pvals2e = vals_f2ne;}
    
    int lengrid1 = lengrid_f1;
    int lengrid2 = lengrid_f2;
    
    // put 1d values into their 2d arrays, get check values  
    int i;
    for(i=0; i<3*3; i++)
    {
    	int xi = i/3;
    	int yi = i%3;
    	vals1[i] = vals_f1[xi];
      vals2[i] = vals_f2[yi];  
      output_check[i] = testfunc1_pw(pvals1e[xi], pvals2e[yi]);
    }
        
    // generate the interpolation table for the given grids
    int i0,i1, genind;
    for(genind = 0; genind < lengrid1*lengrid2; genind++)
        {
        i0 = genind/(lengrid2);
        i1 = genind%lengrid2;
        table[genind] = testfunc1_pw(grid1[i0], grid2[i1]);
        }

     interpolate_2d(vals1, vals2, 
                   9, 
                   table, grid1, lengrid1, 
                          grid2, lengrid2,
                   output_test);

     if(iscloseto(output_test, output_check, 9, 1e-5))
       {printf("passed test %i, function 1\n",gind);}
     else
       {
       printf("failed test %i, function 1\n",gind); 
       print3arr(output_test, shape); 
       print3arr(output_check, shape);
       printf("Table:\n");
       print3arr(table, tableshape);
       allpassed=0;
       }

    
   }// end of loop over different grids (function 1)
   
  printf("\n\ninterpolate_2d, piecewise function 2\n\n");
  // int gind; done above
  for(gind=0;gind<4*4;gind++)
  { 
    // retrieve the values to input for piecewise function 1
    gind1 = gind/4; // slowest index 
    gind2 = gind%4;    
    printf("Test %i: grid1 %i, grid2 %i",gind, gind1,gind2);

    if(gind1 ==0){grid1 = pgrid_f3; pvals1e = vals_f3e;}
    else if(gind1 == 1){grid1 = pdgrid_f3; pvals1e = vals_f3e;}
    else if(gind1 == 2){grid1 = pngrid_f3; pvals1e = vals_f3ne;}
    else if(gind1 == 3){grid1 = pndgrid_f3; pvals1e = vals_f3ne;}
    if(gind2 ==0){grid2 = pgrid_f4; pvals2e = vals_f4e;}
    else if(gind2 == 1){grid2 = pdgrid_f4; pvals2e = vals_f4e;}
    else if(gind2 == 2){grid2 = pngrid_f4; pvals2e = vals_f4ne;}
    else if(gind2 == 3){grid2 = pndgrid_f4; pvals2e = vals_f4ne;}
    
    int lengrid1 = lengrid_f3;
    int lengrid2 = lengrid_f4;
    
    // put 1d values into their 2d arrays, get check values 
    int i; 
    for(i=0; i<3*3; i++)
    {
    	int xi = i/3;
    	int yi = i%3;
    	vals1[i] = vals_f3[xi];
      vals2[i] = vals_f4[yi];  
      output_check[i] = testfunc2_pw(pvals1e[xi], pvals2e[yi]);
    }
        
    // generate the interpolation table for the given grids
    int genind;
    for(genind = 0; genind < lengrid1*lengrid2; genind++)
        {
        int i0 = genind/(lengrid2);
        int i1 = genind%lengrid2;
        table[genind] = testfunc2_pw(grid1[i0], grid2[i1]);
        }

     interpolate_2d(vals1, vals2, 
                   9, 
                   table, grid1, lengrid1, 
                          grid2, lengrid2,
                   output_test);

     if(iscloseto(output_test, output_check, 9, 1e-5))
       {printf("passed test %i, function 2\n",gind);}
     else
       {
       printf("failed test %i, function 2\n",gind); 
       print3arr(output_test, shape); 
       print3arr(output_check, shape);
       printf("Table:\n");
       print3arr(table, tableshape);
       allpassed=0;
       }
    
   }// end of loop over different grids (function 2)
      
 if(allpassed){printf("\n All tests passed (interpolate_2d part 2) !\n");} 
 
 return allpassed;    
 }

int interpolate3d_pw_test()
 {
  // ---------------------------------------------------------
  //          TESTS FOR 3D INTERPOLATION: Part 2
  // ---------------------------------------------------------
  // positive/negative numbers, increasing/decreasing, equal/non-equal spacing
  // end points are fixed to keep edge case outcomes predictable
  // outcomes should be the same for all grid1,2 combinations
  // uses piecewise functions for the tests, to check correct indices use in non-edge cases
  
  printf("\n-----------------------------------------------------------------------------------------\n                             interpolate_3d part 2                         \n-----------------------------------------------------------------------------------------\n");

  // equal spacing, increasing
  float pgrid_f1[5] = { 0.7,  1.2,  1.7,  2.2, 2.7};
  float pgrid_f2[5] = {-2.9, -1.6, -0.3,  1.0, 2.3};
  float pgrid_f3[5] = {-4.9, -3.5, -2.1, -0.7, 1.1};
  float pgrid_f4[5] = {-1.1, -0.3,  0.5,  1.3, 2.1};
  
  // equal spacing, decreasing
  float pdgrid_f1[5] = {2.7,  2.2,  1.7,  1.2,  0.7};
  float pdgrid_f2[5] = {2.3,  1.0, -0.3, -1.6, -2.9};
  float pdgrid_f3[5] = {1.1, -0.7, -2.1, -3.5, -4.9};
  float pdgrid_f4[5] = {2.1,  1.3,  0.5, -0.3, -1.1};

  // unequal spacing, increasing
  float pngrid_f1[5] = { 0.6,  1.2,  1.7,  2.2, 2.7};
  float pngrid_f2[5] = {-3.0, -1.6, -0.3,  1.0, 2.3};
  float pngrid_f3[5] = {-5.0, -3.5, -2.1, -0.7, 1.1};
  float pngrid_f4[5] = {-1.2, -0.3,  0.5,  1.3, 2.1};

  // unequal spacing, decreasing
  float pndgrid_f1[5] = {2.7,  2.2,  1.7,  1.2,  0.6};
  float pndgrid_f2[5] = {2.3,  1.0, -0.3, -1.6, -3.0};
  float pndgrid_f3[5] = {1.1, -0.7, -2.1, -3.5, -5.0};
  float pndgrid_f4[5] = {2.1,  1.3,  0.5, -0.3, -1.2};

  int lengrid_f1 = 5;
  int lengrid_f2 = 5;
  int lengrid_f3 = 5;
  int lengrid_f4 = 5;

  // since values depend on the function being called: 
  // set test values along with grid values for each 
  // test set
  
  // for pw1
  float vals_f1[3]  = { 0.2, 1.3, 3.1};
  float vals_f1e[3] = { 0.7, 1.3, 2.7};  
  float vals_f1ne[3]= { 0.6, 1.3, 2.7}; 
	                    
  // for pw2	                    
  float vals_f2[3]  = { -4.5, -0.7, 3.1};
  float vals_f2e[3] = { -2.9, -0.7, 2.3};  
  float vals_f2ne[3]= { -3.0, -0.7, 2.3}; 
  
  // for pw3
  float vals_f3[3]  = { -5.6, -0.9, 3.1};
  float vals_f3e[3] = { -4.9, -0.9, 1.1};  
  float vals_f3ne[3]= { -5.0, -0.9, 1.1}; 
	                    
  // for pw4	                    
  float vals_f4[3]  = { -5.0, 0.0, 5.0};
  float vals_f4e[3] = { -1.1, 0.0, 2.1};  
  float vals_f4ne[3]= { -1.2, 0.0, 2.1}; 

	                
  float output_test[27], output_check[27];	
  float table[5*5*5]; 
  int tableshape[3] = {5,5,5};
  float *grid1, *grid2, *grid3;
  float *pvals1e, *pvals2e, *pvals3e;
  float vals1[27], vals2[27], vals3[27];     
  int shape[3]  = {3,3,3};        
  int gind1, gind2, gind3;
  int allpassed = 1;

  
  // loop over the different grid combinations (function 1)
  printf("\n\ninterpolate_2d, piecewise function 1\n\n");
  int gind;
  for(gind=0;gind<4*4*4;gind++)
  { 
    // retrieve the values to input for piecewise function 1
    gind1 = gind/(4*4); // slowest index
    gind2 = gind/4 - gind1*4;    
    gind3 = gind%4;    
    printf("Test %i: grid1 %i, grid2 %i, grid3 %i\n",gind, gind1,gind2,gind3);

    if(gind1 ==0){grid1 = pgrid_f1; pvals1e = vals_f1e;}
    else if(gind1 == 1){grid1 = pdgrid_f1; pvals1e = vals_f1e;}
    else if(gind1 == 2){grid1 = pngrid_f1; pvals1e = vals_f1ne;}
    else if(gind1 == 3){grid1 = pndgrid_f1; pvals1e = vals_f1ne;}
    if(gind2 ==0){grid2 = pgrid_f2; pvals2e = vals_f2e;}
    else if(gind2 == 1){grid2 = pdgrid_f2; pvals2e = vals_f2e;}
    else if(gind2 == 2){grid2 = pngrid_f2; pvals2e = vals_f2ne;}
    else if(gind2 == 3){grid2 = pndgrid_f2; pvals2e = vals_f2ne;}
    if(gind3 ==0){grid3 = pgrid_f3; pvals3e = vals_f3e;}
    else if(gind3 == 1){grid3 = pdgrid_f3; pvals3e = vals_f3e;}
    else if(gind3 == 2){grid3 = pngrid_f3; pvals3e = vals_f3ne;}
    else if(gind3 == 3){grid3 = pndgrid_f3; pvals3e = vals_f3ne;}
    
    int lengrid1 = lengrid_f1;
    int lengrid2 = lengrid_f2;
    int lengrid3 = lengrid_f3;
    
    // put 1d values into their 2d arrays, get check values 
    int i; 
    for(i=0; i<3*3*3; i++)
    {
    	int xi = i/(3*3);
    	int yi = i/3 - xi*3;
    	int zi = i%3;
    	vals1[i] = vals_f1[xi];
      vals2[i] = vals_f2[yi]; 
      vals3[i] = vals_f3[zi];
      output_check[i] = test3func1_pw(pvals1e[xi], pvals2e[yi], pvals3e[zi]);
    }
        
    // generate the interpolation table for the given grids
    int i0,i1,i2, genind;
    for(genind = 0; genind < lengrid1*lengrid2*lengrid3; genind++)
        {
        i0 = genind/(lengrid2*lengrid3);
        i1 = genind/lengrid3 - lengrid2*i0;
        i2 = genind%lengrid3;
        table[genind] = test3func1_pw(grid1[i0], grid2[i1], grid3[i2]);
        }

     interpolate_3d(vals1, vals2, vals3,
                   27, 
                   table, grid1, lengrid1, 
                          grid2, lengrid2,
                          grid3, lengrid3,
                   output_test);

     if(iscloseto(output_test, output_check, 27, 1e-5))
       {printf("passed test %i, function 1\n",gind);}
     else
       {
       printf("failed test %i, function 1\n",gind); 
       print3arr(output_test, shape); 
       print3arr(output_check, shape);
       printf("Table:\n");
       print3arr(table, tableshape);
       allpassed=0;
       }

    
   }// end of loop over different grids (function 1)
   
  printf("\n\ninterpolate_2d, piecewise function 2\n\n");
  // int gind; above
  for(gind=0;gind<4*4*4;gind++)
  { 
    // retrieve the values to input for piecewise function 1
    gind1 = gind/(4*4); // slowest index
    gind2 = gind/4 - gind1*4;    
    gind3 = gind%4;    
    printf("Test %i: grid1 %i, grid2 %i, grid3 %i\n",gind, gind1,gind2,gind3);

    if(gind1 ==0){grid1 = pgrid_f3; pvals1e = vals_f3e;}
    else if(gind1 == 1){grid1 = pdgrid_f3; pvals1e = vals_f3e;}
    else if(gind1 == 2){grid1 = pngrid_f3; pvals1e = vals_f3ne;}
    else if(gind1 == 3){grid1 = pndgrid_f3; pvals1e = vals_f3ne;}
    if(gind2 ==0){grid2 = pgrid_f4; pvals2e = vals_f4e;}
    else if(gind2 == 1){grid2 = pdgrid_f4; pvals2e = vals_f4e;}
    else if(gind2 == 2){grid2 = pngrid_f4; pvals2e = vals_f4ne;}
    else if(gind2 == 3){grid2 = pndgrid_f4; pvals2e = vals_f4ne;}
    if(gind3 ==0){grid3 = pgrid_f1; pvals3e = vals_f1e;}
    else if(gind3 == 1){grid3 = pdgrid_f1; pvals3e = vals_f1e;}
    else if(gind3 == 2){grid3 = pngrid_f1; pvals3e = vals_f1ne;}
    else if(gind3 == 3){grid3 = pndgrid_f1; pvals3e = vals_f1ne;}
    
    int lengrid1 = lengrid_f3;
    int lengrid2 = lengrid_f4;
    int lengrid3 = lengrid_f1;
    
    // put 1d values into their 2d arrays, get check values  
    int i;
    for(i=0; i<3*3*3; i++)
    {
    	int xi = i/(3*3);
    	int yi = i/3 - xi*3;
    	int zi = i%3;
    	vals1[i] = vals_f3[xi];
      vals2[i] = vals_f4[yi]; 
      vals3[i] = vals_f1[zi]; 
      output_check[i] = test3func2_pw(pvals1e[xi], pvals2e[yi], pvals3e[zi]);
    }
        
    // generate the interpolation table for the given grids
    int i0,i1,i2, genind;
    for(genind = 0; genind < lengrid1*lengrid2*lengrid3; genind++)
        {
        i0 = genind/(lengrid2*lengrid3);
        i1 = genind/lengrid3 - lengrid2*i0;
        i2 = genind%lengrid3;
        table[genind] = test3func2_pw(grid1[i0], grid2[i1], grid3[i2]);
        }

     interpolate_3d(vals1, vals2, vals3,
                   27, 
                   table, grid1, lengrid1, 
                          grid2, lengrid2,
                          grid3, lengrid3,
                   output_test);

     if(iscloseto(output_test, output_check, 27, 1e-5))
       {printf("passed test %i, function 2\n",gind);}
     else
       {
       printf("failed test %i, function 1\n",gind); 
       print3arr(output_test, shape); 
       print3arr(output_check, shape);
       printf("Table:\n");
       print3arr(table, tableshape);
       allpassed=0;
       }
    
   }// end of loop over different grids (function 2)
      
 if(allpassed){printf("\n All tests passed (interpolate_3d part 2) !\n");} 
 
 return allpassed;    
 }