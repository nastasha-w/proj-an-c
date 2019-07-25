#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define EQTOL_TABLE 1e-6

/* functions to calculate emission from a given rho-T-abundance table */


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

int findindex_eqspacing(float val, float x0, float dx, int lenarr);

int findindex_neqspacing(float val, float* arr, int lenarr);

int interpolate_emdenssq(float* rho, float* T, long long int NumPart2 , float* emissiontable, float* rhotable, int lenrhotable, float* Ttable, int lenTtable, float* emission);



/* for tests */
float testfunc(float x, float y){return 1.;}
float testfunc1(float x, float y){return x;}
float testfunc2(float x, float y){return y;}
float testfunc3(float x, float y){return x*y;}
float test3func1(float x, float y, float z){return 1. + 0.5*x + 3.*y + 5*y*z;}
float test3func2(float x, float y, float z){return 1. + 0.5*x*z + 3.*z*y + 5*x*y*z;}
void printarray(float* arr, int len){printf("{ ");for(int i=0;i<len;i++){printf("%f, " ,arr[i]);}printf("}");}

int isequal(float* arr1, float* arr2, int len) {
    for(int i=0; i<len; i++){
        if(arr1[i] != arr2[i]){
            return 0;
        }
    }
    return 1; // all elements must have been equal
}

int iscloseto(float* arr1, float* arr2, int len, float maxdiff) {
    for(int i=0; i<len; i++){
        //printf("%g\t",fabs(arr1[i] - arr2[i]));
        if(fabs(arr1[i] - arr2[i]) > maxdiff){
            //printf("\n");
            return 0;
        }
    }
    //printf("\n");
    return 1; // all elements must have been close together
}

void print3arr(float* arr, int* shape) {
    for(int i=0; i < shape[0]; i++){
        for(int j=0; j < shape[1]; j++){
            for(int k=0; k < shape[2]; k++){
                printf("%f, ", arr[shape[2]*shape[1]*i + shape[2]*j + k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

/*
--------
| main |
--------
Contains only test cases
*/



int main(void) 
{
/*
 // tests for 2d interpolation	
printf("Table interpolation tests \n");

float emtab[9] = {0.5, 2.0, 0.0, 3.5, 5.0, 0.0, 0, 0, 0};
float Ttab[3] = {2,8,13};
float rhotab[3] = {4,1,-3};
int lenTtab = 3;
int lenrhotab = 3;

float tT [4]  = {4,4,6,6};
float trho[4] = {2,3,2,3};
int NP = 4;

 
float* nemission =(float*)malloc(NP*sizeof(float));

interpolate_emdenssq(trho,tT,NP,emtab,rhotab,lenrhotab,Ttab,lenTtab,nemission);
 
printf("For emission table x*y\n  x: 0.-9., y: 0.-2.\n ");
printf("\n  T array   ");
printarray(tT,NP); 
printf("\n  rho array "); 
printarray(trho,NP); 
printf("\n");
printf("calculated emissions are:\n            ");
printarray(nemission,NP);
printf(".\n");  
*/

   
return 0;
}





/*
--------------------------------
| the functions doing the work |
--------------------------------
*/






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
      if(abs(arr[i] - arr[i-1] -diff)  > tol){eq = 0;}	
      i++;
    }
  }
  return eq;
}



/* returns lower index of array elements bracketing val 
for arrays of equal spacing dx, with first element x0
index -1 flags elements outside the array range */

int findindex_eqspacing(float val, float x0, float dx, int lenarr)
{
  if((dx > 0 && (val < x0 || val > x0+dx*(lenarr-1.)) ) || (dx < 0 && (val > x0 || val < x0+dx*(lenarr-1.)) ) )
  { 
     #ifdef BOUNDSWARNINGS    
	 printf("In findindex_eqspacing: value %f outside array range %f to %f \n",val,x0,x0+dx*(lenarr-1.)); 
	 #endif
	 return -1; 
  }
  else
  {
  	 return floor((val-x0)/dx);
  }
}







/* returns lower index of array elements bracketing val 
assumes the array is monotonic, and has length >=2 
index -1 flags elements outside the array range 
not the most efficient search, but the target tables have equal spacing
anyway */

int findindex_neqspacing(float val, float* arr, int lenarr)
{
  int increasing = 	((arr[1] < arr[0]) ? 0 : 1);
  if( (increasing == 1 && (val < arr[0] || val > arr[lenarr-1] )) || (increasing == 0 && (val > arr[0] || val < arr[lenarr-1])) )
  { 
    #ifdef BOUNDSWARNINGS  
	 printf("In findindex_neqspacing: value %f outside array range %f to %f \n",val,arr[0],arr[lenarr-1]);  
    #endif    
    return -1;  
  }
  else if(increasing ==1){  
    int i = 0;
    while(arr[i]<val){i++;}
    return i-1;
  }
  else if(increasing ==0){
  	 int i = lenarr-1;  
    while(arr[i]<val){i--;}
    return i;
  }
  /* error has occurred */
  return -2;  
}


/* The actual interpolation 
assumes the emission table dimensions match those of rhotable, Ttable
the emission table should already be line-selected, and z-interpolated
emission maps: [T][rho][line] must be flattend, with the rho-index changing fastest */
/*Speed-up?: calculate dx only once, not at every findindex_equalspacing call*/
int interpolate_emdenssq(float* rho, float* T, long long int NumPart2 , float* emissiontable, float* rhotable, int lenrhotable, float* Ttable, int lenTtable, float* emission)
{
  printf("interpolate_emdenssq called \n");
  int lowerindex_rho;
  int lowerindex_T;

  long long int i; 
 
  /*emission = (float*)malloc(NumPart2*sizeof(float)); ALLOCATE MEMORY FOR THIS BEFORE FUNCTION CALL*/
  
  /* Deal with T, rho outside the emission table range: 
  just use the highest/lowest values that are included */ 
  /* Long to deal with 4 cases: equal spacing or not in rho and T tables */
  
  
  if(hasequalspacing(rhotable,lenrhotable))
  {
  	float drho = (rhotable[lenrhotable-1] - rhotable[0])/(lenrhotable-1.); 
   /* In the equal spacing case, this should average-out below-tolerance spacing differences  */ 
      
  	 if(hasequalspacing(Ttable,lenTtable))
  	 {  	 
      float dT = (Ttable[lenTtable-1] - Ttable[0])/(lenTtable-1.); 
      /* In the equal spacing case, this should average-out below-tolerance spacing differences  */     
      printf("drho: %f, dT: %f \n", drho,dT);
      for(i=0;i<NumPart2;i++)
      {
      /* edge cases: lower indices for rho, T interpolation = -1 
      -> rho, T of particle outside inteprolation bounds*/
      	 
	     lowerindex_rho = findindex_eqspacing(rho[i], rhotable[0], drho, lenrhotable); 
	     lowerindex_T = findindex_eqspacing(T[i], Ttable[0], dT, lenTtable);	   	     
	     
	    
	     
	     if(lowerindex_rho > -1 && lowerindex_T > -1)
	     {	    
	     	 emission[i] = bilinear_interpolation(rho[i], T[i], 
		      rhotable[lowerindex_rho], rhotable[lowerindex_rho + 1], Ttable[lowerindex_T], Ttable[lowerindex_T+1], 
		      emissiontable[lowerindex_T*lenrhotable + lowerindex_rho], emissiontable[(lowerindex_T +1)*lenrhotable + lowerindex_rho],
		      emissiontable[lowerindex_T*lenrhotable + lowerindex_rho + 1], emissiontable[(lowerindex_T+1)*lenrhotable + lowerindex_rho+1]);
	     }
	     
	     else if(lowerindex_rho == -1 && lowerindex_T > -1)
	     {
	     	 if((rho[i] < rhotable[0] && rhotable[0] < rhotable[1])||(rho[i] > rhotable[0] && rhotable[0] > rhotable[1]))
	     	   {lowerindex_rho = 0; }
	     	 else{lowerindex_rho = lenrhotable -1;}
    	 
	     	 emission[i] = linear_interpolation(T[i], Ttable[lowerindex_T], Ttable[lowerindex_T+1], \
	     	   emissiontable[lowerindex_T*lenrhotable + lowerindex_rho], emissiontable[(lowerindex_T+1)*lenrhotable + lowerindex_rho]);
	     }    
	     
	     else if(lowerindex_rho > -1 && lowerindex_T == -1)
	     {
	     	 if((T[i] < Ttable[0] && Ttable[0] < Ttable[1])||(T[i] > Ttable[0] && Ttable[0] > Ttable[1]))
	     	   {lowerindex_T = 0;}
	     	 else{lowerindex_T = lenTtable -1;}
     	 
	     	 emission[i] = linear_interpolation(rho[i], rhotable[lowerindex_rho], rhotable[lowerindex_rho+1], 
	     	   emissiontable[lowerindex_T*lenrhotable + lowerindex_rho], emissiontable[lowerindex_T*lenrhotable + lowerindex_rho+1]);
	     }   
	         
	     else if(lowerindex_rho == -1 && lowerindex_T == -1)
	     {
	     	 if((T[i] < Ttable[0] && Ttable[0] < Ttable[1])||(T[i] > Ttable[0] && Ttable[0] > Ttable[1]))
	     	   {lowerindex_T = 0;}
	     	 else{lowerindex_T = lenTtable -1;}
     	 
	     	 if((rho[i] < rhotable[0] && rhotable[0] < rhotable[1])||(rho[i] > rhotable[0] && rhotable[0] > rhotable[1]))
	     	   {lowerindex_rho = 0;}
	     	 else{lowerindex_rho = lenrhotable -1;}
	     	 
	     	 emission[i] = emissiontable[lowerindex_T*lenrhotable + lowerindex_rho];
	     } 
	     
	     else{printf("In interpolate_emdenssq: something has gone wrong with the interpolation for particle %lli, with log temperature %f and log density %f. \n",i,T[i],rho[i]);}
      } 
    }
    else
    {    	
      for(i=0;i<NumPart2;i++)
      {
      /* edge cases: lower indices for rho, T interpolation = -1 
      -> rho, T of particle outside inteprolation bounds*/
      	 
	     lowerindex_rho = findindex_eqspacing(rho[i], rhotable[0], drho, lenrhotable); 
	     lowerindex_T = findindex_neqspacing(T[i], Ttable, lenTtable);	
     
	     if(lowerindex_rho > -1 && lowerindex_T > -1)
	     {

	     	 emission[i] = bilinear_interpolation(rho[i], T[i], 
		      rhotable[lowerindex_rho], rhotable[lowerindex_rho + 1], Ttable[lowerindex_T], Ttable[lowerindex_T+1], 
		      emissiontable[lowerindex_T*lenrhotable + lowerindex_rho], emissiontable[(lowerindex_T +1)*lenrhotable + lowerindex_rho],
		      emissiontable[lowerindex_T*lenrhotable + lowerindex_rho + 1], emissiontable[(lowerindex_T+1)*lenrhotable + lowerindex_rho+1]);     
	     }
	     
	     else if(lowerindex_rho == -1 && lowerindex_T > -1)
	     {
	     	 if((rho[i] < rhotable[0] && rhotable[0] < rhotable[1])||(rho[i] > rhotable[0] && rhotable[0] > rhotable[1]))
	     	   {lowerindex_rho = 0;}
	     	 else{lowerindex_rho = lenrhotable -1;}
	     	 
	     	 emission[i] = linear_interpolation(T[i], Ttable[lowerindex_T], Ttable[lowerindex_T+1], \
	     	   emissiontable[lowerindex_T*lenrhotable + lowerindex_rho], emissiontable[(lowerindex_T+1)*lenrhotable + lowerindex_rho]);
	     }    
	     
	     else if(lowerindex_rho > -1 && lowerindex_T == -1)
	     {
	     	 if((T[i] < Ttable[0] && Ttable[0] < Ttable[1])||(T[i] > Ttable[0] && Ttable[0] > Ttable[1]))
	     	   {lowerindex_T = 0;}
	     	 else{lowerindex_T = lenTtable -1;}
	     	 
	     	 emission[i] = linear_interpolation(rho[i], rhotable[lowerindex_rho], rhotable[lowerindex_rho+1], 
	     	   emissiontable[lowerindex_T*lenrhotable + lowerindex_rho], emissiontable[lowerindex_T*lenrhotable + lowerindex_rho+1]);
	     }   
	         
	     else if(lowerindex_rho == -1 && lowerindex_T == -1)
	     {
	     	 if((T[i] < Ttable[0] && Ttable[0] < Ttable[1])||(T[i] > Ttable[0] && Ttable[0] > Ttable[1]))
	     	   {lowerindex_T = 0;}
	     	 else{lowerindex_T = lenTtable -1;}
	     	 
	     	 if((rho[i] < rhotable[0] && rhotable[0] < rhotable[1])||(rho[i] > rhotable[0] && rhotable[0] > rhotable[1]))
	     	   {lowerindex_rho = 0;}
	     	 else{lowerindex_rho = lenrhotable -1;}
   	 
	     	 emission[i] = emissiontable[lowerindex_T*lenrhotable + lowerindex_rho];
	     } 
	     
	     else{printf("In interpolate_emdenssq: something has gone wrong with the interpolation for particle %lli, with log temperature %f and log density %f. \n",i,T[i],rho[i]);}
      } 
    }    
  }
  else{
  	 if(hasequalspacing(Ttable,lenTtable))
  	 { 
      float dT = (Ttable[lenTtable-1] - Ttable[0])/(lenTtable-1.); 
      /* In the equal spacing case, this should average-out below-tolerance spacing differences  */       	 
  	  	 
      for(i=0;i<NumPart2;i++)
      {
      /* edge cases: lower indices for rho, T interpolation = -1 
      -> rho, T of particle outside inteprolation bounds*/
      	 
	     lowerindex_rho = findindex_neqspacing(rho[i], rhotable, lenrhotable); 
	     lowerindex_T = findindex_eqspacing(T[i], Ttable[0], dT, lenTtable);	
	     
	     if(lowerindex_rho > -1 && lowerindex_T > -1)
	     {
	     	 emission[i] = bilinear_interpolation(rho[i], T[i], 
		      rhotable[lowerindex_rho], rhotable[lowerindex_rho + 1], Ttable[lowerindex_T], Ttable[lowerindex_T+1], 
		      emissiontable[lowerindex_T*lenrhotable + lowerindex_rho], emissiontable[(lowerindex_T +1)*lenrhotable + lowerindex_rho],
		      emissiontable[lowerindex_T*lenrhotable + lowerindex_rho + 1], emissiontable[(lowerindex_T+1)*lenrhotable + lowerindex_rho+1]);
	     }
	     
	     else if(lowerindex_rho == -1 && lowerindex_T > -1)
	     {
	     	 if((rho[i] < rhotable[0] && rhotable[0] < rhotable[1])||(rho[i] > rhotable[0] && rhotable[0] > rhotable[1]))
	     	   {lowerindex_rho = 0;}
	     	 else{lowerindex_rho = lenrhotable -1;}
	     	 
	     	 emission[i] = linear_interpolation(T[i], Ttable[lowerindex_T], Ttable[lowerindex_T+1], \
	     	   emissiontable[lowerindex_T*lenrhotable + lowerindex_rho], emissiontable[(lowerindex_T+1)*lenrhotable + lowerindex_rho]);
	     }    
	     
	     else if(lowerindex_rho > -1 && lowerindex_T == -1)
	     {
	     	 if((T[i] < Ttable[0] && Ttable[0] < Ttable[1])||(T[i] > Ttable[0] && Ttable[0] > Ttable[1]))
	     	   {lowerindex_T = 0;}
	     	 else{lowerindex_T = lenTtable -1;}
	     	 
	     	 emission[i] = linear_interpolation(rho[i], rhotable[lowerindex_rho], rhotable[lowerindex_rho+1], 
	     	   emissiontable[lowerindex_T*lenrhotable + lowerindex_rho], emissiontable[lowerindex_T*lenrhotable + lowerindex_rho+1]);
	     }   
	         
	     else if(lowerindex_rho == -1 && lowerindex_T == -1)
	     {
	     	 if((T[i] < Ttable[0] && Ttable[0] < Ttable[1])||(T[i] > Ttable[0] && Ttable[0] > Ttable[1]))
	     	   {lowerindex_T = 0;}
	     	 else{lowerindex_T = lenTtable -1;}
	     	 
	     	 if((rho[i] < rhotable[0] && rhotable[0] < rhotable[1])||(rho[i] > rhotable[0] && rhotable[0] > rhotable[1]))
	     	   {lowerindex_rho = 0;}
	     	 else{lowerindex_rho = lenrhotable -1;}
	     	 
	     	 emission[i] = emissiontable[lowerindex_T*lenrhotable + lowerindex_rho];     
	     } 
	     
	     else{printf("In interpolate_emdenssq: something has gone wrong with the interpolation for particle %lli, with log temperature %f and log density %f. \n",i,T[i],rho[i]);} 	
      } 
    }
    else
    {    	
      for(i=0;i<NumPart2;i++)
      {
      /* edge cases: lower indices for rho, T interpolation = -1 
      -> rho, T of particle outside inteprolation bounds*/
      	 
	     lowerindex_rho = findindex_neqspacing(rho[i], rhotable, lenrhotable); 
	     lowerindex_T = findindex_neqspacing(T[i], Ttable, lenTtable);	
	     
	     if(lowerindex_rho > -1 && lowerindex_T > -1)
	     {
	     	 emission[i] = bilinear_interpolation(rho[i], T[i], 
		      rhotable[lowerindex_rho], rhotable[lowerindex_rho + 1], Ttable[lowerindex_T], Ttable[lowerindex_T+1], 
		      emissiontable[lowerindex_T*lenrhotable + lowerindex_rho], emissiontable[(lowerindex_T +1)*lenrhotable + lowerindex_rho],
		      emissiontable[lowerindex_T*lenrhotable + lowerindex_rho + 1], emissiontable[(lowerindex_T+1)*lenrhotable + lowerindex_rho+1]);
	     }
	     
	     else if(lowerindex_rho == -1 && lowerindex_T > -1)
	     {
	     	 if((rho[i] < rhotable[0] && rhotable[0] < rhotable[1])||(rho[i] > rhotable[0] && rhotable[0] > rhotable[1]))
	     	   {lowerindex_rho = 0;}
	     	 else{lowerindex_rho = lenrhotable -1;}
	     	 
	     	 emission[i] = linear_interpolation(T[i], Ttable[lowerindex_T], Ttable[lowerindex_T+1], \
	     	   emissiontable[lowerindex_T*lenrhotable + lowerindex_rho], emissiontable[(lowerindex_T+1)*lenrhotable + lowerindex_rho]);
	     }    
	     
	     else if(lowerindex_rho > -1 && lowerindex_T == -1)
	     {
	     	 if((T[i] < Ttable[0] && Ttable[0] < Ttable[1])||(T[i] > Ttable[0] && Ttable[0] > Ttable[1]))
	     	   {lowerindex_T = 0;}
	     	 else{lowerindex_T = lenTtable -1;}
	     	 
	     	 emission[i] = linear_interpolation(rho[i], rhotable[lowerindex_rho], rhotable[lowerindex_rho+1], 
	     	   emissiontable[lowerindex_T*lenrhotable + lowerindex_rho], emissiontable[lowerindex_T*lenrhotable + lowerindex_rho+1]);
	     }   
	         
	     else if(lowerindex_rho == -1 && lowerindex_T == -1)
	     {
	     	 if((T[i] < Ttable[0] && Ttable[0] < Ttable[1])||(T[i] > Ttable[0] && Ttable[0] > Ttable[1]))
	     	   {lowerindex_T = 0;}
	     	 else{lowerindex_T = lenTtable -1;}
	     	 
	     	 if((rho[i] < rhotable[0] && rhotable[0] < rhotable[1])||(rho[i] > rhotable[0] && rhotable[0] > rhotable[1]))
	     	   {lowerindex_rho = 0;}
	     	 else{lowerindex_rho = lenrhotable -1;}
	     	 
	     	 emission[i] = emissiontable[lowerindex_T*lenrhotable + lowerindex_rho];
	     } 
	     
	     else{printf("In interpolate_emdenssq: something has gone wrong with the interpolation for particle %lli, with log temperature %f and log density %f. \n",i,T[i],rho[i]);}
      } 
    }    
  }
  return 0;
}

/* use the interpolated emisivities with particle rho, hydrogen mass, element mass, and total mass 
Bertone et al. interpolation:
  luminosity = emissivity * m_gas,i/rho,i X_elt,i/X_sol 
solar abundances used to find the tables, emission scales with gas volume
emissivity  = previously calculated emission * n_H**2 */


 
 
