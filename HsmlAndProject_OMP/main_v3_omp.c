#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <time.h>

#ifndef SCHEDULE
#define SCHEDULE auto
#endif

// some functions use types in allvars -> keep this order
#include "allvars_v3_omp.h"
#include "proto_v3_omp.h"




/*
Nastasha Wijers added: 
- NOTREE option: turn off tree building and Hsml =/= 0 check if NOTREE is defined
- changed handling of smaller than grid cell smoothing lengths: project into max. 1 cell -> project into min. 1 cell
- NumPart: int -> long int
- OpenMP parallelisation for the projection loop 
*/

int findHsmlAndProject(long int _NumPart, struct particle_data* _P,
		       float* _Hsml, float* _Mass, float* _Quantity,
		       float _Xmin, float _Xmax, float _Ymin, float _Ymax, float _Zmin, float _Zmax,
		       int _Xpixels, int _Ypixels, int _DesDensNgb, int _Axis1, int _Axis2, int _Axis3,
		       float _Hmax, double _BoxSize, float* _Value, float* _ValueQuantity)

{
 
  NumPart = _NumPart;
  P = _P;
  Hsml = _Hsml;
  Mass = _Mass;
  Quantity = _Quantity;
  Xmin = _Xmin;
  Xmax = _Xmax;
  Ymin = _Ymin;
  Ymax = _Ymax;
  Zmin = _Zmin;
  Zmax = _Zmax;
  Xpixels = _Xpixels;
  Ypixels = _Ypixels;
  DesDensNgb =  _DesDensNgb;
  Axis1 = _Axis1;
  Axis2 = _Axis2;
  Axis3 = _Axis3;
  Hmax = _Hmax;
  BoxSize = _BoxSize;
  Value =         _Value;
  ValueQuantity = _ValueQuantity;

  fprintf(stdout, "NumPart = %li\n", NumPart);
  fprintf(stdout, "P = %p \n", P);
  fprintf(stdout, "Hsml = %p \n", Hsml);
  fprintf(stdout, "Mass = %p \n", Mass);
  fprintf(stdout, "Quantity = %p \n", Quantity);
  fprintf(stdout, "Xmin = %f, Xmax = %f \n", Xmin,Xmax);
  fprintf(stdout, "Ymin = %f, Ymax = %f \n", Ymin,Ymax);
  fprintf(stdout, "Zmin = %f, Zmax = %f \n", Zmin,Zmax);
  fprintf(stdout, "Xpixels = %d, Ypixels = %d \n", Xpixels,Ypixels);
  fprintf(stdout, "DesDensNgb = %d \n", DesDensNgb);
  fprintf(stdout, "Axis1 = %d, Axis2 = %d, Axis3 = %d\n", Axis1,Axis2,Axis3);
  fprintf(stdout, "Hmax = %f \n", Hmax);
  fprintf(stdout, "BoxSize = %g \n", BoxSize);
  fprintf(stdout, "Value = %p\n", Value);
  fprintf(stdout, "ValueQuantity = %p\n", ValueQuantity);

  fprintf(stdout, "\nKernel: ");
#ifdef SPH_KERNEL_GADGET
  fprintf(stdout, "Gadget (OWLS)\n\n");
#endif
#ifdef SPH_KERNEL_CUBIC
  fprintf(stdout, "cubic spline\n\n");
#endif
#ifdef SPH_KERNEL_QUARTIC
  fprintf(stdout, "quartic spline\n\n");
#endif
#ifdef SPH_KERNEL_QUINTIC
  fprintf(stdout, "quintic spline\n\n");
#endif
#ifdef SPH_KERNEL_C2
  fprintf(stdout, "Wendland C2\n\n");
#endif
#ifdef SPH_KERNEL_C4
  fprintf(stdout, "omp_get_num_threads();Wendland C4\n\n");
#endif
#ifdef SPH_KERNEL_C6
  fprintf(stdout, "Wendland C6\n\n");
#endif

/* Build tree and check smoothing lengths only if desired (if NOTREE, input smoothing lengths are used unchecked)*/
#ifndef NOTREE
  printf("peano-hilbert order...\n");
  peano_hilbert_order();
  printf("done\n");

  if (NumPart < 100000) {
    printf(" allocating memory for %li tree nodes\n",100*NumPart);
      tree_treeallocate(NumPart, NumPart);
  } else {
    printf(" allocating memory for %li tree nodes\n",100*NumPart);
      tree_treeallocate(10*NumPart, NumPart);
    }
  //Softening = (Xmax-Xmin)/Xpixels/100.;
  Softening = (Xmax-Xmin)/pow(NumPart,0.3333)/10.0;

    printf("build tree...\n");
    tree_treebuild();
    printf("done.\n");

    printf("finding neighbours...\n");
    determine_hsml();
    printf("done.\n");

    tree_treefree();
  
#endif
  
    printf("projecting\n");
    make_map();
    printf("done\n");
    
    // find min/max of arrays
    float minval = 1.0e20;
    float maxval = -minval;
    float minvalQ = 1.0e20;
    float maxvalQ = -minvalQ;
    int nelements = Xpixels*Ypixels;
    int i;
    for (i = 0; i < nelements; i++)
      {
	if (Value[i] > maxval)
	  {
	    maxval = Value[i];
	  }
	else if (Value[i] < minval)
	  {
	    minval = Value[i];
	  }

	if (ValueQuantity[i] > maxvalQ)
	  {
	    maxvalQ = ValueQuantity[i];
	  }
	else if (ValueQuantity[i] < minvalQ)
	  {
	    minvalQ = ValueQuantity[i];
	  }

      }    
    
    fprintf(stdout, "min(Value) = %f, max(Value) = %f\n", minval,maxval);
    fprintf(stdout, "min(ValueQuantity) = %f, max(ValueQuantity) = %f\n", minvalQ,maxvalQ);


  return 0;
}

void determine_hsml(void)
{
  int i, signal;
  double h;

  for(i = 0, signal = 0, h = 0; i < NumPart; i++)
    {
      if(i > (signal / 100.0) * NumPart)
        {
          printf("x");
          fflush(stdout);
          signal++;
        }

      if(Hsml[i] == 0)
        Hsml[i] = h  = ngb_treefind(P[i].Pos, DesDensNgb, h * 1.1);
    }

  printf("\n");
}

#ifdef PERIODIC
#define NGB_PERIODIC(x) (((x)>BoxHalf)?((x)-BoxSize):(((x)<-BoxHalf)?((x)+BoxSize):(x)))
#else
#define NGB_PERIODIC(x) (x)
#endif


/* ---------- set up initialiser and combiner for DOREDUCTION ---------  
  for the array reduction in the loop; no check for array length matching
  has output to match reduction syntax; modification is in-place
  handles freeing of memory of the array that is no longer needed */ 

#if defined(_OPENMP) && defined (DOREDUCTION)

struct arr_len arrayadd(struct arr_len out,struct arr_len in){
   long int ind;
   for(ind=0; ind < out.len; ind++){
     out.a[ind] += in.a[ind];   
   }
   free(in.a); //clean up after array is no longer needed 
   //printf("One copy freed\n");
   return out;
}

// initialise to zeros; use the original copy of the variable to get the array length
struct arr_len arrayinit(struct arr_len toinit, struct arr_len tocopy){
	toinit.a = malloc(tocopy.len*sizeof(float));
   long int ind;
   for(ind=0; ind < tocopy.len; ind++){
     toinit.a[ind] =0.0f;   
   }
   toinit.len = tocopy.len;
   //printf("One copy made\n");
   return toinit;
}
#endif //end of DOREDUCTION


void make_map(void)
{ 
  /* ----------- do the setup for parallel stuff and timing ------------*/
  clock_t starttime = clock();
  
  #ifdef _OPENMP
  double starttime_omp = omp_get_wtime();
  
  // set up reduction operation: initiliser and combiner are defined above
  #ifdef DOREDUCTION
  #pragma omp declare reduction \
    (redarrayadd:struct arr_len:omp_out=arrayadd(omp_out,omp_in)) \
    initializer(omp_priv = arrayinit(omp_priv,omp_orig))

  struct arr_len valtemp, valqtemp; //initialise as shared variables, outside parallel region   
  #endif //end of DOREDUCTION 
   
 
  // set up locks (initialised in parallel region)
  #ifdef DOLOCKS
  omp_lock_t *locks;
  #ifdef ARRCHUNKPIX
  //ARRCHUNKPIX does not need to divide anything exactly; the edge region may just be a bit thinner
  const int xchunks = Xpixels/ARRCHUNKPIX; 
  const int ychunks = Ypixels/ARRCHUNKPIX; 
  locks = malloc(xchunks*ychunks*sizeof(omp_lock_t));
  if(!locks){
    printf("Malloc for lock array failed.\n");        
  }  
  #else
  locks = malloc(Xpixels*Ypixels*sizeof(omp_lock_t));
  if(!locks){
    printf("Malloc for lock array failed.\n");        
  }
  #endif // end of ARRCHUNKPIX  
  #endif // end of DOLOCKS

  
  #endif // end of _OPENMP 
  
/* ----------------------------------------------------------------------------
   ------------------------------ PARALLEL REGION -----------------------------
   --------------------------------------------------------------------------*/  
  #ifdef _OPENMP
  #pragma omp parallel 
  {
  int size = omp_get_num_threads();
  int rank = omp_get_thread_num();
  if(rank==0){
    printf("Running make_map with OpenMP on %i cores\n",size);  
  }
  #else
  printf("Running make_map serially\n");
  { //match opening bracket for parallel block
  #endif 
  
  // declared in parallel region -> private
  int i, j;
  int dx, dy, nx, ny;
  double h, r, u, wk, hinv, hinv3;
  //double pos[2];
  //double LengthX;
  //double LengthY;
  double r2, h2;
  // double hmin, hmax
  double sum, x, y, xx, yy, xxx, yyy;
  //double pixelsizeX, pixelsizeY;
  
  /* Per-thread part of the loop set-ups */
  
  //initialise locks (arrays allocated before the loop)
  #ifdef DOLOCKS
  long unsigned int l;  
  #ifndef ARRCHUNKPIX
  #pragma omp for schedule(static)
  for(l=0; l<Xpixels*Ypixels; l++){
    omp_init_lock(&locks[l]);  
  }
  #else
  #pragma omp for schedule(static)
  for(l=0; l<xchunks*ychunks; l++){
    omp_init_lock(&locks[l]);  
  }
  #endif // end of ARRCHUNKPIX  
  #endif //end of DOLOCKS

    
  // initialise input array to zeros  
  #if defined(_OPENMP) && defined(DOALLPAR)
  #pragma omp for schedule(static)
  #endif
  for(i = 0; i < Xpixels; i++)
    for(j = 0; j < Ypixels; j++)
      {
        Value[i * Ypixels + j] = 0;
        ValueQuantity[i * Ypixels + j] = 0;
      }
 
  // initialise DOREDUCTION structs
  #if defined(DOREDUCTION) && defined(_OPENMP) 
  #pragma omp single // shared variable, just get this right
  {
    valtemp.a = Value;
    valtemp.len = Xpixels*Ypixels;
    valqtemp.a = ValueQuantity;
    valqtemp.len = Xpixels*Ypixels;  
  }
   #endif //end of DOREDUCTION
   
 
  /* set up variables for the loop */ 
 
  BoxHalf = 0.5 * BoxSize; 
 
  const double LengthX = Xmax-Xmin;
  const double LengthY = Ymax-Ymin;

  Xc = 0.5 * (Xmax + Xmin);
  Yc = 0.5 * (Ymax + Ymin);

  const double pixelsizeX = LengthX / Xpixels;
  const double pixelsizeY = LengthY / Ypixels;
  
  const double invpixelsizeX = 1.0 / pixelsizeX;
  const double invpixelsizeY = 1.0 / pixelsizeY;

  /* if(pixelsizeX < pixelsizeY)
    hmin = 1.001 * 1.4142135623730951 *pixelsizeX / 2;
  else
    hmin = 1.001 * 1.4142135623730951 *pixelsizeY / 2; */
  const double hmin = 1.001*sqrt(pixelsizeX*pixelsizeX + pixelsizeY*pixelsizeY)/2.;  

  const double hmax = Hmax;


  //OMP for loop: each process projects a subset of the particles, with OpenMP figuring out the subsets
  long int n;
  #ifdef _OPENMP
  #ifdef DOREDUCTION
  #pragma omp for reduction(redarrayadd:valtemp,valqtemp) schedule(SCHEDULE)
  #else
  #pragma omp for schedule(SCHEDULE)
  #endif //end of doreduction 
  #endif //end of _OPENMP
  for(n = 0; n < NumPart; n++)
    {
   /*   if((n % (NumPart / 100)) == 0)
	{
    printf(".");
	  fflush(stdout);
	}
	*/

      
      if(P[n].Pos[Axis3]< Zmin || P[n].Pos[Axis3] > Zmax)
        continue;

      double pos[2];
      
      pos[0]= P[n].Pos[Axis1]-Xmin;
      pos[1]= P[n].Pos[Axis2]-Ymin;


      h = Hsml[n];
      
      if(h < hmin)
        h = hmin;

      if(h > hmax)
        h = hmax;

#ifdef PERIODIC
      if((NGB_PERIODIC(Xc - P[n].Pos[Axis1]) + 0.5 * (Xmax - Xmin)) < -Hsml[n])
        continue;
      if((NGB_PERIODIC(Xc - P[n].Pos[Axis1]) - 0.5 * (Xmax - Xmin)) > Hsml[n])
        continue;
      
      if((NGB_PERIODIC(Yc - P[n].Pos[Axis2]) + 0.5 * (Ymax - Ymin)) < -Hsml[n])
        continue;
      if((NGB_PERIODIC(Yc - P[n].Pos[Axis2]) - 0.5 * (Ymax - Ymin)) > Hsml[n])
        continue;
#else
      if(pos[0] + h < 0 || pos[0] - h >  LengthX
         || pos[1] + h < 0 || pos[1] - h > LengthY)
        continue;
#endif
 
      h2 = h * h;
      hinv =  1.0 / h;
      hinv3 = hinv * hinv * hinv;

      nx = h * invpixelsizeX + 1;
      ny = h * invpixelsizeY + 1;

      /* x,y central pixel of region covered by the particle on the mesh */
      
      x = (floor(pos[0] * invpixelsizeX) + 0.5) * pixelsizeX;
      y = (floor(pos[1] * invpixelsizeY) + 0.5) * pixelsizeY;

      /* determine kernel normalizaton */

      
      sum = 0;

      
      for(dx = -nx; dx <= nx; dx++)
        for(dy = -ny; dy <= ny; dy++)
          {
            xx = x + dx * pixelsizeX - pos[0];
            yy = y + dy * pixelsizeY - pos[1];
            r2 = xx * xx + yy * yy;
            
            if(r2 < h2)
              {
                r = sqrt(r2);
                
                u = r *hinv;
                
                wk = kernel_wk(u, hinv3);

                //if(u < 0.5)
                //  wk = (2.546479089470 + 15.278874536822 * (u - 1) * u * u);
                //else
                //  wk = 5.092958178941 * (1.0 - u) * (1.0 - u) * (1.0 - u);
                
                sum += wk;
              }
          }
      
      if(sum < 1.0e-10)
        continue;

      for(dx = -nx; dx <= nx; dx++)
        for(dy = -ny; dy <= ny; dy++)
          {
            xxx = x + dx * pixelsizeX;
            yyy = y + dy * pixelsizeY;

#ifdef PERIODIC
            xxx = NGB_PERIODIC(xxx + Xmin - Xc) + Xc - Xmin;
            yyy = NGB_PERIODIC(yyy + Ymin - Yc) + Yc - Ymin;
#endif
           
            if(xxx >= 0 && yyy >= 0)
              {
                i = xxx * invpixelsizeX;
                j = yyy * invpixelsizeY;
                
                // i, j >= 0 if xxx, yyy are, and the pixel size is positive                
                //if(i >= 0 && i < Xpixels)
                  //if(j >= 0 && j < Ypixels)
                if(i < Xpixels && j < Ypixels)
                    {
                      xx = x + dx * pixelsizeX - pos[0];
                      yy = y + dy * pixelsizeY - pos[1];
                      r2 = xx * xx + yy * yy;
                      
                      if(r2 < h2)
                        {
                          r = sqrt(r2);
                          u = r * hinv;

                          wk = kernel_wk(u, hinv3); 

                          //if(u < 0.5)
                          //  wk = (2.546479089470 + 15.278874536822 * (u - 1) * u * u);
                          //else
                          //  wk = 5.092958178941 * (1.0 - u) * (1.0 - u) * (1.0 - u);

                          // set lock
                          #if !(defined(_OPENMP) && defined(DOREDUCTION))

                          #if defined(_OPENMP) && defined(DOLOCKS)
                          #ifdef ARRCHUNKPIX
                          omp_set_lock(&locks[i/ARRCHUNKPIX * ychunks + j/ARRCHUNKPIX]);
                          #else
                          omp_set_lock(&locks[i * Ypixels + j]);
                          #endif //end of ARRCHUNKPIX
                          #endif //end of DOLOCKS

                          #if defined(_OPENMP) && defined(DOATOMIC)
                          #pragma omp atomic // adds to global output array; atomic should prevent issues from different concurrent writes
                          #endif //end of DOATOMIC
                          Value[i * Ypixels + j] += Mass[n] * wk / sum; // done unless DOREDUCTION
                          #if defined(_OPENMP) && defined(DOATOMIC)
                          #pragma omp atomic
                          #endif //end of DOATOMIC
                          ValueQuantity[i * Ypixels + j] += Mass[n]*Quantity[n]*wk / sum; // done unless DOREDUCTION

                          // release lock
                          #if defined(_OPENMP) && defined(DOLOCKS)
                          #ifdef ARRCHUNKPIX
                          omp_unset_lock(&locks[i/ARRCHUNKPIX * ychunks + j/ARRCHUNKPIX]);
                          #else
                          omp_unset_lock(&locks[i * Ypixels + j]);
                          #endif //end of ARRCHUNKPIX
                          #endif //end of DOLOCKS

                          #else //DOREDUCTION
                          valtemp.a[i * Ypixels + j] += Mass[n] * wk / sum;
                          valqtemp.a[i * Ypixels + j] += Mass[n]*Quantity[n]*wk / sum;
                          #endif //end of DOREDUCTION
                        } //end of if( pixel in kernel support)
                    } // end of if( pixel indices in projection domain)
              } //end of if( pixel grid coordinates >= 0)
          } //end of loop over square containing kernel support
    } //end of particle (NumPart) loop
    
  /* clean-up for parallel stuff*/

  // clean up locks
  #ifdef DOLOCKS
  #ifdef ARRCHUNKPIX
  #pragma omp for schedule(static)
  for(l=0; l<xchunks*ychunks; l++){
    omp_destroy_lock(&locks[l]);  
  }
  #else
  #pragma omp for schedule(static)
  for(l=0; l<Xpixels*Ypixels; l++){
    omp_destroy_lock(&locks[l]);  
  }
  #endif // end of ARRCHUNKPIX    
  #pragma omp single
  { // only free once 
    free(locks);  
  }
  #endif // end of DOLOCKS
  

  // for DOREDUCTION, the valtemp and valqtemp shared variables hold the same pointers as Value and ValueQuantitiy


  #if defined(_OPENMP) && defined(DOALLPAR)
  #pragma omp for schedule(static)
  #elif !defined(DOALLPAR) && defined(_OPENMP)// end parallel here; rest is serial
  } /* end of OMP parallel */ 
  int i,j;
  #endif 
  for(i = 0; i < Xpixels; i++){
    for(j = 0; j < Ypixels; j++){
      if(Value[i * Ypixels + j]>0){
        ValueQuantity[i * Ypixels + j] /= Value[i * Ypixels + j];
      }
    }
  }
  #if defined(DOALLPAR) || !defined(_OPENMP) 
  } /* end of OMP parallel */ 
  #endif
  /* ----------------------------------------------------------------------------
   ------------------------ END OF PARALLEL REGION ------------------------------
   --------------------------------------------------------------------------*/ 
  printf("\n");
  
  clock_t endtime = clock();  
  #ifdef _OPENMP
  double endtime_omp = omp_get_wtime();
  printf("Total time for HsmlAndProject loop execution (omp_get_wtime): %g s\n", endtime_omp-starttime_omp);
  #endif  
  printf("Total time for HsmlAndProject loop execution: %g s\n",((float)(endtime-starttime))/CLOCKS_PER_SEC);

  
}

