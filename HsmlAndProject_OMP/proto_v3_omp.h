
#include <malloc.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "allvars_v3_omp.h"

double kernel_wk(double u, double hinv3);

#if defined(_OPENMP) && defined(DOREDUCTION)
struct arr_len arrayadd(struct arr_len out,struct arr_len in);
struct arr_len arrayinit(struct arr_len toinit, struct arr_len tocopy);
#endif //end of DOREDUCTION

void determine_hsml(void);
void make_map(void);
int findHsmlAndProject(long int _NumPart, struct particle_data* _P,
		       float* _Hsml, float* _Mass, float* _Quantity,
		       float _Xmin, float _Xmax, float _Ymin, float _Ymax, float _Zmin, float _Zmax,
		       int _Xpixels, int _Ypixels, int _DesDensNgb, int _Axis1, int _Axis2, int _Axis3,
		       float _Hmax, double _BoxSize, float* _Value, float* _ValueQuantity);  /*int argc, void *argv[]);*/
void peano_hilbert_order(void);

float ngb_treefind(float xyz[3], int desngb, float hguess);

size_t tree_treeallocate(int maxnodes, int maxpart);
void tree_treefree(void);

void endrun(int ierr);
void free_particle_data(void);


int get_next_file(int begsnap, int begfilenr, int endsnap, int endfilenr, 
		  int *snapshot, int *filenr);

int compare_grp_particles(const void *a, const void *b);
int compare_dens(const void *a, const void *b);
int compare_grp_particles(const void *a, const void *b);
int compare_energy(const void *a, const void *b);

void unbind(int lev, int head, int len);
double tree_treeevaluate_potential(int target);

void save_hsml(void);

void check(int i, double h);

void process_group(int gr);

int load_hash_table(void);
void load_group_catalogue(void);
void mark_hash_cells(void);
void load_particle_data(void);
void find_group_indices(void);
int id_sort_compare_id(const void *a, const void *b);

void determine_densities(void);


int tree_treebuild(void);

void tree_update_node_recursive(int no, int sib, int father);


void *mymalloc(size_t n);
void myfree(void *p);

void set_units(void);




int ngb_compare_key(const void *a, const void *b);
float ngb_treefind(float xyz[3], int desngb, float hguess);
int ngb_treefind_variable(float searchcenter[3], float hguess);


void read_parameter_file(char *);
