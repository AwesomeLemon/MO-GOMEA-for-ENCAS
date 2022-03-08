/**
 * Comment from Alexander Chebykin (the original header starts with "MO_GOMEA.c"):
 * In my humble opinion, source code files should not be 4000 lines long.
 * However, that's what the official implementation is like, and I never found the time or courage to refactor it.
 * I introduced the following changes:
 * 1) using fitness functions written in Python. Note that it's not very general (i.e. restricted to my NAS use cases)
 * 2) each variable can take more than 2 values, and different variables may have different alphabet sizes
 * 3) setting random seed to a specific value
 * 4) executing a precise (user-provided) amount of evaluations.
 * P.S. "ezilaitini" is "initialize" backwards - it took me some time to figure that out
 *
 * MO_GOMEA.c
 *
 * IN NO EVENT WILL THE AUTHORS OF THIS SOFTWARE BE LIABLE TO YOU FOR ANY
 * DAMAGES, INCLUDING BUT NOT LIMITED TO LOST PROFITS, LOST SAVINGS, OR OTHER
 * INCIDENTIAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR THE INABILITY
 * TO USE SUCH PROGRAM, EVEN IF THE AUTHOR HAS BEEN ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGES, OR FOR ANY CLAIM BY ANY OTHER PARTY. THE AUTHOR MAKES NO
 * REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE SOFTWARE, EITHER
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT. THE
 * AUTHOR SHALL NOT BE LIABLE FOR ANY DAMAGES SUFFERED BY ANYONE AS A RESULT OF
 * USING, MODIFYING OR DISTRIBUTING THIS SOFTWARE OR ITS DERIVATIVES.
 *
 * Multi-Objective Gene-pool Optimal Mixing Evolutionary Algorithm with IMS
 *
 * In this implementation, maximization is assumed.
 *
 * The software in this file is the result of (ongoing) scientific research.
 * The software has been constructed based on
 * Linkage Tree Genetic Algorithm (LTGA) and
 * Multi-objective Adapted Maximum-Likelihood Gaussian Model Iterated Density 
 * Estimation Evolutionary Algorithm (MAMaLGaM)
 *
 * Interested readers can refer to the following publications for more details:
 *
 * 1. N.H. Luong, H. La Poutré, and P.A.N. Bosman: Multi-objective Gene-pool 
 * Optimal Mixing Evolutionary Algorithms with the Interleaved Multi-start Scheme. 
 * In Swarm and Evolutionary Computation, vol. 40, June 2018, pages 238-254, 
 * Elsevier, 2018.
 * 
 * 2. N.H. Luong, H. La Poutré, and P.A.N. Bosman: Multi-objective Gene-pool 
 * Optimal Mixing Evolutionary Algorithms. In Dirk V. Arnold, editor,
 * Proceedings of the Genetic and Evolutionary Computation Conference GECCO 2014: 
 * pages 357-364, ACM Press New York, New York, 2014.
 *
 * 3. P.A.N. Bosman and D. Thierens. More Concise and Robust Linkage Learning by
 * Filtering and Combining Linkage Hierarchies. In C. Blum and E. Alba, editors,
 * Proceedings of the Genetic and Evolutionary Computation Conference -
 * GECCO-2013, pages 359-366, ACM Press, New York, New York, 2013. 
 *
 * 4. P.A.N. Bosman. The anticipated mean shift and cluster registration 
 * in mixture-based EDAs for multi-objective optimization. In M. Pelikan and
 * J. Branke, editors, Proceedings of the Genetic and Evolutionary Computation 
 * GECCO 2010, pages 351-358, ACM Press, New York, New York, 2010.
 * 
 * 5. J.C. Pereira, F.G. Lobo: A Java Implementation of Parameter-less 
 * Evolutionary Algorithms. CoRR abs/1506.08694 (2015)
 */

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Includes -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <limits.h>
#include <time.h>
#include <sys/time.h>
#include <float.h>

#include <Python.h>
//#include <stdbool.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Constants -=-=-=-=-=-=-=-=-=-=-=-=-=-*/
#define FALSE 0
#define TRUE 1

#define NOT_EXTREME_CLUSTER -1

#define MINIMIZATION 1
#define MAXIMIZATION 2

#define ZEROMAX_ONEMAX 0
#define TRAP5 1
#define KNAPSACK 2
#define LOTZ 3
#define MAXCUT 4
#define PYTHON_FUNCTION 5

/*-=-=-=-=-=-=-=-=-=-=-=-= Section Header Functions -=-=-=-=-=-=-=-=-=-=-=-=*/
/*---------------------------- Utility Functions ---------------------------*/
void *Malloc( long size );
void initializeRandomNumberGenerator();
double randomRealUniform01( void );
int randomInt( int maximum );
double log2( double x );
int* createRandomOrdering(int size_of_the_set);
double distanceEuclidean( double *x, double *y, int number_of_dimensions );

int *mergeSort( double *array, int array_size );
void mergeSortWithinBounds( double *array, int *sorted, int *tosort, int p, int q );
void mergeSortMerge( double *array, int *sorted, int *tosort, int p, int r, int q );
int almosteq(double a, double b);
/*-------------------------Interpret Command Line --------------------------*/
void interpretCommandLine( int argc, char **argv );
void parseCommandLine( int argc, char **argv );
void parseOptions( int argc, char **argv, int *index );
void printAllInstalledProblems( void );
void optionError( char **argv, int index );
void parseParameters( int argc, char **argv, int *index );
void printUsage( void );
void checkOptions( void );
void printVerboseOverview( void );
/*--------------- Load Problem Data and Solution Evaluations ---------------*/
void evaluateIndividual(int *solution, double *obj, double *con, int objective_index_of_extreme_cluster);
char *installedProblemName( int index );
int numberOfInstalledProblems( void );

void onemaxLoadProblemData();
void trap5LoadProblemData();
void lotzLoadProblemData();
void onemaxProblemEvaluation(char *solution, double *obj_values, double *con_value, int objective_index_of_extreme_cluster);
double deceptiveTrapKTightEncodingFunctionProblemEvaluation( char *parameters, int k, char is_one );
void trap5ProblemEvaluation(char *solution, double *obj_values, double *con_value, int objective_index_of_extreme_cluster);
void lotzProblemEvaluation(char *solution, double *obj_values, double *con_value, int objective_index_of_extreme_cluster);

void knapsackLoadProblemData();
void ezilaitiniKnapsackProblemData();
void knapsackSolutionRepair(char *solution, double *solution_profits, double *solution_weights, double *solution_contraint, int objective_index_of_extreme_cluster);
void knapsackSolutionSingleObjectiveRepair(char *solution, double *solution_profits, double *solution_weights, double *solution_constraint, int objective_index);
void knapsackSolutionMultiObjectiveRepair(char *solution, double *solution_profits, double *solution_weights, double *solution_constraint);
void knapsackProblemEvaluation(char *solution, double *obj_values, double *con_value, int objective_index_of_extreme_cluster);

void maxcutLoadProblemData();
void ezilaitiniMaxcutProblemData();
void maxcutReadInstanceFromFile(char *filename, int objective_index);
void maxcutProblemEvaluation( char *solution, double *obj_values, double *con_value, int objective_index_of_extreme_cluster );

double **getDefaultFrontOnemaxZeromax( int *default_front_size );
double **getDefaultFrontTrap5InverseTrap5( int *default_front_size );
double **getDefaultFrontLeadingOneTrailingZero( int *default_front_size );
short haveDPFSMetric( void );
double **getDefaultFront( int *default_front_size );
double computeDPFSMetric( double **default_front, int default_front_size, double **approximation_front, int approximation_front_size );

void pythonFunctionLoadProblemData();
void pythonFunctionProblemEvaluation(int *solution, double *obj_values, double *con_value, int objective_index_of_extreme_cluster);
/*---------------------------- Tracking Progress ---------------------------*/
void writeGenerationalStatistics();
void writeCurrentElitistArchive( char final );
void logElitistArchiveAtSpecificPoints();
char checkTerminationCondition();
char checkNumberOfEvaluationsTerminationCondition();
char checkVTRTerminationCondition();
void logNumberOfEvaluationsAtVTR();
/*---------------------------- Elitist Archive -----------------------------*/
char isDominatedByElitistArchive( double *obj, double con, char *is_new_nondominated_point, int *position_of_existed_member );
short sameObjectiveBox( double *objective_values_a, double *objective_values_b );
int hammingDistanceInParameterSpace(int *solution_1, int *solution_2);
int hammingDistanceToNearestNeighborInParameterSpace(int *solution, int replacement_position);
void updateElitistArchive( int *solution, double *solution_objective_values, double solution_constraint_value );
void updateElitistArchiveWithReplacementOfExistedMember( int *solution, double *solution_objective_values, double solution_constraint_value, char *is_new_nondominated_point, char *is_dominated_by_archive);
void removeFromElitistArchive( int *indices, int number_of_indices );
short isInListOfIndices( int index, int *indices, int number_of_indices );
void addToElitistArchive( int *solution, double *solution_objective_values, double solution_constraint_value);
void adaptObjectiveDiscretization( void );
/*-------------------------- Solution Comparision --------------------------*/
char betterFitness( double *objective_value_x, double constraint_value_x, double *objective_value_y, double constraint_value_y, int objective_index );
char equalFitness( double *objective_value_x, double constraint_value_x, double *objective_value_y, double constraint_value_y, int objective_index );
short constraintParetoDominates( double *objective_values_x, double constraint_value_x, double *objective_values_y, double constraint_value_y );
short constraintWeaklyParetoDominates( double *objective_values_x, double constraint_value_x, double *objective_values_y, double constraint_value_y );
short paretoDominates( double *objective_values_x, double *objective_values_y );
short weaklyParetoDominates( double *objective_values_x, double *objective_values_y );
/*-------------------------- Linkage Tree Learning --------------------------*/
void learnLinkageTree( int cluster_index );
double *estimateParametersForSingleBinaryMarginal(  int cluster_index, int *indices, int number_of_indices, int *factor_size );
double *estimateParametersForArbitraryPairwiseMarginal(  int cluster_index, int *indices, int number_of_indices, int *factor_size );
int determineNearestNeighbour( int index, double **S_matrix, int mpm_length );
void printLTStructure( int cluster_index );
/*-------------------------------- MO-GOMEA --------------------------------*/
void initialize();
void initializeMemory();
void initializePopulationAndFitnessValues();
void computeObjectiveRanges( void );

void learnLinkageOnCurrentPopulation();
int** clustering(double **objective_values_pool, int pool_size, int number_of_dimensions, 
                    int number_of_clusters, int *pool_cluster_size );
int *greedyScatteredSubsetSelection( double **points, int number_of_points, int number_of_dimensions, int number_to_select );
void determineExtremeClusters();
void initializeClusters();
void ezilaitiniClusters();

void improveCurrentPopulation( void );
void copyValuesFromDonorToOffspring(int *solution, int *donor, int cluster_index, int linkage_group_index);
void copyFromAToB(int *solution_a, double *obj_a, double con_a, int *solution_b, double *obj_b, double *con_b);
void mutateSolution(int *solution, int lt_factor_index, int cluster_index);
void performMultiObjectiveGenepoolOptimalMixing( int cluster_index, int *parent, double *parent_obj, double parent_con,
                                            int *result, double *obj, double *con );
void performSingleObjectiveGenepoolOptimalMixing( int cluster_index, int objective_index, 
                                int *parent, double *parent_obj, double parent_con,
                                int *result, double *obj, double *con);

void selectFinalSurvivors();
void freeAuxiliaryPopulations();
/*-------------------------- Parameter-less Scheme -------------------------*/
void initializeMemoryForArrayOfPopulations();
void putInitializedPopulationIntoArray();
void assignPointersToCorrespondingPopulation();
void ezilaitiniArrayOfPopulation();
void ezilaitiniMemoryOfCorrespondingPopulation();
void schedule_runMultiplePop_clusterPop_learnPop_improvePop();
void schedule();

void initializeCommonVariables();
void ezilaitiniCommonVariables();
void loadProblemData();
void ezilaitiniProblemData();
void run();
int main( int argc, char **argv );
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=- Section Global Variables -=-=-=-=-=-=-=-=-=-=-=-=-*/
int    **population,               /* The population containing the solutions. */
        ***array_of_populations,    /* The array containing all populations in the parameter-less scheme. */
        **offspring,                /* Offspring solutions. */
        **elitist_archive,          /* Archive of elitist solutions. */
        **elitist_archive_copy;     /* Copy of the elitist archive. */

int     problem_index,                          /* The index of the optimization problem. */
        number_of_parameters,                   /* The number of parameters to be optimized. */
        number_of_generations,                  /* The current generation count. */
        *array_of_number_of_generations,        /* The array containing generation counts of all populations in the parameter-less scheme.*/
        generation_base,                        /* The number of iterations that population of size N_i is run before running 1 iteration of population of size N_(i+1). */
        population_size,                        /* The size of each population. */
        *array_of_population_sizes,             /* The array containing population sizes of all populations in the parameter-less scheme. */
        smallest_population_size,               /* The size of the first population. */
        population_id,                          /* The index of the population that is currently operating. */
        offspring_size,                         /* The size of the offspring population. */

        number_of_objectives,                   /* The number of objective functions. */
        elitist_archive_size,                   /* Number of solutions in the elitist archive. */
        elitist_archive_size_target,            /* The lower bound of the targeted size of the elitist archive. */
        elitist_archive_copy_size,              /* Number of solutions in the elitist archive copy. */
        elitist_archive_capacity,               /* Current memory allocation to elitist archive. */
        number_of_mixing_components,            /* The number of components in the mixture distribution. */
        *array_of_number_of_clusters,           /* The array containing the number-of-cluster of each population in the parameter-less scheme. */
        *population_cluster_sizes,              /* The size of each cluster. */
        **population_indices_of_cluster_members,/* The corresponding index in the population of each solution in a cluster. */
        *which_extreme,                         /* The corresponding objective of an extreme-region cluster. */
        
        t_NIS,                          /* The number of subsequent generations without an improvement (no-improvement-stretch). */
        *array_of_t_NIS,                /* The array containing the no-improvement-stretch of each population in the parameter-less scheme. */
        maximum_number_of_populations,  /* The maximum number of populations that can be run (depending on memory budget). */
        number_of_populations,          /* The number of populations that have been initialized. */

        **mpm,                          /* The marginal product model. */
        *mpm_number_of_indices,         /* The number of variables in each factor in the mpm. */
        mpm_length,                     /* The number of factors in the mpm. */

        ***lt,                          /* The linkage tree, one for each cluster. */
        **lt_number_of_indices,         /* The number of variables in each factor in the linkage tree of each cluster. */
        *lt_length;                     /* The number of factors in the linkage tree of each cluster. */
	      
long    number_of_evaluations,            /* The current number of times a function evaluation was performed. */
        log_progress_interval,            /* The interval (in terms of number of evaluations) at which the elitist archive is logged. */
		maximum_number_of_evaluations,    /* The maximum number of evaluations. */
        *array_of_number_of_evaluations_per_population; /* The array containing the number of evaluations used by each population in the parameter-less scheme. */
		
double  **objective_values,                 /* Objective values for population members. */
        ***array_of_objective_values,       /* The array containing objective values of all populations in the parameter-less scheme. */
        *constraint_values,                 /* Constraint values of population members. */
        **array_of_constraint_values,       /* The array containing constraint values of all populations in the parameter-less scheme. */
        
        **objective_values_offspring,       /* Objective values of offspring solutions. */
        *constraint_values_offspring,       /* Constraint values of offspring solutions. */
                 
        **elitist_archive_objective_values,         /* Objective values of solutions stored in elitist archive. */
        **elitist_archive_copy_objective_values,    /* Copy of objective values of solutions stored in elitist archive. */
        *elitist_archive_constraint_values,         /* Constraint values of solutions stored in elitist archive. */
        *elitist_archive_copy_constraint_values,    /* Copy of constraint values of solutions stored in elitist archive. */

        *objective_ranges,                          /* Ranges of objectives observed in the current population. */
        **array_of_objective_ranges,                /* The array containing ranges of objectives observed in each population in the parameter-less scheme. */
        **objective_means_scaled,                   /* The means of the clusters in the objective space, linearly scaled according to the observed ranges. */
        *objective_discretization,                  /* The length of the objective discretization in each dimension (for the elitist archive). */
        vtr,                              /* The value-to-reach (in terms of Inverse Generational Distance). */
        **MI_matrix;                      /* Mutual information between any two variables */

int64_t random_seed,                      /* The seed used for the random-number generator. */
        random_seed_changing;             /* Internally used variable for randomly setting a random seed. */

char    use_pre_mutation,                   /* Whether to use weak mutation. */
        use_pre_adaptive_mutation,          /* Whether to use strong mutation. */
        use_print_progress_to_screen,       /* Whether to print the progress of the optimization to screen. */
        use_repair_mechanism,               /* Whether to use a repair mechanism (provided by users) if the problem is constrained. */
        *optimization,                      /* Maximization or Minimization for each objective. */
        print_verbose_overview,             /* Whether to print a overview of settings (0 = no). */
        use_vtr,                            /* Whether to terminate at the value-to-reach (VTR) (0 = no). */
        objective_discretization_in_effect, /* Whether the objective space is currently being discretized for the elitist archive. */
        elitist_archive_front_changed;      /* Whether the Pareto front formed by the elitist archive is changed in this generation. */
// MAXCUT Problem Variables
int     ***maxcut_edges, 
        *number_of_maxcut_edges;
double  **maxcut_edges_weights;
// Knapsack Problem Variables
double  **profits,
        **weights,
        *capacities,
        *ratio_profit_weight;
int     *item_indices_least_profit_order;
int     **item_indices_least_profit_order_according_to_objective;
// PythonFunction Problem Variables
PyObject *functionClass, *functionInstance, *fitnessFunction, *trainFunction;
char* workdir;
char* functionName;
char* string_arg_for_python_fun;
char* init_path;
int* alphabet;
int* alphabet_lower_bound;

int IF_CHECK_CONSTRAINTS = 1;
int ENABLE_DEBUG_OUTPUT = 0;

int max_evals_reached = 0; // bool

int if_write_mi_matrix = 0; //bool
int if_write_linkage_tree = 0; //bool

FILE* file_initialization_genomes;
char if_use_initialization_genomes = 0; //bool
char if_use_initialization_genomes_for_elitist_archive = 0; //bool
char if_call_python_train = FALSE;
long last_eval_when_trained = 0;
long n_evals_between_train = 800;
char* array_population_needs_reevaluation;
/*------------------- Termination of Smaller Populations -------------------*/
char    *array_of_population_statuses;
double  ***array_of_Pareto_front_of_each_population;
int     *array_of_Pareto_front_size_of_each_population;
char    stop_population_when_front_is_covered;
void updateParetoFrontForCurrentPopulation(double **objective_values_pop, double *constraint_values_pop, int pop_size);
void checkWhichSmallerPopulationsNeedToStop();
char checkParetoFrontCover(int pop_index_1, int pop_index_2);
void ezilaitiniArrayOfParetoFronts();
void initializeArrayOfParetoFronts();
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/


/*-=-=-=-=-=-=-=-=-=-=-=-= Section Utility Function -=-=-=-=-=-=-=-=-=-=-=-*/
/**
 * Allocates memory and exits the program in case of a memory allocation failure.
 */
void *Malloc( long size )
{
    void *result;

    result = (void *) malloc( size );
    if( !result )
    {
        printf("\n");
        printf("Error while allocating memory in Malloc( %ld ), aborting program.", size);
        printf("\n");

        exit( 0 );
    }

    return( result );
}
/**
 * Initializes the pseudo-random number generator.
 */
void initializeRandomNumberGenerator()
{
//    printf("Random seed should be initialized by the command line parameter\n");
//    struct timeval tv;
//    struct tm *timep;

//    while( random_seed_changing == 0 )
//    {
//        gettimeofday( &tv, NULL );
//        timep = localtime (&tv.tv_sec);
//        random_seed_changing = timep->tm_hour * 3600 * 1000 + timep->tm_min * 60 * 1000 + timep->tm_sec * 1000 + tv.tv_usec / 1000;
//    }
//    random_seed = random_seed_changing;
//    printf("ACHTUNG");
//    random_seed_changing = 42;
//    random_seed = random_seed_changing;
}
/**
 * Returns a random double, distributed uniformly between 0 and 1.
 */
double randomRealUniform01( void )
{
    int64_t n26, n27;
    double  result;

    random_seed_changing = (random_seed_changing * 0x5DEECE66DLLU + 0xBLLU) & ((1LLU << 48) - 1);
    n26                  = (int64_t)(random_seed_changing >> (48 - 26));
    random_seed_changing = (random_seed_changing * 0x5DEECE66DLLU + 0xBLLU) & ((1LLU << 48) - 1);
    n27                  = (int64_t)(random_seed_changing >> (48 - 27));
    result               = (((int64_t)n26 << 27) + n27) / ((double) (1LLU << 53));

    return( result );
}
        
/**
 * Returns a random integer, distributed uniformly between 0 and maximum.
 */
int randomInt( int maximum )
{
    int result;
    result = (int) (((double) maximum)*randomRealUniform01());
    return( result );
}
/**
 * Computes the two-log of x.
 */
double math_log_two = log(2.0);
double log2( double x )
{
  return( log(x) / math_log_two );
}
int* createRandomOrdering(int size_of_the_set)
{
    int *order, a, b, c, i;

    order = (int *) Malloc( size_of_the_set*sizeof( int ) );
    for( i = 0; i < size_of_the_set; i++ )
        order[i] = i;
    for( i = 0; i < size_of_the_set; i++ )
    {
        a        = randomInt( size_of_the_set );
        b        = randomInt( size_of_the_set );
        c        = order[a];
        order[a] = order[b];
        order[b] = c;
    }

    if (ENABLE_DEBUG_OUTPUT == TRUE) {
        for (i = 0; i < size_of_the_set; i++)
            printf("%d", order[i]);
        printf("\n");
    }

    return order;
}
/**
 * Computes the Euclidean distance between two points.
 */
double distanceEuclidean( double *x, double *y, int number_of_dimensions )
{
    int    i;
    double value, result;

    result = 0.0;
    for( i = 0; i < number_of_dimensions; i++ )
    {
        value   = y[i] - x[i];
        result += value*value;
    }
    result = sqrt( result );

    return( result );
}
/**
 * Sorts an array of doubles and returns the sort-order (small to large).
 */
int *mergeSort( double *array, int array_size )
{
    int i, *sorted, *tosort;

    sorted = (int *) Malloc( array_size * sizeof( int ) );
    tosort = (int *) Malloc( array_size * sizeof( int ) );
    for( i = 0; i < array_size; i++ )
    tosort[i] = i;

    if( array_size == 1 )
        sorted[0] = 0;
    else
        mergeSortWithinBounds( array, sorted, tosort, 0, array_size-1 );

    free( tosort );

    return( sorted );
}
/**
 * Subroutine of merge sort, sorts the part of the array between p and q.
 */
void mergeSortWithinBounds( double *array, int *sorted, int *tosort, int p, int q )
{
    int r;

    if( p < q )
    {
        r = (p + q) / 2;
        mergeSortWithinBounds( array, sorted, tosort, p, r );
        mergeSortWithinBounds( array, sorted, tosort, r+1, q );
        mergeSortMerge( array, sorted, tosort, p, r+1, q );
    }
}
/**
 * Subroutine of merge sort, merges the results of two sorted parts.
 */
void mergeSortMerge( double *array, int *sorted, int *tosort, int p, int r, int q )
{
    int i, j, k, first;

    i = p;
    j = r;
    for( k = p; k <= q; k++ )
    {
        first = 0;
        if( j <= q )
        {
            if( i < r )
            {
                if( array[tosort[i]] < array[tosort[j]] )
                first = 1;
            }
        }
        else
            first = 1;

        if( first )
        {
            sorted[k] = tosort[i];
            i++;
        }
        else
        {
            sorted[k] = tosort[j];
            j++;
        }
    }

    for( k = p; k <= q; k++ )
        tosort[k] = sorted[k];
}

// not sure why, but going from equality to almost equality did nothing, so I reverted that
int almosteq(double a, double b)
{
//    return (fabs(a - b) < (DBL_EPSILON * fabs(a + b)));
//    return (fabs(a - b) < DBL_EPSILON);
//    return (fabs(a - b) < 0.00000000000000001);
    return a == b;
}

/*-=-=-=-=-=-=-=-=-=-=- Section Interpret Command Line -=-=-=-=-=-=-=-=-=-=-*/
/**
 * Parses and checks the command line.
 */
void interpretCommandLine( int argc, char **argv )
{
    parseCommandLine( argc, argv );
  
    checkOptions();
}
/**
 * Parses the command line.
 * For options, see printUsage.
 */
void parseCommandLine( int argc, char **argv )
{
    int index;

    index = 1;

    parseOptions( argc, argv, &index );
  
    parseParameters( argc, argv, &index );
}
/**
 * Parses only the options from the command line.
 */
void parseOptions( int argc, char **argv, int *index )
{
    double dummy;

    print_verbose_overview        = 0;
    use_vtr                       = 0;
    use_print_progress_to_screen  = 0;

    use_pre_mutation              = 0;
    use_pre_adaptive_mutation     = 0;
    use_repair_mechanism          = 0;
    stop_population_when_front_is_covered = 0;

    for( ; (*index) < argc; (*index)++ )
    {
        if( argv[*index][0] == '-' )
        {
            /* If it is a negative number, the option part is over */
            if( sscanf( argv[*index], "%lf", &dummy ) && argv[*index][1] != '\0' )
                break;

            if( argv[*index][1] == '\0' )
                optionError( argv, *index );
            else if( argv[*index][2] != '\0' )
                optionError( argv, *index );
            else
            {
                switch( argv[*index][1] )
                {
                    case '?': printUsage(); break;
                    case 'P': printAllInstalledProblems(); break;
                    case 'v': print_verbose_overview        = 1; break;
                    case 'p': use_print_progress_to_screen  = 1; break;
                    case 'm': use_pre_mutation              = 1; break;
                    case 'M': use_pre_adaptive_mutation     = 1; break;
                    case 'r': use_repair_mechanism          = 1; break; 
                    case 'z': stop_population_when_front_is_covered = 1; break;
                    default : optionError( argv, *index );
                }
            }
        }
        else /* Argument is not an option, so option part is over */
            break;
    }
}
/**
 * Writes the names of all installed problems to the standard output.
 */
void printAllInstalledProblems( void )
{
    int i, n;
    
    n = numberOfInstalledProblems();
    printf("Installed optimization problems:\n");
    for( i = 0; i < n; i++ )
        printf("%3d: %s\n", i, installedProblemName( i ));

    exit( 0 );
}
/**
 * Informs the user of an illegal option and exits the program.
 */
void optionError( char **argv, int index )
{
    printf("Illegal option: %s\n\n", argv[index]);
    printUsage();
}
void readIntsFromFile (const char* file_name, int*  out)
{
    FILE* file = fopen (file_name, "r");
    int i = 0;

    do {
        fscanf (file, "%d", &out[i++]);
    }
    while (!feof (file));

    fclose (file);
}
/**
 * Parses only the EA parameters from the command line.
 */
void parseParameters( int argc, char **argv, int *index )
{
    int noError;

    noError = 1;
    noError = noError && sscanf( argv[*index+0], "%d", &problem_index );
    noError = noError && sscanf( argv[*index+1], "%d", &number_of_objectives );
    noError = noError && sscanf( argv[*index+2], "%d", &number_of_parameters );
    noError = noError && sscanf( argv[*index+3], "%d", &elitist_archive_size_target );
    noError = noError && sscanf( argv[*index+4], "%d", &maximum_number_of_evaluations );
    noError = noError && sscanf( argv[*index+5], "%d", &log_progress_interval );

    // workdir, functionName, randomSeed
    workdir = argv[*index+6];
    functionName = argv[*index+7];
    noError = noError && sscanf( argv[*index+8], "%d", &random_seed_changing );
    int chdir_out = chdir(workdir);

    string_arg_for_python_fun = argv[*index+9];
    char* alphabet_name = argv[*index+10];
    alphabet = (int*) Malloc(number_of_parameters*sizeof(int));
    if (isdigit(alphabet_name[0])) {
        for (int i = 0; i < number_of_parameters; ++i) {
            alphabet[i] = atoi(alphabet_name);
        }
    }
    else {
        readIntsFromFile(alphabet_name, alphabet);
    }
    char* alphabet_lower_bound_name = argv[*index+11];
    alphabet_lower_bound = (int*) Malloc(number_of_parameters*sizeof(int));
    if (isdigit(alphabet_lower_bound_name[0])) {
        for (int i = 0; i < number_of_parameters; ++i) {
            alphabet_lower_bound[i] = atoi(alphabet_lower_bound_name);
        }
    }
    else {
        readIntsFromFile(alphabet_lower_bound_name, alphabet_lower_bound);
    }

    if (argc - *index > 12) {
        init_path = argv[*index + 12];
        file_initialization_genomes = fopen (init_path, "r");

        if_use_initialization_genomes = TRUE;
    }

    if( !noError )
    {
        printf("Error parsing parameters.\n\n");
        printUsage();
    }
}
/**
 * Prints usage information and exits the program.
 */
void printUsage( void )
{
    printf("Usage: MO-GOMEA [-?] [-P] [-s] [-w] [-v] [-r] [-g] pro dim eas eva log gen\n");
    printf(" -?: Prints out this usage information.\n");
    printf(" -P: Prints out a list of all installed optimization problems.\n");
    printf(" -p: Prints optimization progress to screen.\n");
    printf(" -v: Enables verbose mode. Prints the settings before starting the run.\n");
    printf(" -r: Enables use of a repair mechanism if the problem is constrained.\n");
    printf(" -m: Enables use of the weak mutation operator.\n");
    printf(" -M: Enables use of the strong mutation operator.\n");
    printf(" -z: Enable checking if smaller (inefficient) populations should be stopped.\n");
    printf("\n");
    printf("  pro: Index of optimization problem to be solved.\n");
    printf("  num: Number of objectives to be optimized.\n");
    printf("  dim: Number of parameters.\n");
    printf("  eas: Elitist archive size target.\n");
    printf("  eva: Maximum number of evaluations allowed.\n");
    printf("  log: Interval (in terms of number of evaluations) at which the elitist archive is recorded for logging purposes.\n");
    exit( 0 );
}
/**
 * Checks whether the selected options are feasible.
 */
void checkOptions( void )
{
    if( elitist_archive_size_target < 1 )
    {
        printf("\n");
        printf("Error: elitist archive size target < 1 (read: %d).", elitist_archive_size_target);
        printf("\n\n");

        exit( 0 );
    }
    if( maximum_number_of_evaluations < 1 )
    {
        printf("\n");
        printf("Error: maximum number of evaluations < 1 (read: %d). Require maximum number of evaluations >= 1.", maximum_number_of_evaluations);
        printf("\n\n");

        exit( 0 );
    }
    if( installedProblemName( problem_index ) == NULL )
    {
        printf("\n");
        printf("Error: unknown index for problem (read index %d).", problem_index );
        printf("\n\n");

        exit( 0 );
    }
}
/**
 * Prints the settings as read from the command line.
 */
void printVerboseOverview( void )
{
    printf("###################################################\n");
    printf("#\n");
    printf("# Problem                 = %s\n", installedProblemName( problem_index ));
    printf("# Number of objectives    = %d\n", number_of_objectives);
    printf("# Number of parameters    = %d\n", number_of_parameters);
    printf("# Elitist ar. size target = %d\n", elitist_archive_size_target);
    printf("# Maximum numb. of eval.  = %d\n", maximum_number_of_evaluations);
    printf("# Random seed             = %ld\n", random_seed);
    printf("#\n");
    printf("###################################################\n");
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Problems -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
void evaluateIndividual(int *solution, double *obj, double *con, int objective_index_of_extreme_cluster)
{
    number_of_evaluations++;
    if(population_id != -1)
        array_of_number_of_evaluations_per_population[population_id] += 1;
    switch(problem_index)
    {
        case ZEROMAX_ONEMAX: onemaxProblemEvaluation(solution, obj, con, objective_index_of_extreme_cluster); break;
        case TRAP5: trap5ProblemEvaluation(solution, obj, con, objective_index_of_extreme_cluster); break;
        case LOTZ: lotzProblemEvaluation(solution, obj, con, objective_index_of_extreme_cluster); break;
        case KNAPSACK: knapsackProblemEvaluation(solution, obj, con, objective_index_of_extreme_cluster); break;
        case MAXCUT: maxcutProblemEvaluation(solution, obj, con, objective_index_of_extreme_cluster); break;
        case PYTHON_FUNCTION: pythonFunctionProblemEvaluation(solution, obj, con, objective_index_of_extreme_cluster); break;
        default:
            printf("Cannot evaluate this problem!\n");
            exit(1);
    }

    logElitistArchiveAtSpecificPoints();

    if( number_of_evaluations >= maximum_number_of_evaluations ) {
        max_evals_reached = TRUE;
    }
}
/**
 * Returns the name of an installed problem.
 */
char *installedProblemName( int index )
{
    switch( index )
    {
        case  ZEROMAX_ONEMAX:   return( (char *) "Zeromax - Onemax" );
        case  TRAP5:            return( (char *) "Deceptive Trap 5 - Inverse Trap 5 - Tight Encoding" );
        case  KNAPSACK:         return( (char *) "Knapsack - 2 Objectives");
        case  LOTZ:             return( (char *) "Leading One Trailing Zero (LOTZ)");
        case  MAXCUT:           return( (char *) "Maxcut - 2 Objectives");
        case PYTHON_FUNCTION:   return( (char *) "Python function" );
    }
    return( NULL );
}
/**
 * Returns the number of problems installed.
 */
int numberOfInstalledProblems( void )
{
    static int result = -1;
  
    if( result == -1 )
    {
        result = 0;
        while( installedProblemName( result ) != NULL )
            result++;
    }
  
    return( result );
}

void onemaxLoadProblemData()
{
    int k;
    optimization = (char*)Malloc(number_of_objectives*sizeof(char));
    for(k = 0; k < number_of_objectives; k++)
        optimization[k] = MAXIMIZATION;

    use_vtr = 1;
    vtr = 0;
}

void checkPythonError(PyObject *obj, char* msg) {
    if (obj == NULL) {
        printf("%s\n", msg);
        PyErr_Print();
        PyErr_Clear();
        exit(-1);
    }
}

void pythonFunctionLoadProblemData() {
    char moduleName[1000];
    sprintf(moduleName, "import sys; print(f'{sys.path=}')");
    PyRun_SimpleString(moduleName);
    PyObject* module = PyImport_ImportModule("fitness_functions");
    checkPythonError(module, "Module import failed!");

    functionClass = PyObject_GetAttrString(module, functionName);   /* fetch module.class */
    checkPythonError(functionClass, "Class import failed!");

    printf("workdir=%s ; ", workdir);
    printf("gene number=%d ; ", number_of_parameters);
    printf("seed=%d\n", random_seed_changing);

    PyObject *pargs  = Py_BuildValue("(s,s,i,s,i)", workdir, string_arg_for_python_fun,
                                     number_of_parameters, "2", // this parameter does nothing
                                     random_seed_changing);
    functionInstance  = PyEval_CallObject(functionClass, pargs);        /* call class(  ) */
    checkPythonError(functionInstance, "Function init failed!");

    fitnessFunction  = PyObject_GetAttrString(functionInstance, "fitness"); /* fetch bound method */
    checkPythonError(fitnessFunction, "Fitness function retrieval failed!");

    if (strcmp(functionName, "NsgaNetTrainFitness") == 0) {
        printf("Will train!\n");
        if_call_python_train = TRUE;
        trainFunction  = PyObject_GetAttrString(functionInstance, "train"); /* fetch bound method */
        checkPythonError(trainFunction, "Train function retrieval failed!");

        array_population_needs_reevaluation = (char*) Malloc(maximum_number_of_populations*sizeof(char));
        for(int i = 0; i < maximum_number_of_populations; i++)
            array_population_needs_reevaluation[i] = FALSE;
    }

    int k;
    optimization = (char*)Malloc(number_of_objectives*sizeof(char));
    for(k = 0; k < number_of_objectives; k++)
        optimization[k] = MAXIMIZATION;

    use_vtr = 0;
}

void trap5LoadProblemData()
{
    int k;
    optimization = (char*)Malloc(number_of_objectives*sizeof(char));
    for(k = 0; k < number_of_objectives; k++)
        optimization[k] = MAXIMIZATION;

    use_vtr = 1;
    vtr = 0;
}

void lotzLoadProblemData()
{
    int k;
    optimization = (char*)Malloc(number_of_objectives*sizeof(char));
    for(k = 0; k < number_of_objectives; k++)
        optimization[k] = MAXIMIZATION;

    use_vtr = 1;
    vtr = 0;
}

void onemaxProblemEvaluation(char *solution, double *obj_values, double *con_value, int objective_index_of_extreme_cluster)
{
    int i, number_of_1s, number_of_0s;
    
    *con_value = 0.0;
    number_of_0s = 0;
    number_of_1s = 0;
    
    for(i = 0; i < number_of_parameters; i++)
    {
        if(solution[i] == 0)
            number_of_0s++;
        else if(solution[i] == 1)
            number_of_1s++;

        if (ENABLE_DEBUG_OUTPUT == TRUE) {
            printf("%d", solution[i]);
        }
    }
    if (ENABLE_DEBUG_OUTPUT == TRUE) {
        printf("\n");
    }

    obj_values[0] = number_of_0s;
    obj_values[1] = number_of_1s;
}

double what_time_is_it()
{
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    return now.tv_sec + now.tv_nsec*1e-9;
}

void pythonFunctionProblemEvaluation(int *solution, double *obj_values, double *con_value, int objective_index_of_extreme_cluster)
{
//    double time = what_time_is_it();
    *con_value = 0.0;

    PyObject *pySolution = PyTuple_New(number_of_parameters);
    for (Py_ssize_t i = 0; i < number_of_parameters; i++)
    {
        int cur_gene = (int) solution[i];
        PyTuple_SET_ITEM(pySolution, i, PyLong_FromLong(cur_gene));
        if (ENABLE_DEBUG_OUTPUT == TRUE) {
            printf("%d", cur_gene);
        }
    }
    PyObject *arglist = Py_BuildValue("(O)", pySolution);
    PyObject *result = PyEval_CallObject(fitnessFunction, arglist);
    checkPythonError(result, "Fitness calculation failed!");

    PyObject *cacheStatus = PyObject_GetItem(result, PyLong_FromLong(0));
    long cache_status_long = PyLong_AsLong(cacheStatus);
    if (cache_status_long == 1) {
        number_of_evaluations--; //was already increased before calling this function
    }

    PyObject *result0 = PyObject_GetItem(result, PyLong_FromLong(1));
    double fitness0 = PyFloat_AsDouble(result0);
//        PyErr_Print();
//        PyErr_Clear();

    if (ENABLE_DEBUG_OUTPUT == TRUE) {
        printf("___%f", fitness0);
        printf("\n");
    }

    obj_values[0] = fitness0;

    if (number_of_objectives > 1) {
        PyObject *result1 = PyObject_GetItem(result, PyLong_FromLong(2));
        double fitness1 = PyFloat_AsDouble(result1);
        obj_values[1] = fitness1;
        Py_DECREF(result1);
    }
    if (number_of_objectives > 2) {
        PyObject *result2 = PyObject_GetItem(result, PyLong_FromLong(3));
        double fitness2 = PyFloat_AsDouble(result2);
        obj_values[2] = fitness2;
        Py_DECREF(result2);
    }

    Py_DECREF(result);
    Py_DECREF(result0);
    Py_DECREF(arglist);
    Py_DECREF(pySolution);
    Py_DECREF(cacheStatus);
//    printf("time taken %.6lf\n", what_time_is_it() - time);
}

double deceptiveTrapKTightEncodingFunctionProblemEvaluation( char *parameters, int k, char is_one )
{
    int    i, j, m, u;
    double result;

    if( number_of_parameters % k != 0 )
    {
        printf("Error in evaluating deceptive trap k: Number of parameters is not a multiple of k.\n");
        exit( 0 );
    }

    m      = number_of_parameters / k;
    result = 0.0;
    for( i = 0; i < m; i++ )
    {
        u = 0;
        for( j = 0; j < k; j++ )
        u += (parameters[i*k+j] == is_one) ? 1 : 0;

        if( u == k )
            result += k;
        else
            result += (k-1-u);
    }

    return result;
}

void trap5ProblemEvaluation(char *solution, double *obj_values, double *con_value, int objective_index_of_extreme_cluster)
{
    *con_value      = 0.0;
    obj_values[0]   = deceptiveTrapKTightEncodingFunctionProblemEvaluation( solution, 5, TRUE );
    obj_values[1]   = deceptiveTrapKTightEncodingFunctionProblemEvaluation( solution, 5, FALSE );
}

void lotzProblemEvaluation(char *solution, double *obj_values, double *con_value, int objective_index_of_extreme_cluster)
{
    int i;
    double result;

    *con_value = 0.0;
    result = 0.0;
    for(i = 0; i < number_of_parameters; i++)
    {
        if(solution[i] == 0)
            break;
        result += 1;
    }
    obj_values[0] = result; // Leading Ones

    result = 0.0;
    for(i = number_of_parameters - 1; i >= 0; i--)
    {
        if(solution[i] == 1)
            break;
        result += 1;
    }
    obj_values[1] = result; // Trailing Zeros
}

void knapsackLoadProblemData()
{
    int int_number, i, k;
    FILE *file;
    char string[1000];
    double double_number, ratio, *ratios;

    sprintf(string, "./knapsack/knapsack.%d.%d.txt", number_of_parameters, number_of_objectives);
    file = NULL;
    file = fopen(string, "r");
    if(file == NULL)
    {
        printf("Cannot open file %s!\n", string);
        exit(1);
    }

    fscanf(file, "%d", &int_number);
    fscanf(file, "%d", &int_number);

    capacities      = (double*)Malloc(number_of_objectives*sizeof(double));
    weights         = (double**)Malloc(number_of_objectives*sizeof(double*));
    profits         = (double**)Malloc(number_of_objectives*sizeof(double*));
    for(k = 0; k < number_of_objectives; k++)
    {
        weights[k]  = (double*)Malloc(number_of_parameters*sizeof(double));
        profits[k]  = (double*)Malloc(number_of_parameters*sizeof(double));
    }
    for(k = 0; k < number_of_objectives; k++)
    {
        fscanf(file, "%lf", &double_number);
        capacities[k] = double_number;

        for(i = 0; i < number_of_parameters; i++)
        {
            fscanf(file, "%d", &int_number);
            
            fscanf(file, "%d", &int_number);
            weights[k][i] = int_number;
            fscanf(file, "%d", &int_number);
            profits[k][i] = int_number;
        }
    }
    fclose(file);

    ratio_profit_weight = (double*)Malloc(number_of_parameters*sizeof(double));
    for(i = 0; i < number_of_parameters; i++)
    {
        ratio_profit_weight[i] = profits[0][i] / weights[0][i];
        for(k = 1; k < number_of_objectives; k++)
        {
            ratio = profits[k][i] / weights[k][i];
            if(ratio > ratio_profit_weight[i])
                ratio_profit_weight[i] = ratio;
        }
    }

    item_indices_least_profit_order = mergeSort(ratio_profit_weight, number_of_parameters);
    item_indices_least_profit_order_according_to_objective = (int**)Malloc(number_of_objectives*sizeof(int*));
    ratios = (double*)Malloc(number_of_parameters*sizeof(double));
    for(k = 0; k < number_of_objectives; k++)
    {
        for(i = 0; i < number_of_parameters; i++)
            ratios[i] = profits[k][i] / weights[k][i];
        item_indices_least_profit_order_according_to_objective[k] = mergeSort(ratios, number_of_parameters);
    }

    optimization = (char*)Malloc(number_of_objectives*sizeof(char));
    for(k = 0; k < number_of_objectives; k++)
        optimization[k] = MAXIMIZATION;

    free(ratios);
}

void ezilaitiniKnapsackProblemData()
{
    int k;
    for(k = 0; k < number_of_objectives; k++)
    {
        free(weights[k]);
        free(profits[k]);
    }
    free(weights);
    free(profits);
    free(capacities);
    free(ratio_profit_weight);
    free(item_indices_least_profit_order);

    for(k = 0; k < number_of_objectives; k++)
    {
        free(item_indices_least_profit_order_according_to_objective[k]);
    }
    free(item_indices_least_profit_order_according_to_objective);
}

void knapsackSolutionRepair(char *solution, double *solution_profits, double *solution_weights, double *solution_constraint, int objective_index_of_extreme_cluster)
{
    if(objective_index_of_extreme_cluster == -1)
        knapsackSolutionMultiObjectiveRepair(solution, solution_profits, solution_weights, solution_constraint);
    else
        knapsackSolutionSingleObjectiveRepair(solution, solution_profits, solution_weights, solution_constraint, objective_index_of_extreme_cluster);
}

void knapsackSolutionSingleObjectiveRepair(char *solution, double *solution_profits, double *solution_weights, double *solution_constraint, int objective_index)
{
    int i, j, k;
    char isFeasible;

    for(j = 0; j < number_of_parameters; j++)
    {
        i = item_indices_least_profit_order_according_to_objective[objective_index][j];
        if(solution[i] == 0)
            continue;

        solution[i] = 0;
        isFeasible = TRUE;
        for(k = 0; k < number_of_objectives; k++)
        {
            solution_profits[k] = solution_profits[k] - profits[k][i];
            solution_weights[k] = solution_weights[k] - weights[k][i];
            if(solution_weights[k] > capacities[k])
                isFeasible = FALSE;
        }
        if(isFeasible == TRUE)
            break;
    }

    *solution_constraint = 0.0;
    for(k = 0; k < number_of_objectives; k++)
        if(solution_weights[k] > capacities[k])
            (*solution_constraint) = (*solution_constraint) + (solution_weights[k] - capacities[k]);    
}

void knapsackSolutionMultiObjectiveRepair(char *solution, double *solution_profits, double *solution_weights, double *solution_constraint)
{
    int i, j, k;
    char isFeasible;

    for(j = 0; j < number_of_parameters; j++)
    {
        i = item_indices_least_profit_order[j];
        if(solution[i] == 0)
            continue;

        solution[i] = 0;
        isFeasible = TRUE;
        for(k = 0; k < number_of_objectives; k++)
        {
            solution_profits[k] = solution_profits[k] - profits[k][i];
            solution_weights[k] = solution_weights[k] - weights[k][i];
            if(solution_weights[k] > capacities[k])
                isFeasible = FALSE;
        }
        if(isFeasible == TRUE)
            break;
    }

    *solution_constraint = 0.0;
    for(k = 0; k < number_of_objectives; k++)
        if(solution_weights[k] > capacities[k])
            (*solution_constraint) = (*solution_constraint) + (solution_weights[k] - capacities[k]);
}

void knapsackProblemEvaluation(char *solution, double *obj_values, double *con_value, int objective_index_of_extreme_cluster)
{
    int i, k;
    double *solution_profits, *solution_weights;

    solution_weights = (double*)Malloc(number_of_objectives*sizeof(double));
    solution_profits = (double*)Malloc(number_of_objectives*sizeof(double));
    *con_value = 0.0;
    for(k = 0; k < number_of_objectives; k++)
    {
        solution_profits[k] = 0.0;
        solution_weights[k] = 0.0;
        for(i = 0; i < number_of_parameters; i++)
        {
            solution_profits[k] += ((int)solution[i])*profits[k][i];
            solution_weights[k] += ((int)solution[i])*weights[k][i];
        }
        if(solution_weights[k] > capacities[k])
            (*con_value) = (*con_value) + (solution_weights[k] - capacities[k]);
    }

    if(use_repair_mechanism)
    {
        if( (*con_value) > 0)
            knapsackSolutionRepair(solution, solution_profits, solution_weights, con_value, objective_index_of_extreme_cluster);
    }

    for(k = 0; k < number_of_objectives; k++)
        obj_values[k] = solution_profits[k];

    free(solution_weights);
    free(solution_profits);
}

void maxcutLoadProblemData()
{
    int i, k;
    char string[1000];
    maxcut_edges = (int ***) Malloc(number_of_objectives * sizeof(int **));
    number_of_maxcut_edges = (int *) Malloc(number_of_objectives * sizeof(int ));
    maxcut_edges_weights = (double **) Malloc(number_of_objectives * sizeof(double *));

    for (i = 0; i < number_of_objectives; i++)
    {
        sprintf(string, "maxcut/maxcut_instance_%d_%d.txt", number_of_parameters, i);
        maxcutReadInstanceFromFile(string, i);
    }

    optimization = (char*)Malloc(number_of_objectives*sizeof(char));
    for(k = 0; k < number_of_objectives; k++)
        optimization[k] = MAXIMIZATION;
}

void ezilaitiniMaxcutProblemData()
{
    int i,j;
    for(i=0;i<number_of_objectives;i++)
    {
        for(j=0;j<number_of_maxcut_edges[i];j++)
            free(maxcut_edges[i][j]);
        free(maxcut_edges[i]);
        free(maxcut_edges_weights[i]);
    }
    free(maxcut_edges);
    free(maxcut_edges_weights);
    free(number_of_maxcut_edges); 
}

void maxcutReadInstanceFromFile(char *filename, int objective_index)
{
    char  c, string[1000], substring[1000];
    int   i, j, k, q, number_of_vertices, number_of_edges;
    FILE *file;

    //file = fopen( "maxcut_instance.txt", "r" );
    file = fopen( filename, "r" );
    if( file == NULL )
    {
        printf("Error in opening file \"maxcut_instance.txt\"");
        exit( 0 );
    }

    c = fgetc( file );
    k = 0;
    while( c != '\n' && c != EOF )
    {
        string[k] = (char) c;
        c      = fgetc( file );
        k++;
    }
    string[k] = '\0';

    q = 0;
    j = 0;
    while( (string[j] != ' ') && (j < k) )
    {
        substring[q] = string[j];
        q++;
        j++;
    }
    substring[q] = '\0';
    j++;

    number_of_vertices = atoi( substring );
    if( number_of_vertices != number_of_parameters )
    {
        printf("Error during reading of maxcut instance:\n");
        printf("  Read number of vertices: %d\n", number_of_vertices);
        printf("  Doesn't match number of parameters on command line: %d\n", number_of_parameters);
        exit( 1 );
    }

    q = 0;
    while( (string[j] != ' ') && (j < k) )
    {
        substring[q] = string[j];
        q++;
        j++;
    }
    substring[q] = '\0';
    j++;

    number_of_edges = atoi( substring );
    number_of_maxcut_edges[objective_index] = number_of_edges;
    maxcut_edges[objective_index] = (int **) Malloc( number_of_edges*sizeof( int * ) );
    for( i = 0; i < number_of_edges; i++ )
        maxcut_edges[objective_index][i] = (int *) Malloc( 2*sizeof( int ) );
    maxcut_edges_weights[objective_index] = (double *) Malloc( number_of_edges*sizeof( double ) );

    i = 0;
    c = fgetc( file );
    k = 0;
    while( c != '\n' && c != EOF )
    {
        string[k] = (char) c;
        c      = fgetc( file );
        k++;
    }
    string[k] = '\0';
    while( k > 0 )
    {
        q = 0;
        j = 0;
        while( (string[j] != ' ') && (j < k) )
        {
            substring[q] = string[j];
            q++;
            j++;
        }
        substring[q] = '\0';
        j++;

        maxcut_edges[objective_index][i][0] = atoi( substring )-1;

        q = 0;
        while( (string[j] != ' ') && (j < k) )
        {
            substring[q] = string[j];
            q++;
            j++;
        }
        substring[q] = '\0';
        j++;

        maxcut_edges[objective_index][i][1] = atoi( substring )-1;

        q = 0;
        while( (string[j] != ' ') && (j < k) )
        {
            substring[q] = string[j];
            q++;
            j++;
        }
        substring[q] = '\0';
        j++;

        maxcut_edges_weights[objective_index][i] = atof( substring );
        i++;

        c = fgetc( file );
        k = 0;
        while( c != '\n' && c != EOF )
        {
            string[k] = (char) c;
            c      = fgetc( file );
            k++;
        }
        string[k] = '\0';
    }

    fclose( file );
}

void maxcutProblemEvaluation( char *solution, double *obj_values, double *con_value, int objective_index_of_extreme_cluster )
{
    int    i, k;
    double result;

    *con_value = 0;

    for(k = 0; k < number_of_objectives; k++)
    {
        result = 0.0;
        for( i = 0; i < number_of_maxcut_edges[k]; i++ )
        {
            if( solution[maxcut_edges[k][i][0]] != solution[maxcut_edges[k][i][1]] )
                result += maxcut_edges_weights[k][i];
        }

        obj_values[k] = result;
    }
}

double **getDefaultFrontOnemaxZeromax( int *default_front_size )
{
    int  i;
    static double **result = NULL;
    *default_front_size = ( number_of_parameters + 1 );

    if( result == NULL )
    {
        result = (double **) Malloc( (*default_front_size)*sizeof( double * ) );
        for( i = 0; i < (*default_front_size); i++ )
            result[i] = (double *) Malloc( 2*sizeof( double ) );

        for( i = 0; i < (*default_front_size); i++ )
        {
            result[i][0] = i;                                         // Zeromax
            result[i][1] = number_of_parameters - result[i][0];       // Onemax
        }
    }
    return( result );
}

double **getDefaultFrontTrap5InverseTrap5( int *default_front_size )
{
    int  i, number_of_blocks;
    static double **result = NULL;

    number_of_blocks = number_of_parameters / 5;
    *default_front_size = ( number_of_blocks + 1 );

    if( result == NULL )
    {
        result = (double **) Malloc( (*default_front_size)*sizeof( double * ) );
        for( i = 0; i < (*default_front_size); i++ )
            result[i] = (double *) Malloc( 2*sizeof( double ) );

        for( i = 0; i < (*default_front_size); i++ )                    // i = number of all-1 blocks
        {
            result[i][0] = ( 5 * i ) + ( 4 * (number_of_blocks - i) ) ;   // Trap-5
            result[i][1] = ( 5 * (number_of_blocks - i)) + ( 4 * i );     // Inverse Trap-5
        }
    }
    return( result );
}

double **getDefaultFrontLeadingOneTrailingZero( int *default_front_size )
{
    int  i;
    static double **result = NULL;

    *default_front_size = ( number_of_parameters + 1 );

    if( result == NULL )
    {
        result = (double **) Malloc( (*default_front_size)*sizeof( double * ) );
        for( i = 0; i < (*default_front_size); i++ )
            result[i] = (double *) Malloc( 2*sizeof( double ) );

        for( i = 0; i < (*default_front_size); i++ )
        {
            result[i][0] = i;                                         // Leading One
            result[i][1] = number_of_parameters - result[i][0];       // Trailing Zero
        }
    }

    return( result );
}
/**
 * Returns whether the D_{Pf->S} metric can be computed.
 */
short haveDPFSMetric( void )
{
    int default_front_size;

    getDefaultFront( &default_front_size );
    if( default_front_size > 0 )
        return( 1 );

    return( 0 );
}
/**
 * Returns the default front(NULL if there is none).
 * The number of solutions in the default
 * front is returned in the pointer variable.
 */
double **getDefaultFront( int *default_front_size )
{
    switch( problem_index )
    {
        case ZEROMAX_ONEMAX: return( getDefaultFrontOnemaxZeromax( default_front_size ) );
        case TRAP5: return( getDefaultFrontTrap5InverseTrap5( default_front_size ) );
        case LOTZ: return( getDefaultFrontLeadingOneTrailingZero( default_front_size ) );
    }

    *default_front_size = 0;
    return( NULL );
}

double computeDPFSMetric( double **default_front, int default_front_size, double **approximation_front, int approximation_front_size )
{
    int    i, j;
    double result, distance, smallest_distance;

    if( approximation_front_size == 0 )
        return( 1e+308 );

    result = 0.0;
    for( i = 0; i < default_front_size; i++ )
    {
        smallest_distance = 1e+308;
        for( j = 0; j < approximation_front_size; j++ )
        {
            distance = distanceEuclidean( default_front[i], approximation_front[j], number_of_objectives );
            if( distance < smallest_distance )
                smallest_distance = distance;
      
        }
        result += smallest_distance;
    }
    result /= (double) default_front_size;

    return( result );
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=- Tracking Progress =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Writes (appends) statistics about the current generation to a
 * file named "statistics.dat".
 */
void writeGenerationalStatistics( void )
{
    int     i;
    char    string[1000];
    FILE   *file;

    file = NULL;
    if(( number_of_generations == 0 && population_id == 0) ||
        (number_of_generations == 0 && population_id == -1))
    {
        file = fopen( "statistics.dat", "w" );

        sprintf( string, "# Generation  Population  Evaluations   [ Cluster_Index ]\n");
        fputs( string, file );
    }
    else
        file = fopen( "statistics.dat", "a" );

    sprintf( string, "  %10d %10d %11d     [ ", number_of_generations, population_size, number_of_evaluations );
    fputs( string, file );

    for( i = 0; i < number_of_mixing_components; i++ )
    {
        sprintf( string, "%4d", i );
        fputs( string, file );
        if( i < number_of_mixing_components-1 )
        {
            sprintf( string, " " );
            fputs( string, file );
        }
    }
    sprintf( string, " ]\n" );
    fputs( string, file );

    fclose( file );

    freeAuxiliaryPopulations();
}

void writeCurrentElitistArchive( char final )
{
    int   i, j, k, index;
    char  string[1000];
    FILE *file;

    /* Elitist archive */
    if( final )
        sprintf( string, "elitist_archive_generation_final.dat" );
    else
        sprintf( string, "elitist_archive_at_evaluation_%d.dat", number_of_evaluations );
    file = fopen( string, "w" );

    for( i = 0; i < elitist_archive_size; i++ )
    {
        for( j = 0; j < number_of_objectives; j++ )
        {
            sprintf( string, "%6e ", elitist_archive_objective_values[i][j] );
            fputs( string, file );
        }

        sprintf( string, "%f ", elitist_archive_constraint_values[i]);
        fputs( string, file );

        for( j = 0; j < number_of_parameters; j++ )
        {
            sprintf( string, "%d,", elitist_archive[i][j] );
            fputs( string, file );
        }
        sprintf( string, "\n" );
        fputs( string, file );
    }
    fclose( file );
}

void writeToFileMIMatrix(int cluster_index)
{
    FILE *f;
    int i, j;
    char filename[100];

    sprintf( filename, "MI_matrix_at_evaluation_%d_cluster_%d.dat", number_of_evaluations, cluster_index);
    f = fopen( filename, "w" );
    if( f != NULL )
    {
        for( i = 0; i < number_of_parameters; i++ )
        {
            for( j = 0; j < number_of_parameters; j++ )
            {
                fprintf(f, "%10.3e ",MI_matrix[i][j]);
            }
            fprintf(f, "\n");
        }
        fclose( f );
    }

}

void logElitistArchiveAtSpecificPoints()
{
    if(number_of_evaluations % log_progress_interval == 0) {
        writeCurrentElitistArchive(FALSE);
//        writeToFileMIMatrix();
    }
}

/**
 * Returns TRUE if termination should be enforced, FALSE otherwise.
 */
char checkTerminationCondition()
{
    if( maximum_number_of_evaluations >= 0 )
    {
        if( checkNumberOfEvaluationsTerminationCondition() )
            return( TRUE );
    }

    if( use_vtr )
    {
        if( checkVTRTerminationCondition() )
            return( TRUE );
    }

    return( FALSE );
}
/**
 * Returns TRUE if the maximum number of evaluations
 * has been reached, FALSE otherwise.
 */
char checkNumberOfEvaluationsTerminationCondition()
{
  if( number_of_evaluations >= maximum_number_of_evaluations )
    return( TRUE );

  return( FALSE );
}
/**
 * Returns 1 if the value-to-reach has been reached
 * for the multi-objective case. This means that
 * the D_Pf->S metric has reached the value-to-reach.
 * If no D_Pf->S can be computed, 0 is returned.
 */
char checkVTRTerminationCondition( void )
{
  int      default_front_size;
  double **default_front, metric_elitist_archive;

  if( haveDPFSMetric() )
  {
    default_front          = getDefaultFront( &default_front_size );
    metric_elitist_archive = computeDPFSMetric( default_front, default_front_size, elitist_archive_objective_values, elitist_archive_size );

    if( metric_elitist_archive <= vtr )
    {
      return( 1 );
    }
  }

  return( 0 );
}

void logNumberOfEvaluationsAtVTR()
{
    int      default_front_size;
    double **default_front, metric_elitist_archive;
    FILE *file;
    char string[1000];

    if(use_vtr == FALSE)
        return;

    if( haveDPFSMetric() )
    {
        default_front          = getDefaultFront( &default_front_size );
        metric_elitist_archive = computeDPFSMetric( default_front, default_front_size, elitist_archive_objective_values, elitist_archive_size );

        sprintf(string, "number_of_evaluations_when_all_points_found_%d.dat", number_of_parameters);
        file = fopen(string, "a");
        if( metric_elitist_archive <= vtr )
        {
            fprintf(file, "%d\n", number_of_evaluations);
        }
        else
        {
            fprintf(file, "Cannot find all points within current budget!\n");    
        }
        fclose(file);  
    }
}

/*=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=  Elitist Archive -==-=-=-=-=-=-=-=-=-=-=-=*/
char isDominatedByElitistArchive( double *obj, double con, char *is_new_nondominated_point, int *position_of_existed_member )
{
    int j;

    *is_new_nondominated_point = TRUE;
    *position_of_existed_member = -1;
    for( j = 0; j < elitist_archive_size; j++ )
    {
        if( constraintParetoDominates( elitist_archive_objective_values[j], elitist_archive_constraint_values[j], obj, con ) )
        {
            *is_new_nondominated_point = FALSE;
            return( TRUE );
        }
        else
        {
            if( !constraintParetoDominates( obj, con, elitist_archive_objective_values[j], elitist_archive_constraint_values[j] ) )
            {
              if( sameObjectiveBox( elitist_archive_objective_values[j], obj ) )
              {
                *is_new_nondominated_point = FALSE;
                *position_of_existed_member = j;
                return( FALSE );
              }
            }
        }
    }
    return( FALSE );
}
/**
 * Returns 1 if two solutions share the same objective box, 0 otherwise.
 */
short sameObjectiveBox( double *objective_values_a, double *objective_values_b )
{
    int i;

    if( !objective_discretization_in_effect )
    {
        /* If the solutions are identical, they are still in the (infinitely small) same objective box. */
        for( i = 0; i < number_of_objectives; i++ )
        {
            if( objective_values_a[i] != objective_values_b[i] )
                return( 0 );
        }

        return( 1 );
    }


    for( i = 0; i < number_of_objectives; i++ )
    {
        if( ((int) (objective_values_a[i] / objective_discretization[i])) != ((int) (objective_values_b[i] / objective_discretization[i])) )
            return( 0 );
    }

    return( 1 );
}

int hammingDistanceInParameterSpace(int *solution_1, int *solution_2)
{
	int i, distance;
	distance=0;
	for (i=0; i < number_of_parameters; i++)
	{
		if( solution_1[i] != solution_2[i])
			distance++;
	}

	return distance;
}

int hammingDistanceToNearestNeighborInParameterSpace(int *solution, int replacement_position)
{
	int i, distance_to_nearest_neighbor, distance;
	distance_to_nearest_neighbor = -1;
	for (i = 0; i < elitist_archive_size; i++)
	{
		if (i != replacement_position)
		{
			distance = hammingDistanceInParameterSpace(solution, elitist_archive[i]);
			if (distance < distance_to_nearest_neighbor || distance_to_nearest_neighbor < 0)
				distance_to_nearest_neighbor = distance;
		}
	}

	return distance_to_nearest_neighbor;
}
/**
 * Updates the elitist archive by offering a new solution
 * to possibly be added to the archive. If there are no
 * solutions in the archive yet, the solution is added.
 * Solution A is always dominated by solution B that is
 * in the same domination-box if B dominates A or A and
 * B do not dominate each other. If the solution is not
 * dominated, it is added to the archive and all solutions
 * dominated by the new solution, are purged from the archive.
 */
void updateElitistArchive( int *solution, double *solution_objective_values, double solution_constraint_value)
{
    short is_dominated_itself;
    int   i, *indices_dominated, number_of_solutions_dominated;

    if( elitist_archive_size == 0 )
        addToElitistArchive( solution, solution_objective_values, solution_constraint_value);
    else
    {
        indices_dominated             = (int *) Malloc( elitist_archive_size*sizeof( int ) );
        number_of_solutions_dominated = 0;
        is_dominated_itself           = 0;
        for( i = 0; i < elitist_archive_size; i++ )
        {
            if( constraintParetoDominates( elitist_archive_objective_values[i], elitist_archive_constraint_values[i], solution_objective_values, solution_constraint_value ) )
                is_dominated_itself = 1;
            else
            {
                if( !constraintParetoDominates( solution_objective_values, solution_constraint_value, elitist_archive_objective_values[i], elitist_archive_constraint_values[i] ) )
                {
                    if( sameObjectiveBox( elitist_archive_objective_values[i], solution_objective_values ) )
                        is_dominated_itself = 1;
                }
            }

            if( is_dominated_itself )
                break;
        }

        if (ENABLE_DEBUG_OUTPUT == TRUE) printf("                          is_dominated=%d\n", is_dominated_itself);

        if( !is_dominated_itself )
        {
            for( i = 0; i < elitist_archive_size; i++ )
            {
                if( constraintParetoDominates( solution_objective_values, solution_constraint_value, elitist_archive_objective_values[i], elitist_archive_constraint_values[i] ) )
                {
                    indices_dominated[number_of_solutions_dominated] = i;
                    number_of_solutions_dominated++;
                }
            }

            if( number_of_solutions_dominated > 0 )
                removeFromElitistArchive( indices_dominated, number_of_solutions_dominated );
            if (ENABLE_DEBUG_OUTPUT == TRUE) printf("From updateElitistsArchive\n");
            addToElitistArchive( solution, solution_objective_values, solution_constraint_value);
        }

        free( indices_dominated );
    }
}
void updateElitistArchiveWithReplacementOfExistedMember( int *solution, double *solution_objective_values, double solution_constraint_value, char *is_new_nondominated_point, char *is_dominated_by_archive)
{
    short is_existed, index_of_existed_member;
    int   i, *indices_dominated, number_of_solutions_dominated;
    int distance_old, distance_new;

    *is_new_nondominated_point  = TRUE;
    *is_dominated_by_archive    = FALSE;

    if( elitist_archive_size == 0 )
        addToElitistArchive( solution, solution_objective_values, solution_constraint_value);
    else
    {
        indices_dominated             = (int *) Malloc( elitist_archive_size*sizeof( int ) );
        number_of_solutions_dominated = 0;
        is_existed					  = 0;
        for( i = 0; i < elitist_archive_size; i++ )
        {
            if( constraintParetoDominates( elitist_archive_objective_values[i], elitist_archive_constraint_values[i], solution_objective_values, solution_constraint_value ) )
            {
                *is_dominated_by_archive    = TRUE;
                *is_new_nondominated_point  = FALSE;
            }
            else
            {
                if( !constraintParetoDominates( solution_objective_values, solution_constraint_value, elitist_archive_objective_values[i], elitist_archive_constraint_values[i] ) )
                {
                    if( sameObjectiveBox( elitist_archive_objective_values[i], solution_objective_values ) )
                    {
                        is_existed                  = 1;
                        index_of_existed_member     = i;
                        *is_new_nondominated_point  = FALSE;
                    }
                }
            }

            if( (*is_new_nondominated_point) == FALSE )
                break;
        }

        if( (*is_new_nondominated_point) == TRUE )
        {
            for( i = 0; i < elitist_archive_size; i++ )
            {
                if( constraintParetoDominates( solution_objective_values, solution_constraint_value, elitist_archive_objective_values[i], elitist_archive_constraint_values[i] ) )
                {
                    indices_dominated[number_of_solutions_dominated] = i;
                    number_of_solutions_dominated++;
                }
            }

            if( number_of_solutions_dominated > 0 )
                removeFromElitistArchive( indices_dominated, number_of_solutions_dominated );

            if (ENABLE_DEBUG_OUTPUT == TRUE) printf("From updateElitistsArchiveWithReplacement...\n");
            addToElitistArchive( solution, solution_objective_values, solution_constraint_value);
            elitist_archive_front_changed = TRUE;
        }

        if( is_existed )
        {
            distance_old = hammingDistanceToNearestNeighborInParameterSpace(elitist_archive[index_of_existed_member], index_of_existed_member);
            distance_new = hammingDistanceToNearestNeighborInParameterSpace(solution, index_of_existed_member);

            if (distance_new > distance_old)
            {
                for(i = 0; i < number_of_parameters; i++) {
                    elitist_archive[index_of_existed_member][i] = solution[i];
                }
                if (ENABLE_DEBUG_OUTPUT == TRUE) printf("Replace!\n");
                for(i=0; i < number_of_objectives; i++)
                    elitist_archive_objective_values[index_of_existed_member][i] = solution_objective_values[i];
                elitist_archive_constraint_values[index_of_existed_member] = solution_constraint_value;
            }
        }
    
        free( indices_dominated );
    }
}

/**
 * Removes a set of solutions (identified by their archive-indices)
 * from the elitist archive.
 */
void removeFromElitistArchive( int *indices, int number_of_indices )
{
    int      i, j, elitist_archive_size_new;
    int **elitist_archive_new;
    double **elitist_archive_objective_values_new;
	double *elitist_archive_constraint_values_new;

    elitist_archive_new                   = (int**) Malloc( elitist_archive_capacity*sizeof( int * ) );
    elitist_archive_objective_values_new  = (double **) Malloc( elitist_archive_capacity*sizeof( double * ) );
    elitist_archive_constraint_values_new = (double *) Malloc( elitist_archive_capacity*sizeof( double ) );

    for( i = 0; i < elitist_archive_capacity; i++ )
    {
        elitist_archive_new[i]                  = (int *) Malloc( number_of_parameters*sizeof( int ) );
        elitist_archive_objective_values_new[i] = (double *) Malloc( number_of_objectives*sizeof( double ) );
    }

    elitist_archive_size_new = 0;
    for( i = 0; i < elitist_archive_size; i++ )
    {
        if( !isInListOfIndices( i, indices, number_of_indices ) )
        {
            for( j = 0; j < number_of_parameters; j++ )
                elitist_archive_new[elitist_archive_size_new][j] = elitist_archive[i][j];
            for( j = 0; j < number_of_objectives; j++ )
                elitist_archive_objective_values_new[elitist_archive_size_new][j] = elitist_archive_objective_values[i][j];
            elitist_archive_constraint_values_new[elitist_archive_size_new] = elitist_archive_constraint_values[i];

            elitist_archive_size_new++;
        }
    }

    for( i = 0; i < elitist_archive_capacity; i++ )
    {
        free( elitist_archive[i] );
        free( elitist_archive_objective_values[i] );
    }
    free( elitist_archive );
    free( elitist_archive_objective_values );
    free( elitist_archive_constraint_values );

    elitist_archive_size              = elitist_archive_size_new;
    elitist_archive                   = elitist_archive_new;
    elitist_archive_objective_values  = elitist_archive_objective_values_new;
    elitist_archive_constraint_values = elitist_archive_constraint_values_new;
}

/**
 * Returns 1 if index is in the indices array, 0 otherwise.
 */
short isInListOfIndices( int index, int *indices, int number_of_indices )
{
    int i;

    for( i = 0; i < number_of_indices; i++ )
        if( indices[i] == index )
        return( 1 );

    return( 0 );
}

/**
 * Adds a solution to the elitist archive.
 */
void addToElitistArchive( int *solution, double *solution_objective_values, double solution_constraint_value )
{
    int      i, j, elitist_archive_capacity_new;
    int **elitist_archive_new;
    double **elitist_archive_objective_values_new;
	double *elitist_archive_constraint_values_new;

    if( elitist_archive_capacity == elitist_archive_size )
    {
        elitist_archive_capacity_new          = elitist_archive_capacity*2+1;
        elitist_archive_new                   = (int **) Malloc( elitist_archive_capacity_new*sizeof( int * ) );
        elitist_archive_objective_values_new  = (double **) Malloc( elitist_archive_capacity_new*sizeof( double * ) );
        elitist_archive_constraint_values_new = (double *) Malloc( elitist_archive_capacity_new*sizeof( double ) );

        for( i = 0; i < elitist_archive_capacity_new; i++ )
        {
            elitist_archive_new[i]                    = (int *) Malloc( number_of_parameters*sizeof( int ) );
            elitist_archive_objective_values_new[i]   = (double *) Malloc( number_of_objectives*sizeof( double ) );
        }

        for( i = 0; i < elitist_archive_size; i++ )
        {
            for( j = 0; j < number_of_parameters; j++ )
                elitist_archive_new[i][j] = elitist_archive[i][j];
            for( j = 0; j < number_of_objectives; j++ )
                elitist_archive_objective_values_new[i][j] = elitist_archive_objective_values[i][j];
            elitist_archive_constraint_values_new[i] = elitist_archive_constraint_values[i];
        }

        for( i = 0; i < elitist_archive_capacity; i++ )
        {
            free( elitist_archive[i] );
            free( elitist_archive_objective_values[i] );
        }
        free( elitist_archive );
        free( elitist_archive_objective_values );
        free( elitist_archive_constraint_values );

        elitist_archive_capacity          = elitist_archive_capacity_new;
        elitist_archive                   = elitist_archive_new;
        elitist_archive_objective_values  = elitist_archive_objective_values_new;
        elitist_archive_constraint_values = elitist_archive_constraint_values_new;
    }

    if (ENABLE_DEBUG_OUTPUT == TRUE) printf("adding ");
    for( j = 0; j < number_of_parameters; j++ ) {
        elitist_archive[elitist_archive_size][j] = solution[j];
        if (ENABLE_DEBUG_OUTPUT == TRUE) printf("%d", solution[j]);
    }
    if (ENABLE_DEBUG_OUTPUT == TRUE) printf("\n");
    for( j = 0; j < number_of_objectives; j++ )
        elitist_archive_objective_values[elitist_archive_size][j] = solution_objective_values[j];
    elitist_archive_constraint_values[elitist_archive_size] = solution_constraint_value; // Notice here //

    elitist_archive_size++;
}
/**
 * Adapts the objective box discretization. If the numbre
 * of solutions in the elitist archive is too high or too low
 * compared to the population size, the objective box
 * discretization is adjusted accordingly. In doing so, the
 * entire elitist archive is first emptied and then refilled.
 */
void adaptObjectiveDiscretization( void )
{
    int    i, j, k, na, nb, nc, elitist_archive_size_target_lower_bound, elitist_archive_size_target_upper_bound;
    double low, high, *elitist_archive_objective_ranges;

    elitist_archive_size_target_lower_bound = (int) (0.75*elitist_archive_size_target);
    elitist_archive_size_target_upper_bound = (int) (1.25*elitist_archive_size_target);

    if( objective_discretization_in_effect && (elitist_archive_size < elitist_archive_size_target_lower_bound) )
        objective_discretization_in_effect = 0;

    if( elitist_archive_size > elitist_archive_size_target_upper_bound )
    {
        objective_discretization_in_effect = 1;

        elitist_archive_objective_ranges = (double *) Malloc( number_of_objectives*sizeof( double ) );
        for( j = 0; j < number_of_objectives; j++ )
        {
            low  = elitist_archive_objective_values[0][j];
            high = elitist_archive_objective_values[0][j];

            for( i = 0; i < elitist_archive_size; i++ )
            {
                if( elitist_archive_objective_values[i][j] < low )
                    low = elitist_archive_objective_values[i][j];
                if( elitist_archive_objective_values[i][j] > high )
                    high = elitist_archive_objective_values[i][j];
            }

            elitist_archive_objective_ranges[j] = high - low;
        }

        na = 1;
        nb = (int) pow(2.0,25.0);
        
        for( k = 0; k < 25; k++ )
        {
            nc = (na + nb) / 2;
            for( i = 0; i < number_of_objectives; i++ )
                objective_discretization[i] = elitist_archive_objective_ranges[i]/((double) nc);

            /* Restore the original elitist archive after the first cycle in this loop */
            if( k > 0 )
            {
                elitist_archive_size = 0;
                for( i = 0; i < elitist_archive_copy_size; i++ )
                    addToElitistArchive( elitist_archive_copy[i], elitist_archive_copy_objective_values[i], elitist_archive_copy_constraint_values[i] );
            }

            /* Copy the entire elitist archive */
            if( elitist_archive_copy != NULL )
            {
                for( i = 0; i < elitist_archive_copy_size; i++ )
                {
                    free( elitist_archive_copy[i] );
                    free( elitist_archive_copy_objective_values[i] );
                }
                free( elitist_archive_copy );
                free( elitist_archive_copy_objective_values );
                free( elitist_archive_copy_constraint_values );
            }

            elitist_archive_copy_size              = elitist_archive_size;
            elitist_archive_copy                   = (int **) Malloc( elitist_archive_copy_size*sizeof( int * ) );
            elitist_archive_copy_objective_values  = (double **) Malloc( elitist_archive_copy_size*sizeof( double * ) );
            elitist_archive_copy_constraint_values = (double *) Malloc( elitist_archive_copy_size*sizeof( double ) );
      
            for( i = 0; i < elitist_archive_copy_size; i++ )
            {
                elitist_archive_copy[i]                  = (int *) Malloc( number_of_parameters*sizeof( int ) );
                elitist_archive_copy_objective_values[i] = (double *) Malloc( number_of_objectives*sizeof( double ) );
            }
            for( i = 0; i < elitist_archive_copy_size; i++ )
            {
                for( j = 0; j < number_of_parameters; j++ )
                    elitist_archive_copy[i][j] = elitist_archive[i][j];
                for( j = 0; j < number_of_objectives; j++ )
                    elitist_archive_copy_objective_values[i][j] = elitist_archive_objective_values[i][j];
                elitist_archive_copy_constraint_values[i] = elitist_archive_constraint_values[i];
            }

            /* Clear the elitist archive */
            elitist_archive_size = 0;

            /* Rebuild the elitist archive */
            for( i = 0; i < elitist_archive_copy_size; i++ )
                updateElitistArchive( elitist_archive_copy[i], elitist_archive_copy_objective_values[i], elitist_archive_copy_constraint_values[i]);

            if( elitist_archive_size <= elitist_archive_size_target_lower_bound )
                na = nc;
            else
                nb = nc;
        }

        free( elitist_archive_objective_ranges );
    }
}

/*-=-=-=-=-=-=-=-=-=-=-=-=- Solution Comparison -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
char betterFitness( double *objective_value_x, double constraint_value_x, double *objective_value_y, double constraint_value_y, int objective_index )
{
    short result;

    result = FALSE;

    if (IF_CHECK_CONSTRAINTS == TRUE) {
        if (constraint_value_x > 0) /* x is infeasible */
        {
            if (constraint_value_y > 0) /* Both are infeasible */
            {
                if (constraint_value_x < constraint_value_y)
                    result = TRUE;
            }
        } else {
            if (constraint_value_y > 0)
                result = TRUE;
            else {
                if (optimization[objective_index] == MINIMIZATION) {
                    if (objective_value_x[objective_index] < objective_value_y[objective_index])
                        result = TRUE;
                } else if (optimization[objective_index] == MAXIMIZATION) {
                    if (objective_value_x[objective_index] > objective_value_y[objective_index])
                        result = TRUE;
                }
            }
        }
    }
    else {
        if (optimization[objective_index] == MINIMIZATION) {
            if (objective_value_x[objective_index] < objective_value_y[objective_index])
                result = TRUE;
        } else if (optimization[objective_index] == MAXIMIZATION) {
            if (objective_value_x[objective_index] > objective_value_y[objective_index])
                result = TRUE;
        }
    }

    return ( result );
}

char equalFitness(double *objective_value_x, double constraint_value_x, double *objective_value_y, double constraint_value_y, int objective_index )
{
    short result;

    result = FALSE;
    if (IF_CHECK_CONSTRAINTS == TRUE) {
        if (constraint_value_x > 0) /* x is infeasible */
        {
            if (constraint_value_y > 0) /* Both are infeasible */
            {
                if (almosteq(constraint_value_x, constraint_value_y) == TRUE) {
                    result = TRUE;
                }
            }
        } else {
            if (almosteq(constraint_value_y, 0) == TRUE) {
                if (almosteq(objective_value_x[objective_index], objective_value_y[objective_index]) == TRUE) {
                    result = TRUE;
                }
            }
        }
    }
    else {
        if (almosteq(objective_value_x[objective_index], objective_value_y[objective_index])) {
            if (ENABLE_DEBUG_OUTPUT == TRUE)
                printf("        (%f, %f)", objective_value_x[objective_index], objective_value_y[objective_index]);
            result = TRUE;
        }
    }

    return ( result );
}
/**
 * Returns 1 if x constraint-Pareto-dominates y, 0 otherwise.
 * x is not better than y unless:
 * - x and y are both infeasible and x has a smaller sum of constraint violations, or
 * - x is feasible and y is not, or
 * - x and y are both feasible and x Pareto dominates y
 */
short constraintParetoDominates( double *objective_values_x, double constraint_value_x, double *objective_values_y, double constraint_value_y )
{
    short result;

    result = FALSE;

    if (IF_CHECK_CONSTRAINTS == TRUE) {
        if (constraint_value_x > 0) /* x is infeasible */
        {
            if (constraint_value_y > 0) /* Both are infeasible */
            {
                if (constraint_value_x < constraint_value_y)
                    result = TRUE;
            }
        } else /* x is feasible */
        {
            if (constraint_value_y > 0) /* x is feasible and y is not */
                result = TRUE;
            else /* Both are feasible */
                result = paretoDominates(objective_values_x, objective_values_y);
        }
    }
    else {
        result = paretoDominates(objective_values_x, objective_values_y);
    }

    return( result );
}

short constraintWeaklyParetoDominates( double *objective_values_x, double constraint_value_x, double *objective_values_y, double constraint_value_y  )
{
    short result;

    result = FALSE;

    if( constraint_value_x > 0 ) /* x is infeasible */
    {
        if( constraint_value_y > 0 ) /* Both are infeasible */
        {
            if(constraint_value_x  <= constraint_value_y )      
                result = TRUE;
        }
    }
    else /* x is feasible */
    {
        if( constraint_value_y > 0 ) /* x is feasible and y is not */
            result = TRUE;
        else /* Both are feasible */
            result = weaklyParetoDominates( objective_values_x, objective_values_y );
    }

    return( result );
}

/**
 * Returns 1 if x Pareto-dominates y, 0 otherwise.
 */
short paretoDominates( double *objective_values_x, double *objective_values_y )
{
    short strict;
    int   i, result;

    result = 1;
    strict = 0;

    for( i = 0; i < number_of_objectives; i++ )
    {
        if( fabs( objective_values_x[i] - objective_values_y[i] ) >= 0.00001 )
        {
            if(optimization[i] == MINIMIZATION)
            {
                if( objective_values_x[i] > objective_values_y[i] )
                {
                    result = 0;
                    break;
                }
                if( objective_values_x[i] < objective_values_y[i] )
                    strict = 1;    
            }
            else if(optimization[i] == MAXIMIZATION)
            {
                if( objective_values_x[i] < objective_values_y[i] )
                {
                    result = 0;
                    break;
                }
                if( objective_values_x[i] > objective_values_y[i] )
                    strict = 1;                    
            }
            
        }
    }

    if( strict == 0 && result == 1 )
        result = 0;

    return( result );
}

short weaklyParetoDominates( double *objective_values_x, double *objective_values_y )
{
    int   i, result;
    result = 1;

    for( i = 0; i < number_of_objectives; i++ )
    {
        if( fabs( objective_values_x[i] - objective_values_y[i] ) >= 0.00001 )
        {
            if(optimization[i] == MINIMIZATION)
            {
                if( objective_values_x[i] > objective_values_y[i] )
                {
                    result = 0;
                    break;
                }
            }
            else if(optimization[i] == MAXIMIZATION)
            {
                if( objective_values_x[i] < objective_values_y[i] )
                {
                    result = 0;
                    break;
                }                
            }

        }
    }
    
    return( result );
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-= Linkage Tree Learning -==-=-=-=-=-=-=-=-=-=-=*/
/**
 * Learn the linkage for a cluster (subpopulation).
 */
void learnLinkageTree( int cluster_index )
{
    char   done;
    int    i, j, k, a, b, c, r0, r1, *indices, *order,
         lt_index, factor_size, **mpm_new, *mpm_new_number_of_indices, mpm_new_length,
        *NN_chain, NN_chain_length;
    double p, *cumulative_probabilities, **S_matrix, mul0, mul1;

    /* Compute joint entropy matrix */
    for( i = 0; i < number_of_parameters; i++ )
    {
        for( j = i+1; j < number_of_parameters; j++ )
        {
            indices                  = (int *) Malloc( 2*sizeof( int ) );
            indices[0]               = i;
            indices[1]               = j;
//            cumulative_probabilities = estimateParametersForSingleBinaryMarginal( cluster_index, indices, 2, &factor_size );
            cumulative_probabilities = estimateParametersForArbitraryPairwiseMarginal( cluster_index, indices, 2, &factor_size );
            MI_matrix[i][j] = 0.0;
            for( k = 0; k < factor_size; k++ )
            {
                if( k == 0 )
                    p = cumulative_probabilities[k];
                else
                    p = cumulative_probabilities[k]-cumulative_probabilities[k-1];
                if( p > 0 )
                    MI_matrix[i][j] += -p*log2(p);
            }

            MI_matrix[j][i] = MI_matrix[i][j];

            free( indices );
            free( cumulative_probabilities );
        }
        indices                  = (int *) Malloc( 1*sizeof( int ) );
        indices[0]               = i;
//        cumulative_probabilities = estimateParametersForSingleBinaryMarginal( cluster_index, indices, 1, &factor_size );
        cumulative_probabilities = estimateParametersForArbitraryPairwiseMarginal( cluster_index, indices, 1, &factor_size );
        MI_matrix[i][i] = 0.0;
        for( k = 0; k < factor_size; k++ )
        {
            if( k == 0 )
                p = cumulative_probabilities[k];
            else
                p = cumulative_probabilities[k]-cumulative_probabilities[k-1];
            if( p > 0 )
                MI_matrix[i][i] += -p*log2(p);
        }

        free( indices );
        free( cumulative_probabilities );
    }

    /* Then transform into mutual information matrix MI(X,Y)=H(X)+H(Y)-H(X,Y) */
    for( i = 0; i < number_of_parameters; i++ )
        for( j = i+1; j < number_of_parameters; j++ )
        {
            MI_matrix[i][j] = MI_matrix[i][i] + MI_matrix[j][j] - MI_matrix[i][j];
            MI_matrix[j][i] = MI_matrix[i][j];
        }

    if (if_write_mi_matrix) writeToFileMIMatrix(cluster_index);

    /* Initialize MPM to the univariate factorization */
    order                 = createRandomOrdering( number_of_parameters );
    mpm                   = (int **) Malloc( number_of_parameters*sizeof( int * ) );
    mpm_number_of_indices = (int *) Malloc( number_of_parameters*sizeof( int ) );
    mpm_length            = number_of_parameters;
    for( i = 0; i < number_of_parameters; i++ )
    {
        indices                  = (int *) Malloc( 1*sizeof( int ) );
        indices[0]               = order[i];
        mpm[i]                   = indices;
        mpm_number_of_indices[i] = 1;
    }
    free( order );

    /* Initialize LT to the initial MPM */
    if( lt[cluster_index] != NULL )
    {
        for( i = 0; i < lt_length[cluster_index]; i++ )
            free( lt[cluster_index][i] );
        free( lt[cluster_index] );
        free( lt_number_of_indices[cluster_index] );
    }
    lt[cluster_index]                   = (int **) Malloc( (number_of_parameters+number_of_parameters-1)*sizeof( int * ) );
    lt_number_of_indices[cluster_index] = (int *) Malloc( (number_of_parameters+number_of_parameters-1)*sizeof( int ) );
    lt_length[cluster_index]            = number_of_parameters+number_of_parameters-1;
    lt_index             = 0;
    for( i = 0; i < mpm_length; i++ )
    {
        lt[cluster_index][lt_index]                   = mpm[i];
        lt_number_of_indices[cluster_index][lt_index] = mpm_number_of_indices[i];
        lt_index++;
    }

    /* Initialize similarity matrix */
    S_matrix = (double **) Malloc( number_of_parameters*sizeof( double * ) );
    for( i = 0; i < number_of_parameters; i++ )
        S_matrix[i] = (double *) Malloc( number_of_parameters*sizeof( double ) );
    for( i = 0; i < mpm_length; i++ )
        for( j = 0; j < mpm_length; j++ )
            S_matrix[i][j] = MI_matrix[mpm[i][0]][mpm[j][0]];
    for( i = 0; i < mpm_length; i++ )
        S_matrix[i][i] = 0;

    NN_chain        = (int *) Malloc( (number_of_parameters+2)*sizeof( int ) );
    NN_chain_length = 0;
    done            = FALSE;
    while( done == FALSE )
    {
        if( NN_chain_length == 0 )
        {
            NN_chain[NN_chain_length] = randomInt( mpm_length );
            NN_chain_length++;
        }

        while( NN_chain_length < 3 )
        {
            NN_chain[NN_chain_length] = determineNearestNeighbour( NN_chain[NN_chain_length-1], S_matrix, mpm_length );
            NN_chain_length++;
        }

        while( NN_chain[NN_chain_length-3] != NN_chain[NN_chain_length-1] )
        {
            NN_chain[NN_chain_length] = determineNearestNeighbour( NN_chain[NN_chain_length-1], S_matrix, mpm_length );
            if( ((TRUE == almosteq(S_matrix[NN_chain[NN_chain_length-1]][NN_chain[NN_chain_length]], S_matrix[NN_chain[NN_chain_length-1]][NN_chain[NN_chain_length-2]]))) && (NN_chain[NN_chain_length] != NN_chain[NN_chain_length-2]) )
                NN_chain[NN_chain_length] = NN_chain[NN_chain_length-2];
            NN_chain_length++;
        }
        r0 = NN_chain[NN_chain_length-2];
        r1 = NN_chain[NN_chain_length-1];
        if( r0 > r1 )
        {
            a  = r0;
            r0 = r1;
            r1 = a;
        }
        NN_chain_length -= 3;

        if( r1 < mpm_length ) // This test is required for exceptional cases in which the nearest-neighbor ordering has changed within the chain while merging within that chain
        {
            indices = (int *) Malloc( (mpm_number_of_indices[r0]+mpm_number_of_indices[r1])*sizeof( int ) );
  
            i = 0;
            for( j = 0; j < mpm_number_of_indices[r0]; j++ )
            {
                indices[i] = mpm[r0][j];
                i++;
            }
            for( j = 0; j < mpm_number_of_indices[r1]; j++ )
            {
                indices[i] = mpm[r1][j];
                i++;
            }
    
            lt[cluster_index][lt_index]                   = indices;
            lt_number_of_indices[cluster_index][lt_index] = mpm_number_of_indices[r0]+mpm_number_of_indices[r1];
            lt_index++;
  
            mul0 = ((double) mpm_number_of_indices[r0])/((double) mpm_number_of_indices[r0]+mpm_number_of_indices[r1]);
            mul1 = ((double) mpm_number_of_indices[r1])/((double) mpm_number_of_indices[r0]+mpm_number_of_indices[r1]);
            for( i = 0; i < mpm_length; i++ )
            {
                if( (i != r0) && (i != r1) )
                {
                    S_matrix[i][r0] = mul0*S_matrix[i][r0] + mul1*S_matrix[i][r1];
                    S_matrix[r0][i] = S_matrix[i][r0];
                }
            }
  
            mpm_new                   = (int **) Malloc( (mpm_length-1)*sizeof( int * ) );
            mpm_new_number_of_indices = (int *) Malloc( (mpm_length-1)*sizeof( int ) );
            mpm_new_length            = mpm_length-1;
            for( i = 0; i < mpm_new_length; i++ )
            {
                mpm_new[i]                   = mpm[i];
                mpm_new_number_of_indices[i] = mpm_number_of_indices[i];
            }
  
            mpm_new[r0]                   = indices;
            mpm_new_number_of_indices[r0] = mpm_number_of_indices[r0]+mpm_number_of_indices[r1];
            if( r1 < mpm_length-1 )
            {
                mpm_new[r1]                   = mpm[mpm_length-1];
                mpm_new_number_of_indices[r1] = mpm_number_of_indices[mpm_length-1];
  
                for( i = 0; i < r1; i++ )
                {
                    S_matrix[i][r1] = S_matrix[i][mpm_length-1];
                    S_matrix[r1][i] = S_matrix[i][r1];
                }
  
                for( j = r1+1; j < mpm_new_length; j++ )
                {
                    S_matrix[r1][j] = S_matrix[j][mpm_length-1];
                    S_matrix[j][r1] = S_matrix[r1][j];
                }
            }
  
            for( i = 0; i < NN_chain_length; i++ )
            {
                if( NN_chain[i] == mpm_length-1 )
                {
                    NN_chain[i] = r1;
                    break;
                }
            }
  
            free( mpm );
            free( mpm_number_of_indices );
            mpm                   = mpm_new;
            mpm_number_of_indices = mpm_new_number_of_indices;
            mpm_length            = mpm_new_length;
  
            if( mpm_length == 1 )
                done = TRUE;
        }
    }

    free( NN_chain );

    free( mpm_new );
    free( mpm_number_of_indices );

    for( i = 0; i < number_of_parameters; i++ )
        free( S_matrix[i] );
    free( S_matrix );
}

/**
 * Estimates the cumulative probability distribution of a
 * single binary marginal for a cluster (subpopulation).
 */
double *estimateParametersForSingleBinaryMarginal( int cluster_index, int *indices, int number_of_indices, int *factor_size )
{
    int     i, j, index, power_of_two;
    int *solution;
    double *result;

    *factor_size = (int) pow( 2, number_of_indices );
    result       = (double *) Malloc( (*factor_size)*sizeof( double ) );

    for( i = 0; i < (*factor_size); i++ )
        result[i] = 0.0;

    for( i = 0; i < population_cluster_sizes[cluster_index]; i++ ) 
    {
        index        = 0;
        power_of_two = 1;
        for( j = number_of_indices-1; j >= 0; j-- )
        {
            solution = population[population_indices_of_cluster_members[cluster_index][i]];
            index += (solution[indices[j]] == TRUE) ? power_of_two : 0;
            power_of_two *= 2;
        }

        result[index] += 1.0;
    }

    for( i = 0; i < (*factor_size); i++ )
        result[i] /= (double) population_cluster_sizes[cluster_index];

    for( i = 1; i < (*factor_size); i++ )
        result[i] += result[i-1];

    result[(*factor_size)-1] = 1.0;

    return( result );
}

/**
 * Estimates the cumulative probability distribution of a
 * marginal for one or two arbitrary-sized variables in a cluster (subpopulation).
 */
double *estimateParametersForArbitraryPairwiseMarginal( int cluster_index, int *indices, int number_of_indices, int *factor_size )
{
    int     i, j, index, power_of_two;
    int *solution;
    double *result;

    int n_possible_vals_first_gene = alphabet[indices[0]] - alphabet_lower_bound[indices[0]];
    if (number_of_indices == 2) {
        *factor_size = (int) n_possible_vals_first_gene * (alphabet[indices[1]] - alphabet_lower_bound[indices[1]]);
    }
    else *factor_size = (int) n_possible_vals_first_gene;

    result       = (double *) Malloc( (*factor_size)*sizeof( double ) );

    for( i = 0; i < (*factor_size); i++ )
        result[i] = 0.0;

    for( i = 0; i < population_cluster_sizes[cluster_index]; i++ )
    {
        solution = population[population_indices_of_cluster_members[cluster_index][i]];
        if (number_of_indices == 2) {
            index = (solution[indices[0]] - alphabet_lower_bound[indices[0]]) *
                    (alphabet[indices[1]] - alphabet_lower_bound[indices[1]]) +
                    (solution[indices[1]] - alphabet_lower_bound[indices[1]]);
        }
        else {
            index = solution[indices[0]] - alphabet_lower_bound[indices[0]];
        }
        result[index] += 1.0;
    }

    for( i = 0; i < (*factor_size); i++ )
        result[i] /= (double) population_cluster_sizes[cluster_index];

    for( i = 1; i < (*factor_size); i++ )
        result[i] += result[i-1];

    result[(*factor_size)-1] = 1.0;

    return( result );
}

/**
 * Determines nearest neighbour according to similarity values.
 */
int determineNearestNeighbour( int index, double **S_matrix, int mpm_length )
{
    int i, result;

    result = 0;
    if( result == index )
        result++;
    for( i = 1; i < mpm_length; i++ )
    {
        if( ((S_matrix[index][i] > S_matrix[index][result]) || ((TRUE == almosteq(S_matrix[index][i],S_matrix[index][result])) && (mpm_number_of_indices[i] < mpm_number_of_indices[result]))) && (i != index) )
        result = i;
    }

    return( result );
}

void printLTStructure( int cluster_index )
{
    int i, j;

    for( i = 0; i < lt_length[cluster_index]; i++ )
    {
        printf("[");
        for( j = 0; j < lt_number_of_indices[cluster_index][i]; j++ )
        {
            printf("%d",lt[cluster_index][i][j]);
            if( j < lt_number_of_indices[cluster_index][i]-1 )
                printf(" ");
        }
        printf("]\n");
    }
    printf("\n");
    fflush( stdout );
}

void writeLinkageTreeToFile(int cluster_index)
{
    int i, j;
    char c[1000];
    FILE *f;

    sprintf(c,"linkage_tree_evaluation_%d_cluster_%d.dat",number_of_evaluations, cluster_index);

    f = fopen(c,"w");
    if( f != NULL )
    {
        for( i = 0; i < lt_length[cluster_index]; i++ )
        {
            fprintf(f,"[");
            for( j = 0; j < lt_number_of_indices[cluster_index][i]; j++ )
            {
                fprintf(f,"%d",lt[cluster_index][i][j]);
                if( j < lt_number_of_indices[cluster_index][i]-1 )
                    fprintf(f," ");
            }
            fprintf(f,"]\n");
        }
        fprintf(f,"\n");
    }
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- MO-GOMEA -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Performs initializations that are required before starting a run.
 */
void initialize()
{
    number_of_populations++;

    initializeMemory();

    initializePopulationAndFitnessValues();

    computeObjectiveRanges();
}

/**
 * Initializes the memory.
 */
void initializeMemory( void )
{
    int i;

    objective_ranges         = (double *) Malloc( population_size*sizeof( double ) );
    population               = (int **) Malloc( population_size*sizeof( int * ) );
    objective_values         = (double **) Malloc( population_size*sizeof( double * ) );
    constraint_values        = (double *) Malloc( population_size*sizeof( double ) );
    
    for( i = 0; i < population_size; i++ )
    {
        population[i]        = (int *) Malloc( number_of_parameters*sizeof( int ) );
        objective_values[i]  = (double *) Malloc( number_of_objectives*sizeof( double ) );
    }   

    t_NIS                    = 0;
    number_of_generations    = 0;
}
/**
 * Initializes the population and the objective values by randomly
 * generation n solutions.
 */
void initializePopulationAndFitnessValues()
{
    int i, j;
    char if_read_from_file_successfully = FALSE;
    int tmp = 17;

    for( i = 0; i < population_size; i++ )
    {
        if_read_from_file_successfully = FALSE;
        if (if_use_initialization_genomes == TRUE) {
            if (!feof (file_initialization_genomes)) {
                for( j = 0; j < number_of_parameters; j++ ) {
                    fscanf(file_initialization_genomes, "%d", &tmp);
                    population[i][j] = (int) tmp;
                }
                if_read_from_file_successfully = TRUE;
            }
            else {
                fclose (file_initialization_genomes);
                if_use_initialization_genomes = FALSE;
            }
        }
        if (if_read_from_file_successfully == FALSE) {
            for (j = 0; j < number_of_parameters; j++)
                population[i][j] = (int) (randomInt(alphabet[j] - alphabet_lower_bound[j]) + alphabet_lower_bound[j]);
        }
        evaluateIndividual( population[i], objective_values[i],  &(constraint_values[i]), NOT_EXTREME_CLUSTER );
        updateElitistArchive( population[i], objective_values[i], constraint_values[i]);
    }
}

/**
 * Computes the ranges of all fitness values
 * of all solutions currently in the populations.
 */
void computeObjectiveRanges( void )
{
    int    i, j;
    double low, high;

    for( j = 0; j < number_of_objectives; j++ )
    {
        low  = objective_values[0][j];
        high = objective_values[0][j];

        for( i = 0; i < population_size; i++ )
        {
            if( objective_values[i][j] < low )
                low = objective_values[i][j];
            if( objective_values[i][j] > high )
                high = objective_values[i][j];
        }

        objective_ranges[j] = high - low;
    }
}

void learnLinkageOnCurrentPopulation()
{
    int i, j, k, size_of_one_cluster;

    initializeClusters();

    population_indices_of_cluster_members = clustering(objective_values, population_size, number_of_objectives, 
                            number_of_mixing_components, &size_of_one_cluster);
    population_cluster_sizes = (int*)Malloc(number_of_mixing_components*sizeof(int));
    for(k = 0; k < number_of_mixing_components; k++)    
        population_cluster_sizes[k] = size_of_one_cluster;

    // find extreme-region clusters
    determineExtremeClusters();

    // learn linkage tree for every cluster
    for( i = 0; i < number_of_mixing_components; i++ ) {
        learnLinkageTree( i );
        if (if_write_linkage_tree == TRUE) writeLinkageTreeToFile(i);
    }
}

int** clustering(double **objective_values_pool, int pool_size, int number_of_dimensions, 
                    int number_of_clusters, int *pool_cluster_size )
{
    int i, j, k, j_min, number_to_select,
        *pool_indices_of_leaders, *k_means_cluster_sizes, **pool_indices_of_cluster_members_k_means,
        **pool_indices_of_cluster_members, size_of_one_cluster;
    double distance, distance_smallest, epsilon,
            **objective_values_pool_scaled, **objective_means_scaled_new, *distances_to_cluster;
            
    if (number_of_clusters > 1) {
        *pool_cluster_size = (2 * pool_size) / number_of_clusters;
    }
    else
    {
        *pool_cluster_size   = pool_size;
        pool_indices_of_cluster_members = (int**)Malloc(number_of_clusters * sizeof(int*));
        pool_indices_of_cluster_members[0] = (int*)Malloc(pool_size * sizeof(int));
        for(i = 0; i < pool_size; i++)
            pool_indices_of_cluster_members[0][i] = i;
        return (pool_indices_of_cluster_members);
    }

    size_of_one_cluster  = *pool_cluster_size;

    /* Determine the leaders */
    objective_values_pool_scaled = (double **) Malloc( pool_size*sizeof( double * ) );
    for( i = 0; i < pool_size; i++ )
        objective_values_pool_scaled[i] = (double *) Malloc( number_of_dimensions*sizeof( double ) );
    for( i = 0; i < pool_size; i++ )
        for( j = 0; j < number_of_dimensions; j++ )
            objective_values_pool_scaled[i][j] = objective_values_pool[i][j]/objective_ranges[j];

    /* Heuristically find k far-apart leaders */
    number_to_select             = number_of_clusters;
    pool_indices_of_leaders = greedyScatteredSubsetSelection( objective_values_pool_scaled, pool_size, number_of_dimensions, number_to_select );

    for( i = 0; i < number_of_clusters; i++ )
        for( j = 0; j < number_of_dimensions; j++ )
            objective_means_scaled[i][j] = objective_values_pool[pool_indices_of_leaders[i]][j]/objective_ranges[j];

    /* Perform k-means clustering with leaders as initial mean guesses */
    objective_means_scaled_new = (double **) Malloc( number_of_clusters*sizeof( double * ) );
    for( i = 0; i < number_of_clusters; i++ )
        objective_means_scaled_new[i] = (double *) Malloc( number_of_dimensions*sizeof( double ) );

    pool_indices_of_cluster_members_k_means = (int **) Malloc( number_of_clusters*sizeof( int * ) );
    for( i = 0; i < number_of_clusters; i++ )
        pool_indices_of_cluster_members_k_means[i] = (int *) Malloc( pool_size*sizeof( int ) );

    k_means_cluster_sizes = (int *) Malloc( number_of_clusters*sizeof( int ) );

    epsilon = 1e+308;
    while( epsilon > 1e-10 )
    {
        for( j = 0; j < number_of_clusters; j++ )
        {
            k_means_cluster_sizes[j] = 0;
            for( k = 0; k < number_of_dimensions; k++ )
                objective_means_scaled_new[j][k] = 0.0;
        }

        for( i = 0; i < pool_size; i++ )
        {
            j_min             = -1;
            distance_smallest = -1;
            for( j = 0; j < number_of_clusters; j++ )
            {
                distance = distanceEuclidean( objective_values_pool_scaled[i], objective_means_scaled[j], number_of_dimensions );
                if( (distance_smallest < 0) || (distance < distance_smallest) )
                {
                    j_min             = j;
                    distance_smallest = distance;
                }
            }
            pool_indices_of_cluster_members_k_means[j_min][k_means_cluster_sizes[j_min]] = i;
            for( k = 0; k < number_of_dimensions; k++ )
                objective_means_scaled_new[j_min][k] += objective_values_pool_scaled[i][k];
            k_means_cluster_sizes[j_min]++;
        }

        for( j = 0; j < number_of_clusters; j++ )
            for( k = 0; k < number_of_dimensions; k++ )
                objective_means_scaled_new[j][k] /= (double) k_means_cluster_sizes[j];

        epsilon = 0;
        for( j = 0; j < number_of_clusters; j++ )
        {
            epsilon += distanceEuclidean( objective_means_scaled[j], objective_means_scaled_new[j], number_of_dimensions );
            for( k = 0; k < number_of_dimensions; k++ )
                objective_means_scaled[j][k] = objective_means_scaled_new[j][k];
        }
    }

    /* Shrink or grow the result of k-means clustering to get the final equal-sized clusters */
    pool_indices_of_cluster_members = (int**)Malloc(number_of_clusters * sizeof(int*));
    distances_to_cluster = (double *) Malloc( pool_size*sizeof( double ) );
    for( i = 0; i < number_of_clusters; i++ )
    {
        for( j = 0; j < pool_size; j++ )
            distances_to_cluster[j] = distanceEuclidean( objective_values_pool_scaled[j], objective_means_scaled[i], number_of_dimensions );

        for( j = 0; j < k_means_cluster_sizes[i]; j++ )
            distances_to_cluster[pool_indices_of_cluster_members_k_means[i][j]] = 0;

        pool_indices_of_cluster_members[i]          = mergeSort( distances_to_cluster, pool_size );
    }

    // Re-calculate clusters' means
    for( i = 0; i < number_of_clusters; i++)
    {
        for (j = 0; j < number_of_dimensions; j++)
            objective_means_scaled[i][j] = 0.0;

        for (j = 0; j < size_of_one_cluster; j++)
        {
            for( k = 0; k < number_of_dimensions; k++)
                objective_means_scaled[i][k] +=
                    objective_values_pool_scaled[pool_indices_of_cluster_members[i][j]][k];
        }

        for (j = 0; j < number_of_dimensions; j++)
        {
            objective_means_scaled[i][j] /= (double) size_of_one_cluster;
        }
    }

    free( distances_to_cluster );
    free( k_means_cluster_sizes );
    for( i = 0; i < number_of_clusters; i++ )
        free( pool_indices_of_cluster_members_k_means[i] );
    free( pool_indices_of_cluster_members_k_means );
    for( i = 0; i < number_of_clusters; i++ )
        free( objective_means_scaled_new[i] );
    free( objective_means_scaled_new );
    for( i = 0; i < pool_size; i++ )
        free( objective_values_pool_scaled[i] );
    free( objective_values_pool_scaled );
    free( pool_indices_of_leaders );   

    return (pool_indices_of_cluster_members);
}
/**
 * Selects n points from a set of points. A
 * greedy heuristic is used to find a good
 * scattering of the selected points. First,
 * a point is selected with a maximum value
 * in a randomly selected dimension. The
 * remaining points are selected iteratively.
 * In each iteration, the point selected is
 * the one that maximizes the minimal distance
 * to the points selected so far.
 */
int *greedyScatteredSubsetSelection( double **points, int number_of_points, int number_of_dimensions, int number_to_select )
{
    int     i, index_of_farthest, random_dimension_index, number_selected_so_far,
            *indices_left, *result;
    double *nn_distances, distance_of_farthest, value;

    if( number_to_select > number_of_points )
    {
        printf("\n");
        printf("Error: greedyScatteredSubsetSelection asked to select %d solutions from set of size %d.", number_to_select, number_of_points);
        printf("\n\n");

        exit( 0 );
    }

    result = (int *) Malloc( number_to_select*sizeof( int ) );

    indices_left = (int *) Malloc( number_of_points*sizeof( int ) );
    for( i = 0; i < number_of_points; i++ )
        indices_left[i] = i;

    /* Find the first point: maximum value in a randomly chosen dimension */
    random_dimension_index = randomInt( number_of_dimensions );

    index_of_farthest    = 0;
    distance_of_farthest = points[indices_left[index_of_farthest]][random_dimension_index];
    for( i = 1; i < number_of_points; i++ )
    {
        if( points[indices_left[i]][random_dimension_index] > distance_of_farthest )
        {
            index_of_farthest    = i;
            distance_of_farthest = points[indices_left[i]][random_dimension_index];
        }
    }

    number_selected_so_far          = 0;
    result[number_selected_so_far]  = indices_left[index_of_farthest];
    indices_left[index_of_farthest] = indices_left[number_of_points-number_selected_so_far-1];
    number_selected_so_far++;

    /* Then select the rest of the solutions: maximum minimum
     * (i.e. nearest-neighbour) distance to so-far selected points */
    nn_distances = (double *) Malloc( number_of_points*sizeof( double ) );
    for( i = 0; i < number_of_points-number_selected_so_far; i++ )
        nn_distances[i] = distanceEuclidean( points[indices_left[i]], points[result[number_selected_so_far-1]], number_of_dimensions );

    while( number_selected_so_far < number_to_select )
    {
        index_of_farthest    = 0;
        distance_of_farthest = nn_distances[0];
        for( i = 1; i < number_of_points-number_selected_so_far; i++ )
        {
            if( nn_distances[i] > distance_of_farthest )
            {
                index_of_farthest    = i;
                distance_of_farthest = nn_distances[i];
            }
        }

        result[number_selected_so_far]  = indices_left[index_of_farthest];
        indices_left[index_of_farthest] = indices_left[number_of_points-number_selected_so_far-1];
        nn_distances[index_of_farthest] = nn_distances[number_of_points-number_selected_so_far-1];
        number_selected_so_far++;

        for( i = 0; i < number_of_points-number_selected_so_far; i++ )
        {
            value = distanceEuclidean( points[indices_left[i]], points[result[number_selected_so_far-1]], number_of_dimensions );
            if( value < nn_distances[i] )
                nn_distances[i] = value;
        }
    }

    free( nn_distances );
    free( indices_left );
    return( result );
}

void determineExtremeClusters()
{
    int i,j, index_best, a,b,c, *order;
    // find extreme clusters
    order = createRandomOrdering(number_of_objectives);
        
    for (i = 0; i < number_of_mixing_components; i++)
        which_extreme[i] = -1;  // not extreme cluster
    
    if(number_of_mixing_components > 1)
    {
        for (i = 0; i < number_of_objectives; i++)
        {
            index_best = -1;
        
            for (j = 0; j < number_of_mixing_components; j++)
            {
                if(optimization[order[i]] == MINIMIZATION)
                {
                    if( ((index_best == -1) || (objective_means_scaled[j][order[i]] < objective_means_scaled[index_best][order[i]]) )&&
                        (which_extreme[j] == -1) )
                        index_best = j;
                }
                else if(optimization[order[i]] == MAXIMIZATION)
                {
                    if( ((index_best == -1) || (objective_means_scaled[j][order[i]] > objective_means_scaled[index_best][order[i]]) )&&
                        (which_extreme[j] == -1) )
                        index_best = j;
                }
            }
            which_extreme[index_best] = order[i];
        }
    }

    if (FALSE) {
        for (i = 0; i < number_of_mixing_components; i++)
            printf("%d %d ; ", i, which_extreme[i]);
        printf("\n");
    }

    free(order);
}

void initializeClusters()
{
    int i;
    lt                            = (int ***) Malloc( number_of_mixing_components*sizeof( int ** ) );
    lt_length                     = (int *) Malloc( number_of_mixing_components*sizeof( int ) );
    lt_number_of_indices          = (int **) Malloc( number_of_mixing_components*sizeof( int *) );
    for( i = 0; i < number_of_mixing_components; i++)
    {
        lt[i]                     = NULL;
        lt_number_of_indices[i]   = NULL;
        lt_length[i]              = 0;
    }

    which_extreme                 = (int*)Malloc(number_of_mixing_components*sizeof(int));

    objective_means_scaled        = (double **) Malloc( number_of_mixing_components*sizeof( double * ) );    
    for( i = 0; i < number_of_mixing_components; i++ )
        objective_means_scaled[i] = (double *) Malloc( number_of_objectives*sizeof( double ) );
}

void ezilaitiniClusters()
{
    int i, j;

    if(lt == NULL)
        return;

    for( i = 0; i < number_of_mixing_components; i++ )
    {
        if( lt[i] != NULL )
        {
            for( j = 0; j < lt_length[i]; j++ )
                free( lt[i][j] );
            free( lt[i] );
            free( lt_number_of_indices[i] );
        }
    }

    free( lt ); lt = NULL;
    free( lt_length );
    free( lt_number_of_indices );

    free(which_extreme);

    for(i = 0; i < number_of_mixing_components; i++)
        free(objective_means_scaled[i]);
    free( objective_means_scaled );
}

void improveCurrentPopulation( void )
{
    int     i, j, k, j_min, cluster_index, objective_index, number_of_cluster,
            *sum_cluster, *clusters ;
    double *objective_values_scaled,
          distance, distance_smallest;

    offspring_size                  = population_size;
    offspring                       = (int**)Malloc(offspring_size*sizeof(int*));
    objective_values_offspring      = (double**)Malloc(offspring_size*sizeof(double*));
    constraint_values_offspring     = (double*)Malloc(offspring_size*sizeof(double));
    
    for(i = 0; i < offspring_size; i++)
    {
        offspring[i]                = (int*)Malloc(number_of_parameters*sizeof(int));
        objective_values_offspring[i]  = (double*)Malloc(number_of_objectives*sizeof(double));
    }

    objective_values_scaled = (double *) Malloc( number_of_objectives*sizeof( double ) );
    sum_cluster = (int*)Malloc(number_of_mixing_components*sizeof(int));

    for(i = 0; i < number_of_mixing_components; i++)
        sum_cluster[i] = 0;

    elitist_archive_front_changed = FALSE;
    for( i = 0; i < population_size; i++ )
    {
        number_of_cluster = 0;
        clusters = (int*)Malloc(number_of_mixing_components*sizeof(int));
        for(j = 0; j < number_of_mixing_components; j++)
        {
            for (k = 0; k < population_cluster_sizes[j]; k++)
            {
                if(population_indices_of_cluster_members[j][k] == i)
                {
                    clusters[number_of_cluster] = j;
                    number_of_cluster++;
                    break;
                }
            }
        }
        if(number_of_cluster > 0)
            cluster_index = clusters[randomInt(number_of_cluster)];
        else
        {
            for( j = 0; j < number_of_objectives; j++ )
                objective_values_scaled[j] = objective_values[i][j]/objective_ranges[j];

            distance_smallest = -1;
            j_min = -1;
            for( j = 0; j < number_of_mixing_components; j++ )
            {
                distance = distanceEuclidean( objective_values_scaled, objective_means_scaled[j], number_of_objectives );
                if( (distance_smallest < 0) || (distance < distance_smallest) )
                {
                    j_min = j;
                    distance_smallest  = distance;
                }
            
            }

            cluster_index = j_min;
        }

        sum_cluster[cluster_index]++;
        if(which_extreme[cluster_index] == -1)
        {
            performMultiObjectiveGenepoolOptimalMixing( cluster_index, population[i], objective_values[i], constraint_values[i],
                                offspring[i], objective_values_offspring[i], &(constraint_values_offspring[i]));
        }
        else
        {
            objective_index = which_extreme[cluster_index];
            performSingleObjectiveGenepoolOptimalMixing(cluster_index, objective_index, population[i], objective_values[i], constraint_values[i], 
                                offspring[i], objective_values_offspring[i], &(constraint_values_offspring[i]));
        }
        free(clusters);
        if (max_evals_reached == TRUE) {break;}
    }

    free( objective_values_scaled ); free( sum_cluster );

    if(!elitist_archive_front_changed)
        t_NIS++;
    else
        t_NIS = 0;
}

void copyValuesFromDonorToOffspring(int *solution, int *donor, int cluster_index, int linkage_group_index)
{
    int i, parameter_index;
    if (ENABLE_DEBUG_OUTPUT == TRUE) printf("   param_idx=");
    for (i = 0; i < lt_number_of_indices[cluster_index][linkage_group_index]; i++)
    {
        parameter_index = lt[cluster_index][linkage_group_index][i];
        if (ENABLE_DEBUG_OUTPUT == TRUE) printf("(%d,%d,%d)", parameter_index,solution[parameter_index],donor[parameter_index]);
        solution[parameter_index] = donor[parameter_index];    
    }
    if (ENABLE_DEBUG_OUTPUT == TRUE) printf("\n");
}

void copyFromAToB(int *solution_a, double *obj_a, double con_a, int *solution_b, double *obj_b, double *con_b)
{
    int i;
    for (i = 0; i < number_of_parameters; i++)
        solution_b[i] = solution_a[i];
    for (i = 0; i < number_of_objectives; i++)
        obj_b[i] = obj_a[i];
    *con_b = con_a;
}

void mutateSolution(int *solution, int lt_factor_index, int cluster_index)
{
    double mutation_rate, prob;
    int i, parameter_index;

    if(use_pre_mutation == FALSE && use_pre_adaptive_mutation == FALSE)
        return;
    if (ENABLE_DEBUG_OUTPUT == TRUE) printf("Never get here\n");
    mutation_rate = 0.0;
    if(use_pre_mutation == TRUE)
        mutation_rate = 1.0/((double)number_of_parameters);
    else if(use_pre_adaptive_mutation == TRUE)
        mutation_rate = 1.0/((double)lt_number_of_indices[cluster_index][lt_factor_index]);


    for(i = 0; i < lt_number_of_indices[cluster_index][lt_factor_index]; i++)
    {
        prob = randomRealUniform01();
        if (ENABLE_DEBUG_OUTPUT == TRUE) printf("  prob=%f\n", prob);
        if(prob < mutation_rate)
        {
            parameter_index = lt[cluster_index][lt_factor_index][i];
            if(solution[parameter_index] == 0) 
                solution[parameter_index] = 1;
            else
                solution[parameter_index] = 0;
        }
    }
    
}
/**
 * Multi-objective Gene-pool Optimal Mixing
 * Construct an offspring from a parent solution in a middle-region cluster.
 */
void performMultiObjectiveGenepoolOptimalMixing( int cluster_index, int *parent, double *parent_obj, double parent_con,
                            int *result, double *obj,  double *con)
{
    char    is_unchanged, changed, is_improved, is_new_nondominated_point, is_dominated_by_archive;
    int     *backup, *donor, i, j, index, donor_index, position_of_existed_member, *order, linkage_group_index, number_of_linkage_sets;
    double  *obj_backup, con_backup;

    /* Clone the parent solution. */
    copyFromAToB(parent, parent_obj, parent_con, result, obj, con);
    
    /* Create a backup version of the parent solution. */
    backup = (int *) Malloc( number_of_parameters*sizeof( int ) );
    obj_backup = (double *) Malloc( number_of_objectives*sizeof( double ) );
    copyFromAToB(result, obj, *con, backup, obj_backup, &con_backup);

    number_of_linkage_sets = lt_length[cluster_index] - 1; /* Remove root from the linkage tree. */
    order = createRandomOrdering(number_of_linkage_sets);
    
    /* Traverse the linkage tree for Gene-pool Optimal Mixing */
    changed = FALSE;
    for( i = 0; i < number_of_linkage_sets; i++ )
    {
        linkage_group_index = order[i];

        donor_index = randomInt( population_cluster_sizes[cluster_index] );

        donor = population[population_indices_of_cluster_members[cluster_index][donor_index]];
        copyValuesFromDonorToOffspring(result, donor, cluster_index, linkage_group_index);     
        mutateSolution(result, linkage_group_index, cluster_index);

        /* Check if the new intermediate solution is different from the previous state. */
        is_unchanged = TRUE;
        for( j = 0; j < lt_number_of_indices[cluster_index][linkage_group_index]; j++ )
        {
            if( backup[lt[cluster_index][linkage_group_index][j]] != result[lt[cluster_index][linkage_group_index][j]] )
            {
                is_unchanged = FALSE;
                break;
            }
        }

        if( is_unchanged == FALSE )
        {
            is_improved = FALSE;            
            evaluateIndividual(result, obj, con, NOT_EXTREME_CLUSTER);
            updateElitistArchiveWithReplacementOfExistedMember(result, obj, *con, &is_new_nondominated_point, &is_dominated_by_archive);

            /* Check for weak Pareto domination. */
            if ( constraintWeaklyParetoDominates( obj, *con, obj_backup, con_backup) )
                is_improved = TRUE;

            /* Check if the new intermediate solution is dominated by any solution in the archive. 
                Note that if the new intermediate solution is a point in the archive, it is NOT considered to be dominated by the archive.*/
            if ( !is_dominated_by_archive )
                is_improved = TRUE;
            
            if ( is_improved )
            {
                changed = TRUE;
                copyFromAToB(result, obj, *con,  backup, obj_backup, &con_backup);
            }
            else
                copyFromAToB(backup, obj_backup, con_backup, result, obj, con);

            if (max_evals_reached == TRUE) {break;}
        }
    }
    free(order);

    /* Forced Improvement */
    if  ((max_evals_reached == FALSE) && (  (!changed) || (t_NIS > (1+floor(log10(population_size))))  ))
    {
        changed = FALSE;
        order = createRandomOrdering(number_of_linkage_sets);
        /* Perform another round of Gene-pool Optimal Mixing with the donors randomly selected from the archive. */
        for(i = 0; i < number_of_linkage_sets; i++)
        {
            donor_index = randomInt(elitist_archive_size);
            linkage_group_index = order[i];
            copyValuesFromDonorToOffspring(result, elitist_archive[donor_index], cluster_index, linkage_group_index);
            mutateSolution(result, linkage_group_index, cluster_index);

            /* Check if the new intermediate solution is different from the previous state. */
            is_unchanged = TRUE;    
            for( j = 0; j < lt_number_of_indices[cluster_index][linkage_group_index]; j++ )
            {
                if( backup[lt[cluster_index][linkage_group_index][j]] != result[lt[cluster_index][linkage_group_index][j]] )
                {
                    is_unchanged = FALSE;
                    break;
                }
            }           

            if( is_unchanged == FALSE )
            {
                is_improved = FALSE;

                evaluateIndividual(result, obj, con, NOT_EXTREME_CLUSTER);
                updateElitistArchiveWithReplacementOfExistedMember(result, obj, *con, &is_new_nondominated_point, &is_dominated_by_archive);

                /* Check for (strict) Pareto domination. */
                if ( constraintParetoDominates( obj, *con, obj_backup, con_backup) )
                    is_improved = TRUE;

                /* Check if a truly new non-dominated solution is created. */
                if(is_new_nondominated_point)
                    is_improved = TRUE;
                
                if ( is_improved )
                {
                    changed = TRUE;
                    copyFromAToB(result, obj, *con,  backup, obj_backup, &con_backup);
                    break;
                }
                else
                    copyFromAToB(backup, obj_backup, con_backup, result, obj, con);
                if (max_evals_reached == TRUE) {break;}
            }
        }
        free(order);

        if(!changed)
        {
            donor_index = randomInt( elitist_archive_size );

            copyFromAToB(elitist_archive[donor_index], elitist_archive_objective_values[donor_index], 
                    elitist_archive_constraint_values[donor_index], 
                    result, obj, con);
        }
    }

    free( backup ); free( obj_backup ); 
}
/**
 * Single-objective Gene-pool Optimal Mixing
 * Construct an offspring from a parent solution in an extreme-region cluster.
 */
void performSingleObjectiveGenepoolOptimalMixing( int cluster_index, int objective_index, 
                                int *parent, double *parent_obj, double parent_con,
                                int *result, double *obj, double *con )
{
    char    is_unchanged, changed, is_improved, is_new_nondominated_point, is_dominated_by_archive;
    int     *backup, *donor, *elitist_copy, i, j, index, donor_index, number_of_linkage_sets, linkage_group_index, *order;
    double  *obj_backup, con_backup;
    if (ENABLE_DEBUG_OUTPUT == TRUE) printf("Single-objective GOM begin!\n");
    if (ENABLE_DEBUG_OUTPUT == TRUE) printf("paren=");
    for(int jj = 0; jj < number_of_parameters; jj++) {
        if (ENABLE_DEBUG_OUTPUT == TRUE) printf("%d", parent[jj]);
    }
    if (ENABLE_DEBUG_OUTPUT == TRUE) printf("\n");

    /* Clone the parent solution. */
    copyFromAToB(parent, parent_obj, parent_con, result, obj, con);

    /* Create a backup version of the parent solution. */
    backup = (int *) Malloc( number_of_parameters*sizeof( int ) );
    obj_backup = (double *) Malloc( number_of_objectives*sizeof( double ) );
    copyFromAToB(result, obj, *con, backup, obj_backup, &con_backup);

    number_of_linkage_sets = lt_length[cluster_index] - 1; /* Remove root from the linkage tree. */
    
    order = createRandomOrdering(number_of_linkage_sets);

    /* Traverse the linkage tree for Gene-pool Optimal Mixing */
    changed = FALSE;
    for( i = 0; i < number_of_linkage_sets; i++ )
    {
        linkage_group_index = order[i];
        donor_index = randomInt( population_cluster_sizes[cluster_index] );
        
        donor = population[population_indices_of_cluster_members[cluster_index][donor_index]];
        if (ENABLE_DEBUG_OUTPUT == TRUE) {
            printf("resul=");
            for (int jj = 0; jj < number_of_parameters; jj++) {
                printf("%d", result[jj]);
            }
            printf("\n");
            printf("donor=");
            for (int jj = 0; jj < number_of_parameters; jj++) {
                printf("%d", donor[jj]);
            }
            printf("\n");
        }
        copyValuesFromDonorToOffspring(result, donor, cluster_index, linkage_group_index);        
        mutateSolution(result, linkage_group_index, cluster_index);

        if (ENABLE_DEBUG_OUTPUT == TRUE) {
            printf("mutant=");
            for (int jj = 0; jj < number_of_parameters; jj++) {
                printf("%d", result[jj]);
            }
            printf("\n");
        }


        /* Check if the new intermediate solution is different from the previous state. */
        is_unchanged = TRUE;
        for( j = 0; j < lt_number_of_indices[cluster_index][linkage_group_index]; j++ )
        {
            if( backup[lt[cluster_index][linkage_group_index][j]] != result[lt[cluster_index][linkage_group_index][j]] )
            {
                is_unchanged = FALSE;
                break;
            }
        }

        if (ENABLE_DEBUG_OUTPUT == TRUE) printf("        is_unchanged=%d\n", is_unchanged);

        if( is_unchanged == FALSE )
        {
            is_improved = FALSE;
            evaluateIndividual(result, obj, con, objective_index);
            updateElitistArchiveWithReplacementOfExistedMember(result, obj, *con, &is_new_nondominated_point, &is_dominated_by_archive);

            if (betterFitness(obj, *con, obj_backup, con_backup, objective_index))  is_improved = TRUE;
            if (ENABLE_DEBUG_OUTPUT == TRUE) printf("        is_improved=%d", is_improved);
            if (equalFitness(obj, *con, obj_backup, con_backup, objective_index)) is_improved = TRUE;
            if (ENABLE_DEBUG_OUTPUT == TRUE) printf("        is_improved=%d\n", is_improved);


            if ( is_improved )
            {
                changed = TRUE;
                copyFromAToB(result, obj, *con,  backup, obj_backup, &con_backup);
            }
            else
                copyFromAToB(backup, obj_backup, con_backup, result, obj, con);
            if (max_evals_reached == TRUE) {break;}
        }
    }
    free(order);

    if (ENABLE_DEBUG_OUTPUT == TRUE) {
        printf("objes=\n");
        for (j = 0; j < elitist_archive_size; j++) {
            for (int jjj = 0; jjj < number_of_parameters; jjj++) {
                printf("%d", elitist_archive[j][jjj]);
            }
            printf("\n");
        }
    }

    elitist_copy = (int*)Malloc(number_of_parameters*sizeof(int));
    if (ENABLE_DEBUG_OUTPUT == TRUE) printf("Start forced improvements\n");
    /* Forced Improvement*/
    if ((max_evals_reached == FALSE) && ( (!changed) || ( t_NIS > (1+floor(log10(population_size))) ) ))
    {
        changed = FALSE;
        /* Find in the archive the solution having the best value in the corresponding objective. */
        donor_index = 0;
        if (ENABLE_DEBUG_OUTPUT == TRUE) printf("objes=\n");
        for (j= 0; j < elitist_archive_size; j++) {
            if (optimization[objective_index] == MINIMIZATION) {
                if (elitist_archive_objective_values[j][objective_index] <
                    elitist_archive_objective_values[donor_index][objective_index])
                    donor_index = j;
            } else if (optimization[objective_index] == MAXIMIZATION) {
                if (elitist_archive_objective_values[j][objective_index] >
                    elitist_archive_objective_values[donor_index][objective_index]) {
                    donor_index = j;
                }
            }
            if (ENABLE_DEBUG_OUTPUT == TRUE) {
                for (int jjj = 0; jjj < number_of_parameters; jjj++) {
                    printf("%d", elitist_archive[j][jjj]);
                }
                printf("\n");
            }
        }

        if (ENABLE_DEBUG_OUTPUT == TRUE) {
            printf("resul=");
            for (int jj = 0; jj < number_of_parameters; jj++) {
                printf("%d", result[jj]);
            }
            printf("\n");

            printf("eliti=");
        }
        for (j = 0; j < number_of_parameters; j++) {
            elitist_copy[j] = elitist_archive[donor_index][j];
            if (ENABLE_DEBUG_OUTPUT == TRUE) printf("%d", elitist_copy[j]);
        }
        if (ENABLE_DEBUG_OUTPUT == TRUE) printf("\n");

        /* Perform Gene-pool Optimal Mixing with the single-objective best-found solution as the donor. */
        order = createRandomOrdering(number_of_linkage_sets);
        for( i = 0; i < number_of_linkage_sets; i++ )
        {
            linkage_group_index = order[i];
            copyValuesFromDonorToOffspring(result, elitist_copy, cluster_index, linkage_group_index);
            mutateSolution(result, linkage_group_index, cluster_index);

            /* Check if the new intermediate solution is different from the previous state. */
            is_unchanged = TRUE;
            for( j = 0; j < lt_number_of_indices[cluster_index][linkage_group_index]; j++ )
            {
                if( backup[lt[cluster_index][linkage_group_index][j]] != result[lt[cluster_index][linkage_group_index][j]] )
                {
                    is_unchanged = FALSE;
                    break;
                }
            }           

            if( is_unchanged == FALSE )
            {
                is_improved = FALSE;
                evaluateIndividual(result, obj, con, objective_index);
                updateElitistArchiveWithReplacementOfExistedMember(result, obj, *con, &is_new_nondominated_point, &is_dominated_by_archive);

                /* Check if strict improvement in the corresponding objective. */
                if(betterFitness(obj, *con, obj_backup, con_backup, objective_index) )
                    is_improved = TRUE;

                if (is_improved == TRUE)
                {
                    changed = TRUE;
                    copyFromAToB(result, obj, *con, backup, obj_backup, &con_backup );
                    break;
                }
                else
                    copyFromAToB(backup, obj_backup, con_backup, result, obj, con);
                if (max_evals_reached == TRUE) {break;}
            }
        }
        free(order);

        if(!changed)
        {
            donor_index = 0;
            for (j= 0; j < elitist_archive_size; j++)
            {
                if(optimization[objective_index] == MINIMIZATION)
                {
                    if(elitist_archive_objective_values[j][objective_index] < elitist_archive_objective_values[donor_index][objective_index])
                        donor_index = j;
                }
                else if(optimization[objective_index] == MAXIMIZATION)
                {
                    if(elitist_archive_objective_values[j][objective_index] > elitist_archive_objective_values[donor_index][objective_index])
                        donor_index = j;   
                }
            }

            copyFromAToB(elitist_archive[donor_index], elitist_archive_objective_values[donor_index], 
                    elitist_archive_constraint_values[donor_index], 
                    result, obj, con);
        }
    }
    if (ENABLE_DEBUG_OUTPUT == TRUE) printf("End forced improvements\n");
    
    free( backup ); free( obj_backup ); free( elitist_copy );
    if (ENABLE_DEBUG_OUTPUT == TRUE) printf("Single-objective GOM end!\n");
}
/**
 * Determines the solutions that finally survive the generation (offspring only).
 */
void selectFinalSurvivors()
{
    int i, j;

    for( i = 0; i < population_size; i++ )
    {
        for( j = 0; j < number_of_parameters; j++ )
            population[i][j] = offspring[i][j];
        for( j = 0; j < number_of_objectives; j++)
            objective_values[i][j]  = objective_values_offspring[i][j];
        constraint_values[i] = constraint_values_offspring[i];
    }
}

void freeAuxiliaryPopulations()
{
    int i, k;

    if(population_indices_of_cluster_members != NULL)
    {
        for(k = 0; k < number_of_mixing_components; k++)
            free(population_indices_of_cluster_members[k]);
        free(population_indices_of_cluster_members);
        population_indices_of_cluster_members = NULL;
        free(population_cluster_sizes);
    }

    if(offspring != NULL)
    {
        for(i = 0; i < offspring_size; i++)
        {
            free(offspring[i]);
            free(objective_values_offspring[i]);
        }
        free(offspring);
        free(objective_values_offspring);
        free(constraint_values_offspring);
        offspring = NULL;
    }
    
    ezilaitiniClusters();
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Parameter-free Mechanism -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
void initializeMemoryForArrayOfPopulations()
{
    int i;
    maximum_number_of_populations      = 20;

    array_of_populations                = (int***)Malloc(maximum_number_of_populations*sizeof(int**));
    array_of_objective_values           = (double***)Malloc(maximum_number_of_populations*sizeof(double**));
    array_of_constraint_values          = (double**)Malloc(maximum_number_of_populations*sizeof(double*));
    array_of_objective_ranges           = (double**)Malloc(maximum_number_of_populations*sizeof(double));

    array_of_t_NIS                      = (int*)Malloc(maximum_number_of_populations*sizeof(int));
    array_of_number_of_generations                = (int*)Malloc(maximum_number_of_populations*sizeof(int));
    for(i = 0; i < maximum_number_of_populations; i++)
    {
        array_of_number_of_generations[i]         = 0;
        array_of_t_NIS[i]               = 0;
    }

    array_of_number_of_evaluations_per_population = (long*)Malloc(maximum_number_of_populations*sizeof(long));
    for(i = 0; i < maximum_number_of_populations; i++)
        array_of_number_of_evaluations_per_population[i] = 0;

    /* Popupulation-sizing free scheme. */
    array_of_population_sizes           = (int*)Malloc(maximum_number_of_populations*sizeof(int));
    array_of_population_sizes[0]        = smallest_population_size;
    for(i = 1; i < maximum_number_of_populations; i++)
        array_of_population_sizes[i]    = array_of_population_sizes[i-1]*2;

    /* Number-of-clusters parameter-free scheme. */
    array_of_number_of_clusters         = (int*)Malloc(maximum_number_of_populations*sizeof(int));
    array_of_number_of_clusters[0]      = number_of_objectives + 1;
    for(i = 1; i < maximum_number_of_populations; i++)
        array_of_number_of_clusters[i]  = array_of_number_of_clusters[i-1] + 1;
}

void putInitializedPopulationIntoArray()
{
    array_of_objective_ranges[population_id]    = objective_ranges;
    array_of_populations[population_id]         = population;
    array_of_objective_values[population_id]    = objective_values;
    array_of_constraint_values[population_id]   = constraint_values;
    array_of_t_NIS[population_id]               = 0;
}

void assignPointersToCorrespondingPopulation()
{
    population                  = array_of_populations[population_id];
    objective_values            = array_of_objective_values[population_id];
    constraint_values           = array_of_constraint_values[population_id];
    population_size             = array_of_population_sizes[population_id];
    objective_ranges            = array_of_objective_ranges[population_id];
    t_NIS                       = array_of_t_NIS[population_id];
    number_of_generations       = array_of_number_of_generations[population_id];
    number_of_mixing_components = array_of_number_of_clusters[population_id];
}

void ezilaitiniMemoryOfCorrespondingPopulation()
{
    int i;

    for( i = 0; i < population_size; i++ )
    {
        free( population[i] );
        free( objective_values[i] );
    }
    free( population );
    free( objective_values );
    free( constraint_values );
    free( objective_ranges );
}

void ezilaitiniArrayOfPopulation()
{
    int i;
    for(i = 0; i < number_of_populations; i++)
    {
        population_id = i;
        assignPointersToCorrespondingPopulation();
        ezilaitiniMemoryOfCorrespondingPopulation();
    }
    free(array_of_populations);
    free(array_of_objective_values);
    free(array_of_constraint_values);
    free(array_of_population_sizes);
    free(array_of_objective_ranges);
    free(array_of_t_NIS);
    free(array_of_number_of_generations);
    free(array_of_number_of_evaluations_per_population);
    free(array_of_number_of_clusters);
}
/**
 * Schedule the run of multiple populations.
 */
void schedule_runMultiplePop_clusterPop_learnPop_improvePop()
{
    int i;
    smallest_population_size = 8;
    initializeMemoryForArrayOfPopulations();
    initializeArrayOfParetoFronts();
    while( !checkTerminationCondition() )
    {
        population_id = 0;
        do
        {
            if(array_of_number_of_generations[population_id] == 0)
            {
                population_size = array_of_population_sizes[population_id];
                number_of_mixing_components = array_of_number_of_clusters[population_id];

                initialize();

                putInitializedPopulationIntoArray();

                if(stop_population_when_front_is_covered)
                {
                    updateParetoFrontForCurrentPopulation(objective_values, constraint_values, population_size);
                    checkWhichSmallerPopulationsNeedToStop();
                }

                writeGenerationalStatistics();
            }
            else if(array_of_population_statuses[population_id] == TRUE)
            {
                assignPointersToCorrespondingPopulation();

                if (if_call_python_train == TRUE) {
                    if (number_of_evaluations - last_eval_when_trained >= n_evals_between_train) {
                        // train
                        printf("Calling 'train' at eval %d\n", number_of_evaluations);
                        writeCurrentElitistArchive(FALSE);
                        last_eval_when_trained = number_of_evaluations;
                        PyObject *arglist = Py_BuildValue("()");
                        PyObject *result = PyEval_CallObject(trainFunction, arglist);
                        checkPythonError(result, "Calling 'train' function failed!");
                        Py_DECREF(result);
                        Py_DECREF(arglist);
                        // reevaluate
                        // first, reevaluate elitist archive ('cause it'll be easier to update afterwards)
                        for(int i_el = 0; i_el < elitist_archive_size; i_el++ ) {
                            evaluateIndividual( elitist_archive[i_el], elitist_archive_objective_values[i_el],
                                                &(elitist_archive_constraint_values[i_el]), NOT_EXTREME_CLUSTER );
                        }
                        // second, reevaluate current population & all smaller ones (needed for 'stop_population_when_front_is_covered')
                        for (int i_pop = 0; i_pop <= population_id; i_pop++) {
                            for (int i_sol = 0; i_sol < array_of_population_sizes[i_pop]; i_sol++) {
                                evaluateIndividual(array_of_populations[i_pop][i_sol], array_of_objective_values[i_pop][i_sol], &(array_of_constraint_values[i_pop][i_sol]),
                                                   NOT_EXTREME_CLUSTER);
                                updateElitistArchive(array_of_populations[i_pop][i_sol], array_of_objective_values[i_pop][i_sol], array_of_constraint_values[i_pop][i_sol]);
                            }
                        }
                        // third, recompute objective ranges for all populations
                        computeObjectiveRanges();

                        // finally, record that bigger populations will need to be reevaluated
                        for(int i_reeval = 0; i_reeval < maximum_number_of_populations; i_reeval++)
                            array_population_needs_reevaluation[i_reeval] = TRUE;
                    }
                    else {
                        // if some training occured for another populaiton, may need to reevaluate the current one too
                        if (array_population_needs_reevaluation[population_id] == TRUE) {
                            for (int i_sol = 0; i_sol < array_of_population_sizes[population_id]; i_sol++) {
                                evaluateIndividual(array_of_populations[population_id][i_sol], array_of_objective_values[population_id][i_sol], &(array_of_constraint_values[population_id][i_sol]),
                                                   NOT_EXTREME_CLUSTER);
                                updateElitistArchive(array_of_populations[population_id][i_sol], array_of_objective_values[population_id][i_sol], array_of_constraint_values[population_id][i_sol]);
                            }
                            array_population_needs_reevaluation[population_id] = FALSE;
                        }
                    }
                }

                learnLinkageOnCurrentPopulation();

                improveCurrentPopulation();

                selectFinalSurvivors();

                computeObjectiveRanges();

                adaptObjectiveDiscretization();

                array_of_t_NIS[population_id] = t_NIS;

                if(stop_population_when_front_is_covered)
                {
                    updateParetoFrontForCurrentPopulation(objective_values, constraint_values, population_size);
                    checkWhichSmallerPopulationsNeedToStop();
                }

                writeGenerationalStatistics();
            }
            
            array_of_number_of_generations[population_id]++;
            if(use_print_progress_to_screen)
                printf("%d ", array_of_number_of_generations[population_id]);
            population_id++;
            if(checkTerminationCondition() == TRUE)
                break;
        } while(array_of_number_of_generations[population_id-1] % generation_base == 0);
        if(use_print_progress_to_screen)
            printf(":   %d\n", number_of_evaluations);
    }
    
    if(use_print_progress_to_screen)
    {
        printf("Population Status:\n");
        for(i=0; i < number_of_populations; i++)
            printf("Pop %d: %d\n", ((int)(pow(2,i)))*smallest_population_size, array_of_population_statuses[i]);
    }
    logNumberOfEvaluationsAtVTR();
    writeCurrentElitistArchive( TRUE );
    ezilaitiniArrayOfPopulation();
    ezilaitiniArrayOfParetoFronts();
}

void schedule()
{   
    schedule_runMultiplePop_clusterPop_learnPop_improvePop();
}

/*---------------------Section Stop Smaller Populations -----------------------------------*/
void initializeArrayOfParetoFronts()
{
    int i;
    
    array_of_population_statuses                    = (char*)Malloc(maximum_number_of_populations*sizeof(char));
    for(i = 0; i < maximum_number_of_populations; i++)
        array_of_population_statuses[i] = TRUE;
    
    array_of_Pareto_front_size_of_each_population   = (int*)Malloc(maximum_number_of_populations*sizeof(int));
    for(i = 0; i < maximum_number_of_populations; i++)
        array_of_Pareto_front_size_of_each_population[i] = 0;

    array_of_Pareto_front_of_each_population        = (double***)Malloc(maximum_number_of_populations*sizeof(double**));
    for(i = 0; i < maximum_number_of_populations; i++)
        array_of_Pareto_front_of_each_population[i] = NULL;
}

void ezilaitiniArrayOfParetoFronts()
{
    int i, j;

    FILE *file;
    file = fopen("population_status.dat", "w");
    for(i = 0; i < number_of_populations; i++)
    {
        fprintf(file, "Pop %d: %d\n", ((int)(pow(2,i)))*smallest_population_size, array_of_population_statuses[i]);
    }
    fclose(file);
    for(i = 0; i < maximum_number_of_populations; i++)
    {
        if(array_of_Pareto_front_size_of_each_population[i] > 0)
        {
            for(j = 0; j < array_of_Pareto_front_size_of_each_population[i]; j++)
                free(array_of_Pareto_front_of_each_population[i][j]);
            free(array_of_Pareto_front_of_each_population[i]);
        }
    }
    free(array_of_Pareto_front_of_each_population);
    free(array_of_Pareto_front_size_of_each_population);
    free(array_of_population_statuses);
}

char checkParetoFrontCover(int pop_index_1, int pop_index_2)
{
    int i, j, count;
    count = 0;
    
    for(i = 0; i < array_of_Pareto_front_size_of_each_population[pop_index_2]; i++)
    {
        for(j = 0; j < array_of_Pareto_front_size_of_each_population[pop_index_1]; j++)
            if((constraintParetoDominates(array_of_Pareto_front_of_each_population[pop_index_1][j], 0, 
                array_of_Pareto_front_of_each_population[pop_index_2][i], 0) == TRUE) ||
                sameObjectiveBox(array_of_Pareto_front_of_each_population[pop_index_1][j], array_of_Pareto_front_of_each_population[pop_index_2][i]) == TRUE)
        {
            count++;
            break;
        }
    }
    // Check if all points in front 2 are dominated by or exist in front 1
    if(count == array_of_Pareto_front_size_of_each_population[pop_index_2])
        return TRUE;
    return FALSE;
}

void checkWhichSmallerPopulationsNeedToStop()
{
    int i;
    for(i = population_id - 1; i >= 0; i--)
    {
        if(array_of_population_statuses[i] == FALSE)
            continue;
        if(checkParetoFrontCover(population_id, i) == TRUE)
            array_of_population_statuses[i] = FALSE;
    }
}

void updateParetoFrontForCurrentPopulation(double **objective_values_pop, double *constraint_values_pop, int pop_size)
{
    int i, j, index, rank0_size;
    char *isDominated;
    isDominated = (char*)Malloc(pop_size*sizeof(char));
    for(i = 0; i < pop_size; i++)
        isDominated[i] = FALSE;
    for (i = 0; i < pop_size; i++)
    {
        if(isDominated[i] == TRUE)
            continue;
        for(j = i+1; j < pop_size; j++)
        {
            if(isDominated[j] == TRUE)
                continue;
            if(constraintParetoDominates(objective_values_pop[i], constraint_values_pop[i], objective_values_pop[j],constraint_values_pop[j]) == TRUE)
                isDominated[j]=TRUE;
            else if(constraintParetoDominates(objective_values_pop[j], constraint_values_pop[j], objective_values_pop[i],constraint_values_pop[i]) == TRUE)
            {
                isDominated[i]=TRUE;
                break;
            }
        }
    }

    rank0_size = 0;
    for(i = 0; i < pop_size; i++)
        if(isDominated[i]==FALSE)
            rank0_size++;

    if(array_of_Pareto_front_size_of_each_population[population_id] > 0)
    {
        for(i = 0; i < array_of_Pareto_front_size_of_each_population[population_id]; i++)
        {
            free(array_of_Pareto_front_of_each_population[population_id][i]);
        }
        free(array_of_Pareto_front_of_each_population[population_id]);        
    }

    array_of_Pareto_front_of_each_population[population_id] = (double**)Malloc(rank0_size*sizeof(double*));
    for(i = 0; i < rank0_size; i++)
        array_of_Pareto_front_of_each_population[population_id][i] = (double*)Malloc(number_of_objectives*sizeof(double));
    array_of_Pareto_front_size_of_each_population[population_id] = rank0_size;

    index = 0;
    for(i = 0; i < pop_size; i++)
    {
        if(isDominated[i] == TRUE)
            continue;
        for(j = 0; j < number_of_objectives; j++)
            array_of_Pareto_front_of_each_population[population_id][index][j] = objective_values_pop[i][j];
        index++;
    }
    free(isDominated);
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Run -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
void initializeCommonVariables()
{
    int i;

    initializeRandomNumberGenerator();
    generation_base                         = 2;

    number_of_generations                   = 0;
    number_of_evaluations                   = 0;
    objective_discretization_in_effect      = 0;
    elitist_archive_size                    = 0;
    elitist_archive_capacity                = 10;
    elitist_archive                         = (int **) Malloc( elitist_archive_capacity*sizeof( int * ) );
    elitist_archive_objective_values        = (double **) Malloc( elitist_archive_capacity*sizeof( double * ) );
    elitist_archive_constraint_values       = (double *) Malloc( elitist_archive_capacity*sizeof( double ) );
    
    for( i = 0; i < elitist_archive_capacity; i++ )
    {
        elitist_archive[i]                  = (int *) Malloc( number_of_parameters*sizeof( int ) );
        elitist_archive_objective_values[i] = (double *) Malloc( number_of_objectives*sizeof( double ) );
    }
    elitist_archive_copy                    = NULL;
    objective_discretization = (double *) Malloc( number_of_objectives*sizeof( double ) );

    MI_matrix = (double **) Malloc( number_of_parameters*sizeof( double * ) );
    for( i = 0; i < number_of_parameters; i++ )
        MI_matrix[i] = (double *) Malloc( number_of_parameters*sizeof( double ) );

    population_indices_of_cluster_members   = NULL;
    population_cluster_sizes                = NULL;

    offspring = NULL;
    
    number_of_populations = 0;

    lt = NULL;

    if (if_use_initialization_genomes_for_elitist_archive == TRUE) {
        population_id = -1; // need to set this before evaluation
        int j = 0;
        int* solution = (int*) Malloc(number_of_parameters * sizeof(int));
        double* objective_values_cur  = (double *) Malloc( number_of_objectives*sizeof( double ) );
        double constraint_value_cur  = 0;

        while (!feof (file_initialization_genomes)) {
            for( j = 0; j < number_of_parameters; j++ ) {
                fscanf(file_initialization_genomes, "%d", &solution[j]);
            }
            evaluateIndividual( solution, objective_values_cur,  &(constraint_value_cur), NOT_EXTREME_CLUSTER );
            updateElitistArchive( solution, objective_values_cur, constraint_value_cur);
        }
        fclose (file_initialization_genomes);
        writeCurrentElitistArchive( FALSE );
    }
}

void ezilaitiniCommonVariables( void )
{
    int      i, j;
    
    if( elitist_archive_copy != NULL )
    {
        for( i = 0; i < elitist_archive_copy_size; i++ )
        {
            free( elitist_archive_copy[i] );
            free( elitist_archive_copy_objective_values[i] );
        }
        free( elitist_archive_copy );
        free( elitist_archive_copy_objective_values );
        free( elitist_archive_copy_constraint_values );
    }

    for( i = 0; i < elitist_archive_capacity; i++ )
    {
        free( elitist_archive[i] );
        free( elitist_archive_objective_values[i] );        
    }
    free( elitist_archive );
    free( elitist_archive_objective_values );
    free( elitist_archive_constraint_values );
    free( objective_discretization );
    
    for( i = 0; i < number_of_parameters; i++ )
        free( MI_matrix[i] );

    free( MI_matrix );
}

void loadProblemData()
{
    switch(problem_index)
    {
        case ZEROMAX_ONEMAX: onemaxLoadProblemData(); break;
        case TRAP5: trap5LoadProblemData(); break;
        case LOTZ: lotzLoadProblemData(); break;
        case KNAPSACK: knapsackLoadProblemData(); break;
        case MAXCUT: maxcutLoadProblemData(); break;
        case PYTHON_FUNCTION: pythonFunctionLoadProblemData(); break;
        default: 
            printf("Cannot load problem data!\n");
            exit(1);
    }
}

void ezilaitiniProblemData()
{
    double **default_front;
    int i, default_front_size;

    switch(problem_index)
    {
        case KNAPSACK: ezilaitiniKnapsackProblemData(); break;
        case MAXCUT: ezilaitiniMaxcutProblemData(); break;
    }

    free(optimization);
    
    default_front = getDefaultFront( &default_front_size );
    if( default_front )
    {
        for( i = 0; i < default_front_size; i++ )
            free( default_front[i] );
        free( default_front );
    }
}

void run( void )
{
    loadProblemData();

    initializeCommonVariables();

    schedule();

    ezilaitiniProblemData();

    ezilaitiniCommonVariables();
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Main -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/
/**
 * The main function:
 * - interpret parameters on the command line
 * - run the algorithm with the interpreted parameters
 */
int main( int argc, char **argv )
{
    Py_Initialize();
    interpretCommandLine( argc, argv );
    run();

    Py_Finalize();
    return( 0 );
}
