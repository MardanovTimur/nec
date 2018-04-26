#include <iostream>
#include <stdio.h>
#include <mpi.h>

#ifdef __mpi_ball__ 
using namespace std; 
int main(int argc, char **argv) 
{ 
    int rank, size; 
    MPI_Init(&argc, &argv); 
    MPI_Comm_size(MPI_COMM_WORLD, &size); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    int n = 500000; 
    int loc_n = n / size; 
    int k = 30; 

    vector<int> points; 
    vector<int> global_points; 
    vector<double> result; 
    points.resize(k); 
    global_points.resize(k); 
    result.resize(k); 

    random_device rd; // non-deterministic generator 
    mt19937_64 gen(rd()); // to seed mersenne twister. 
    uniform_real_distribution<> dist(-1, 1); 


    for (int j = 0; j < k; j++) { 
        points[j] = 0; 
    } 

    for (int i = 0; i < loc_n; i++) { 
        double sum = 0; 
        for (int j = 1; j < k; j++) { 
            double x = dist(rd); 
            sum += x*x; 

            if (sum < 1 && j>1) { 
                points[j]++; 
            } 
        } 
    } 


    MPI_Reduce(points.data(), global_points.data(), k, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD); 
    if (rank == 0) { 
        int step = 1; 
        for (int j = 0; j < k; j++) { 
            result[j] = ((double)global_points[j]) / n*step; 
            step *= 2; 
        } 

        for (int j = 2; j < k; j++) { 
            printf("%d-dimensional = %f\n", j, result[j]); 
        } 
    } 

    MPI_Finalize(); 
    return 0; 
} 
#endif
