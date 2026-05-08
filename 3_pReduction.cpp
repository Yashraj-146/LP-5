#include<iostream>
#include<climits>
#include<algorithm>
#include<vector>
#include<ctime>
#include <omp.h>

using namespace std;

/* ---------- SEQUENTIAL ---------- */
int seq_min(int arr[], int n) {
    int min_val = INT_MAX;

    for(int i = 0; i < n; i++)
        if(arr[i] < min_val)
            min_val = arr[i];

    return min_val;
}

int seq_max(int arr[], int n) {
    int max_val = INT_MIN;

    for(int i = 0; i < n; i++)
        if(arr[i] > max_val)
            max_val = arr[i];

    return max_val;
}

int seq_sum(int arr[], int n) {
    int sum = 0;

    for(int i = 0; i < n; i++)
        sum += arr[i];

    return sum;
}

double seq_avg(int arr[], int n) {
    int sum = 0;

    for(int i = 0; i < n; i++)
        sum += arr[i];

    return (double)sum / n;
}

/* ---------- PARALLEL ---------- */

int par_min(int arr[], int n) {
    int min_val = INT_MAX;

    #pragma omp parallel for reduction(min:min_val)
    for(int i = 0; i < n; i++)
        if(arr[i] < min_val)
            min_val = arr[i];

    return min_val;
}

int par_max(int arr[], int n) {
    int max_val = INT_MIN;

    #pragma omp parallel for reduction(max:max_val)
    for(int i = 0; i < n; i++)
        if(arr[i] > max_val)
            max_val = arr[i];

    return max_val;
}

int par_sum(int arr[], int n) {
    int sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for(int i = 0; i < n; i++)
        sum += arr[i];

    return sum;
}

double par_avg(int arr[], int n) {
    int sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for(int i = 0; i < n; i++)
        sum += arr[i];

    return (double)sum / n;
}

/* ---------- MAIN ---------- */
int main() {
    int n;

    cout << "Enter size: ";
    cin >> n;

    int *arr = new int[n];

    for(int i = 0; i < n; i++)
        arr[i] = rand() % 1000;

    double t1, t2;

    /* ---------- Sequential ---------- */
    t1 = omp_get_wtime();
    int smin = seq_min(arr, n);
    t2 = omp_get_wtime();
    cout << "Sequential Min: " << smin << " | Time: " << t2 - t1 << "s\n";

    t1 = omp_get_wtime();
    int smax = seq_max(arr, n);
    t2 = omp_get_wtime();
    cout << "Sequential Max: " << smax << " | Time: " << t2 - t1 << "s\n";

    t1 = omp_get_wtime();
    int ssum = seq_sum(arr, n);
    t2 = omp_get_wtime();
    cout << "Sequential Sum: " << ssum << " | Time: " << t2 - t1 << "s\n";

    t1 = omp_get_wtime();
    double savg = seq_avg(arr, n);
    t2 = omp_get_wtime();
    cout << "Sequential Avg: " << savg << " | Time: " << t2 - t1 << "s\n";


    /* ---------- Parallel ---------- */
    t1 = omp_get_wtime();
    int pmin = par_min(arr, n);
    t2 = omp_get_wtime();
    cout << "Parallel Min: " << pmin << " | Time: " << t2 - t1 << "s\n";

    t1 = omp_get_wtime();
    int pmax = par_max(arr, n);
    t2 = omp_get_wtime();
    cout << "Parallel Max: " << pmax << " | Time: " << t2 - t1 << "s\n";

    t1 = omp_get_wtime();
    int psum = par_sum(arr, n);
    t2 = omp_get_wtime();
    cout << "Parallel Sum: " << psum << " | Time: " << t2 - t1 << "s\n";

    t1 = omp_get_wtime();
    double pavg = par_avg(arr, n);
    t2 = omp_get_wtime();
    cout << "Parallel Avg: " << pavg << " | Time: " << t2 - t1 << "s\n";

    delete[] arr;

    return 0;
}

// Compile: g++ -fopenmp -o parallelreduction parallel_reduction.cpp
// Run: ./parallelreduction
