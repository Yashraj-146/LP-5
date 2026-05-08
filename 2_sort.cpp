#include<iostream>
#include<vector>
#include<algorithm>
#include<ctime>
#include<omp.h>

using namespace std;

class Sorting{
public:
    void generateArray(vector<int> &arr, int SIZE) {
        arr.resize(SIZE);
        for(int i = 0; i < SIZE; i++) {
            arr[i] = rand() % 10000;
        }
    }
    
    void bubbleSortSequential(vector<int> &arr) {
        int n = arr.size();
        for(int i = 0; i < n-1; i++) {
            for(int j = 0; j < n - i - 1; j++) {
                if(arr[j] > arr[j+1]) {
                    swap(arr[j], arr[j+1]);
                }
            }
        }
    }
    
    void bubbleSortParallel(vector<int> &arr) {
        int n = arr.size();
        for(int i = 0; i < n; i++) {
            #pragma omp parallel for
            for(int j = 0; j < n-1; j+=2) {
                if(arr[j] > arr[j+1])
                    swap(arr[j], arr[j+1]);
            }
        }
    }
    
    void merge(vector<int>& arr, int left, int mid, int right) {
        vector<int> temp(right - left + 1);
        int i = left, j = mid+1, k = 0;
    
        while(i <= mid && j <= right) {
            if(arr[i] < arr[j]) {
                temp[k] = arr[i];
                i++;
            } else {
                temp[k] = arr[j];
                j++;
            }
            k++;
        }
        while(i <= mid) temp[k++] = arr[i++];
        while(j <= right) temp[k++] = arr[j++];
    
        for(int i = 0; i < k; i++) {
            arr[left + i] = temp[i];
        }
    }
    
    void mergeSortSequential(vector<int> &arr, int left, int right) {
        if(left < right) {
            int mid = (left+right)/2;
            mergeSortSequential(arr, left, mid);
            mergeSortSequential(arr, mid+1, right);
            merge(arr, left, mid, right);
        }
    }
    
    void mergeSortParallel(vector<int> &arr, int left, int right, int depth = 0) {
        if(left < right) {
            int mid = (left+right) / 2;
            if(depth < 4) {
                #pragma omp sections
                {
                    #pragma omp section
                    mergeSortParallel(arr, left, mid, depth+1);
                    #pragma omp section
                    mergeSortParallel(arr, mid+1, right, depth+1);
                }
            } else { 
                mergeSortSequential(arr, left, mid);
                mergeSortSequential(arr, mid+1, right);
            }
            merge(arr, left, mid, right);
        }
    }
};

int main() {
    Sorting s;
    int SIZE = 5000;
    vector<int> original;
    original.resize(SIZE);

    vector<int> a1 = original, a2 = original;
    vector<int> a3 = original, a4 = original;

    double t1, t2;
    t1 = omp_get_wtime();
    s.bubbleSortSequential(a1);
    t2 = omp_get_wtime();
    cout<<"Sequential time for Bubble Sort: "<<t2-t1<<"s\n";

    t1 = omp_get_wtime();
    s.bubbleSortParallel(a2);
    t2 = omp_get_wtime();
    cout<<"Parallel time for Bubble Sort: "<<t2-t1<<"s\n";

    t1 = omp_get_wtime();
    s.mergeSortSequential(a3, 0, SIZE-1);
    t2 = omp_get_wtime();
    cout<<"Sequential time for Merge Sort: "<<t2-t1<<"s\n";

    t1 = omp_get_wtime();
    s.mergeSortParallel(a4, 0, SIZE-1);
    t2 = omp_get_wtime();
    cout<<"Parallel time for Bubble Sort: "<<t2-t1<<"s\n";

    return 0;
}


// g++ -fopenmp sort.cpp -o sort
// ./sort
