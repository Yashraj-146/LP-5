#include<iostream>
#include<vector>
#include<math.h>
#include<limits>
#include<algorithm>
#include<omp.h>

using namespace std;

template <typename E>
class ParallelReduction {
    vector<E> arr;

public:
    ParallelReduction(const vector<E>& inputArr) : arr(inputArr) {}

    E sum() {
        E sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for(size_t i = 0; i < arr.size(); i++) {
            sum += arr[i];
        }
        return sum;
    }

    E min() {
        // Initialise minElement to the max possible value of its data type
        E minElement = numeric_limits<E>::max();
        #pragma omp parallel for reduction(min: minElement)
        for(size_t i = 0; i < arr.size(); i++) {
            minElement = std::min(minElement, arr[i]);
        }
        return minElement;
    }

    E max() {
        // Initialise maxElement to the smallest possible value of its data type
        E maxElement = numeric_limits<E>::min();
        #pragma omp parallel for reduction(max: maxElement)
        for(size_t i = 0; i < arr.size(); i++) {
            maxElement = std::max(maxElement, arr[i]);
        }
        return maxElement;
    }

    double mean() {
        E sumValue = sum();
        return static_cast<double>(sumValue) / arr.size(); // Cast sum to double for accurate mean calculation
    }
};

int main() {
    vector<int> arr = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    ParallelReduction<int> pr(arr);

    int sumValue = pr.sum();
    cout << "Sum: " << sumValue << '\n';

    int maxValue = pr.max();
    cout << "Max: " << maxValue << '\n';

    int minValue = pr.min();
    cout << "Min: " << minValue << '\n';

    double meanValue = pr.mean();
    cout << "Mean: " << meanValue << '\n';

    return 0;
}

// Compile: g++ -fopenmp -o parallelreduction parallel_reduction.cpp
// Run: ./parallelreduction
