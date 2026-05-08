#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>
using namespace std;

class Graph {
    int V;
    vector<vector<int>> adj;

public:
    Graph(int v) {
        V = v;
        adj.resize(V);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // -------- Sequential BFS --------
    void bfs_seq(int start) {
        vector<bool> visited(V, false);
        queue<int> q;

        visited[start] = true;
        q.push(start);

        while (!q.empty()) {
            int node = q.front(); q.pop();
            cout << node << " ";

            for (int nb : adj[node]) {
                if (!visited[nb]) {
                    visited[nb] = true;
                    q.push(nb);
                }
            }
        }
    }

    // -------- Sequential DFS --------
    void dfs_seq(int start) {
        vector<bool> visited(V, false);
        stack<int> st;

        st.push(start);
        visited[start] = true;

        while (!st.empty()) {
            int node = st.top(); st.pop();
            cout << node << " ";

            for (auto it = adj[node].rbegin(); it != adj[node].rend(); it++) {
                if (!visited[*it]) {
                    visited[*it] = true;
                    st.push(*it);
                }
            }
        }
    }

    // -------- Parallel BFS --------
    void bfs_par(int start) {
        vector<bool> visited(V, false);
        queue<int> q;

        visited[start] = true;
        q.push(start);

        while (!q.empty()) {
            int size = q.size();
            vector<int> level;

            for (int i = 0; i < size; i++) {
                int node = q.front(); q.pop();
                cout << node << " ";
                level.push_back(node);
            }

#pragma omp parallel for
            for (int i = 0; i < level.size(); i++) {
                int node = level[i];
                for (int nb : adj[node]) {
#pragma omp critical
                    {
                        if (!visited[nb]) {
                            visited[nb] = true;
                            q.push(nb);
                        }
                    }
                }
            }
        }
    }

    // -------- Parallel DFS (task-based) --------
    void dfs_task(int node, vector<bool>& visited) {
    #pragma omp critical
        {
            if (visited[node]) return;
            visited[node] = true;
            cout << node << " ";
        }

        for (int nb : adj[node]) {
            if (!visited[nb]) {
    #pragma omp task
                dfs_task(nb, visited);
            }
        }

    #pragma omp taskwait
    }

    void dfs_par(int start) {
        vector<bool> visited(V, false);

#pragma omp parallel
        {
#pragma omp single
            dfs_task(start, visited);
        }
    }
};

int main() {
    cout << "Program started\n";
    Graph g(6);

    g.addEdge(0,1);
    g.addEdge(0,2);
    g.addEdge(1,3);
    g.addEdge(1,4);
    g.addEdge(2,4);
    g.addEdge(3,5);
    g.addEdge(4,5);

    double t1, t2;

    cout << "Sequential BFS: ";
    t1 = omp_get_wtime();
    g.bfs_seq(0);
    t2 = omp_get_wtime();
    cout << "\nTime: " << (t2 - t1)*1000 << " ms\n\n";

    cout << "Parallel BFS: ";
    t1 = omp_get_wtime();
    g.bfs_par(0);
    t2 = omp_get_wtime();
    cout << "\nTime: " << (t2 - t1)*1000 << " ms\n\n";

    cout << "Sequential DFS: ";
    t1 = omp_get_wtime();
    g.dfs_seq(0);
    t2 = omp_get_wtime();
    cout << "\nTime: " << (t2 - t1)*1000 << " ms\n\n";

    cout << "Parallel DFS: ";
    t1 = omp_get_wtime();
    g.dfs_par(0);
    t2 = omp_get_wtime();
    cout << "\nTime: " << (t2 - t1)*1000 << " ms\n";

    return 0;
}


// g++ -fopenmp bfsDfs.cpp -o output
// ./output
