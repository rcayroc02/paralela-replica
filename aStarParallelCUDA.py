import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from collections import defaultdict

import numpy as np
import networkx as nx
import urllib.request
import gzip
import os
from tqdm import tqdm
import time
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import seaborn as sns

def run_enhanced_benchmarks(astar_class, num_gpus=1, num_trials=10):
    """
    Run comprehensive benchmarks with multiple trials per configuration
    
    Args:
        astar_class: The A* implementation class
        num_gpus: Number of GPUs to use
        num_trials: Number of trials per configuration
    
    Returns:
        results: List of benchmark results
        summary: Dictionary containing average metrics
    """
    results = []
    sizes = [1000, 5000, 10000]
    
    for size in sizes:
        print(f"\nTesting graphs of size {size}:")
        
        # Generate different graph types
        print("Generating graphs...")
        edges_random = GraphGenerator.generate_random_directed_graph(
            size, edge_probability=0.01)
        edges_scale_free = GraphGenerator.generate_scale_free_directed_graph(size)
        width = int(np.sqrt(size))
        edges_grid = GraphGenerator.generate_grid_like_directed_graph(width, width)
        
        graph_configs = [
            ('Random', edges_random),
            ('Scale-free', edges_scale_free),
            ('Grid-like', edges_grid)
        ]
        
        # Test each graph type
        for graph_type, edges in graph_configs:
            print(f"\nTesting {graph_type} graph with {len(edges)} edges:")
            astar = astar_class(size, edges, num_gpus)
            
            # Run multiple trials
            trial_times = []
            trial_paths = []
            successful_paths = 0
            
            for trial in tqdm(range(num_trials), desc=f"Running {num_trials} trials"):
                # Generate random start and end nodes
                start = np.random.randint(0, size)
                end = np.random.randint(0, size)
                while end == start:  # Ensure different start and end
                    end = np.random.randint(0, size)
                
                start_time = time.time()
                path = astar.find_path(start, end)
                execution_time = time.time() - start_time
                
                trial_times.append(execution_time)
                if path is not None:
                    trial_paths.append(len(path))
                    successful_paths += 1
                
                # Store detailed results for each trial
                results.append({
                    'graph_type': graph_type,
                    'size': size,
                    'num_edges': len(edges),
                    'trial': trial,
                    'start_node': start,
                    'end_node': end,
                    'execution_time': execution_time,
                    'path_found': path is not None,
                    'path_length': len(path) if path else 0
                })
            
            # Calculate statistics
            avg_time = np.mean(trial_times)
            std_time = np.std(trial_times)
            success_rate = (successful_paths / num_trials) * 100
            avg_path_length = np.mean(trial_paths) if trial_paths else 0
            
            print(f"Results for {graph_type} graph (size {size}):")
            print(f"Average time: {avg_time:.4f}s (±{std_time:.4f}s)")
            print(f"Success rate: {success_rate:.1f}%")
            print(f"Average path length: {avg_path_length:.1f}")
    
    return results

def save_and_visualize_results(results, output_dir='benchmark_results'):
    """
    Save results to file and create visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert results to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Save raw results to JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(output_dir, f'benchmark_results_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualizations
    plt.style.use('seaborn-v0_8')
    
    # 1. Execution Time Distribution by Graph Type and Size
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Distribucion del tiempo de ejecucion de acuerdo al grafo y al tamano')
    
    for i, size in enumerate(df['size'].unique()):
        size_data = df[df['size'] == size]
        
        for graph_type in df['graph_type'].unique():
            times = size_data[size_data['graph_type'] == graph_type]['execution_time']
            axes[i].hist(times, alpha=0.5, label=graph_type, bins=20)
            
        axes[i].set_title(f'Tamano: {size} vertices')
        axes[i].set_xlabel('Tiempo de ejecucion (s)')
        axes[i].set_ylabel('Frecuencia')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'time_distribution_{timestamp}.png'))
    
    # 2. Success Rate Analysis
    plt.figure(figsize=(12, 6))
    success_rates = df.groupby(['graph_type', 'size'])['path_found'].mean() * 100
    success_rates.unstack().plot(kind='bar')
    plt.title('Eficacia de encontrar el camino mas corto de acuerdo al grafo y al tamano')
    plt.xlabel('Tipo de Grafo')
    plt.ylabel('Eficacia (%)')
    plt.legend(title='Tamano del grafo')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'success_rates_{timestamp}.png'))
    
    # 3. Average Path Length Analysis
    plt.figure(figsize=(12, 6))
    path_lengths = df[df['path_found']].groupby(['graph_type', 'size'])['path_length'].mean()
    path_lengths.unstack().plot(kind='bar')
    plt.title('Tamano del camino de acuerdo al grafo y al tamano')
    plt.xlabel('Tipo de Grafo')
    plt.ylabel('Tamano del camino en promedio')
    plt.legend(title='Tamano del grafo')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'path_lengths_{timestamp}.png'))
    
    # 4. Performance Summary Table
    summary = df.groupby(['graph_type', 'size']).agg({
        'execution_time': ['mean', 'std'],
        'path_found': 'mean',
        'path_length': 'mean'
    }).round(4)
    
    summary.to_csv(os.path.join(output_dir, f'summary_{timestamp}.csv'))
    
    print(f"\nResults saved to {output_dir}/")
    return summary


class BenchmarkLoader:
    @staticmethod
    def load_snap_dataset(dataset_name):
        """
        Load a graph dataset from Stanford SNAP

        Args:
            dataset_name: Name of the dataset (e.g., 'roadNet-CA', 'web-Google')

        Returns:
            edges: List of tuples representing directed edges
            num_vertices: Number of vertices in the graph
        """
        # Dictionary of SNAP datasets and their URLs
        datasets = {
            'web-Google': 'https://snap.stanford.edu/data/web-Google.txt.gz',
            'roadNet-CA': 'https://snap.stanford.edu/data/roadNet-CA.txt.gz',
            'wiki-Vote': 'https://snap.stanford.edu/data/wiki-Vote.txt.gz'
        }

        if dataset_name not in datasets:
            raise ValueError(f"Dataset {dataset_name} not found. Available datasets: {list(datasets.keys())}")

        url = datasets[dataset_name]
        filename = f"{dataset_name}.txt.gz"

        # Download if not already present
        if not os.path.exists(filename):
            print(f"Downloading {dataset_name}...")
            urllib.request.urlretrieve(url, filename)

        # Load edges
        edges = []
        max_vertex = 0

        # Open gzipped file
        with gzip.open(filename, 'rt') as f:
            for line in tqdm(f, desc="Loading edges"):
                if line.startswith('#'):
                    continue
                values = line.strip().split()
                if len(values) >= 2:
                    src = int(values[0])
                    dst = int(values[1])
                    edges.append((src, dst))
                    max_vertex = max(max_vertex, src, dst)

        return edges, max_vertex + 1

class GraphGenerator:
    @staticmethod
    def generate_random_directed_graph(num_vertices, edge_probability=0.1, seed=None):
        """
        Generate a random directed graph using Erdos-Renyi model
        """
        if seed is not None:
            np.random.seed(seed)

        edges = []
        for i in range(num_vertices):
            for j in range(num_vertices):
                if i != j and np.random.random() < edge_probability:
                    edges.append((i, j))
        return edges

    @staticmethod
    def generate_scale_free_directed_graph(num_vertices, m=2, seed=None):
        """
        Generate a scale-free directed graph using Barabási-Albert model
        """
        if seed is not None:
            np.random.seed(seed)

        G = nx.barabasi_albert_graph(n=num_vertices, m=m, seed=seed)
        edges = []
        for u, v in G.edges():
            if np.random.random() < 0.5:
                edges.append((u, v))
            else:
                edges.append((v, u))
        return edges

    @staticmethod
    def generate_grid_like_directed_graph(width, height, seed=None):
        """
        Generate a grid-like directed graph with some random long-range connections
        """
        if seed is not None:
            np.random.seed(seed)

        edges = []
        # Add grid edges
        for i in range(height):
            for j in range(width):
                current = i * width + j
                # Add right edge
                if j < width - 1:
                    edges.append((current, current + 1))
                # Add down edge
                if i < height - 1:
                    edges.append((current, current + width))

        # Add some random long-range connections
        num_extra_edges = (width * height) // 10
        for _ in range(num_extra_edges):
            start = np.random.randint(0, width * height)
            end = np.random.randint(0, width * height)
            if start != end and (start, end) not in edges:
                edges.append((start, end))

        return edges

def run_benchmarks(astar_class, num_gpus=1):
    """
    Run comprehensive benchmarks with different graph types and sizes
    """
    results = []

    # Test synthetic graphs
    print("\nTesting synthetic graphs:")
    #sizes = [20, 100, 1000, 5000, 10000]
    sizes = [20, 100]

    for size in sizes:
        print(f"\nTesting graphs of size {size}:")

        # Random graph
        print("Generating random graph...")
        edges_random = GraphGenerator.generate_random_directed_graph(
            size, edge_probability=0.01)
        astar = astar_class(size, edges_random, num_gpus)

        start_time = time.time()
        path = astar.find_path(0, size-1)
        execution_time = time.time() - start_time

        results.append({
            'graph_type': 'Random',
            'size': size,
            'num_edges': len(edges_random),
            'execution_time': execution_time,
            'path_found': path is not None,
            'path_length': len(path) if path else 0
        })
        print(f"Random graph results: Time={execution_time:.2f}s, Path found={path is not None}")

        # Scale-free graph
        print("Generating scale-free graph...")
        edges_scale_free = GraphGenerator.generate_scale_free_directed_graph(size)
        astar = astar_class(size, edges_scale_free, num_gpus)

        start_time = time.time()
        path = astar.find_path(0, size-1)
        execution_time = time.time() - start_time

        results.append({
            'graph_type': 'Scale-free',
            'size': size,
            'num_edges': len(edges_scale_free),
            'execution_time': execution_time,
            'path_found': path is not None,
            'path_length': len(path) if path else 0
        })
        print(f"Scale-free graph results: Time={execution_time:.2f}s, Path found={path is not None}")

        # Grid-like graph
        print("Generating grid-like graph...")
        width = int(np.sqrt(size))
        edges_grid = GraphGenerator.generate_grid_like_directed_graph(width, width)
        astar = astar_class(size, edges_grid, num_gpus)

        start_time = time.time()
        path = astar.find_path(0, size-1)
        execution_time = time.time() - start_time

        results.append({
            'graph_type': 'Grid-like',
            'size': size,
            'num_edges': len(edges_grid),
            'execution_time': execution_time,
            'path_found': path is not None,
            'path_length': len(path) if path else 0
        })
        print(f"Grid-like graph results: Time={execution_time:.2f}s, Path found={path is not None}")

    # Test SNAP dataset
    print("\nTesting SNAP dataset:")
    try:
        edges, num_vertices = BenchmarkLoader.load_snap_dataset('wiki-Vote')
        print(f"Loaded wiki-Vote dataset with {num_vertices} vertices and {len(edges)} edges")

        astar = astar_class(num_vertices, edges, num_gpus)

        start_time = time.time()
        path = astar.find_path(0, num_vertices-1)
        execution_time = time.time() - start_time

        results.append({
            'graph_type': 'SNAP wiki-Vote',
            'size': num_vertices,
            'num_edges': len(edges),
            'execution_time': execution_time,
            'path_found': path is not None,
            'path_length': len(path) if path else 0
        })
        print(f"SNAP dataset results: Time={execution_time:.2f}s, Path found={path is not None}")

    except Exception as e:
        print(f"Error processing SNAP dataset: {e}")

    return results

class DirectedGraphAStarCUDA:
    def __init__(self, num_vertices, edges, num_gpus):
        self.num_vertices = num_vertices
        self.edges = edges
        self.num_gpus = min(num_gpus, cuda.Device.count())

        # Initialize partitioner
        self.partitioner = DirectedGraphPartitioner(num_vertices, edges, self.num_gpus)
        self.partitions, self.update_set = self.partitioner.partition()

        # Compile kernel once
        self.mod = SourceModule("""
        __device__ float heuristic(int current, int goal) {
            return abs(goal - current);
        }

        __global__ void astar_kernel(
            const int* edges,
            const int num_edges,
            float* g_score,
            bool* open_set,
            bool* closed_set,
            int* came_from,
            const int start,
            const int goal,
            bool* found
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= num_edges) return;

            int src = edges[idx * 2];
            int dst = edges[idx * 2 + 1];

            if (!open_set[src] || closed_set[src]) return;

            if (src == goal) {
                *found = true;
                return;
            }

            float tentative_g = g_score[src] + 1.0f;

            if (tentative_g < g_score[dst]) {
                g_score[dst] = tentative_g;
                came_from[dst] = src;
                open_set[dst] = true;
            }

            open_set[src] = false;
            closed_set[src] = true;
        }
        """)

        self.kernel = self.mod.get_function("astar_kernel")

    def find_path(self, start, goal):
        # Initialize arrays on CPU
        g_score = np.full(self.num_vertices, np.inf, dtype=np.float32)
        g_score[start] = 0

        open_set = np.zeros(self.num_vertices, dtype=np.bool_)
        open_set[start] = True

        closed_set = np.zeros(self.num_vertices, dtype=np.bool_)
        came_from = np.full(self.num_vertices, -1, dtype=np.int32)

        # Process each partition
        for partition in self.partitions:
            edges = np.array(partition['edges'], dtype=np.int32).flatten()
            if len(edges) == 0:
                continue

            # Allocate GPU memory
            edges_gpu = cuda.mem_alloc(edges.nbytes)
            g_score_gpu = cuda.mem_alloc(g_score.nbytes)
            open_set_gpu = cuda.mem_alloc(open_set.nbytes)
            closed_set_gpu = cuda.mem_alloc(closed_set.nbytes)
            came_from_gpu = cuda.mem_alloc(came_from.nbytes)
            found_gpu = cuda.mem_alloc(np.array([False], dtype=np.bool_).nbytes)

            try:
                # Copy data to GPU
                cuda.memcpy_htod(edges_gpu, edges)
                cuda.memcpy_htod(g_score_gpu, g_score)
                cuda.memcpy_htod(open_set_gpu, open_set)
                cuda.memcpy_htod(closed_set_gpu, closed_set)
                cuda.memcpy_htod(came_from_gpu, came_from)
                cuda.memcpy_htod(found_gpu, np.array([False], dtype=np.bool_))

                # Configure grid and block dimensions
                block_size = 256
                grid_size = (len(partition['edges']) + block_size - 1) // block_size

                # Execute kernel
                self.kernel(
                    edges_gpu,
                    np.int32(len(partition['edges'])),
                    g_score_gpu,
                    open_set_gpu,
                    closed_set_gpu,
                    came_from_gpu,
                    np.int32(start),
                    np.int32(goal),
                    found_gpu,
                    block=(block_size, 1, 1),
                    grid=(grid_size, 1)
                )

                # Copy results back to CPU
                cuda.memcpy_dtoh(g_score, g_score_gpu)
                cuda.memcpy_dtoh(open_set, open_set_gpu)
                cuda.memcpy_dtoh(closed_set, closed_set_gpu)
                cuda.memcpy_dtoh(came_from, came_from_gpu)

                found = np.array([False], dtype=np.bool_)
                cuda.memcpy_dtoh(found, found_gpu)

                if found[0]:
                    break

            finally:
                # Free GPU memory
                edges_gpu.free()
                g_score_gpu.free()
                open_set_gpu.free()
                closed_set_gpu.free()
                came_from_gpu.free()
                found_gpu.free()

        # Reconstruct path
        if came_from[goal] == -1:
            return None

        path = []
        current = goal
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        return path[::-1]

class DirectedGraphPartitioner:
    def __init__(self, num_vertices, edges, num_gpus):
        self.num_vertices = num_vertices
        self.edges = edges
        self.num_gpus = num_gpus
        self.vertices_per_partition = num_vertices // num_gpus
        self.partitions = []
        self.update_set = set()

    def partition(self):
        # Create initial partitions
        for i in range(self.num_gpus):
            start_vertex = i * self.vertices_per_partition
            end_vertex = start_vertex + self.vertices_per_partition if i < self.num_gpus - 1 else self.num_vertices

            partition = {
                'vertex_range': (start_vertex, end_vertex),
                'vertices': set(range(start_vertex, end_vertex)),
                'edges': [],
                'boundary_vertices': set()
            }
            self.partitions.append(partition)

        # Assign edges to partitions
        for edge in self.edges:
            src, dst = edge
            src_partition = min(src // self.vertices_per_partition, self.num_gpus - 1)
            dst_partition = min(dst // self.vertices_per_partition, self.num_gpus - 1)

            self.partitions[src_partition]['edges'].append(edge)

            if src_partition != dst_partition:
                self.partitions[src_partition]['boundary_vertices'].add(src)
                self.partitions[dst_partition]['boundary_vertices'].add(dst)
                self.update_set.add(src)
                self.update_set.add(dst)
        # print number of vertices in partition
        print(f"Vertices per partition: {self.vertices_per_partition}")
        # print number of edges
        print(f"Total edges: {len(self.edges)}")

        return self.partitions, self.update_set


        
if __name__ == "__main__":
    
    print("Running enhanced benchmarks...")
    results = run_enhanced_benchmarks(DirectedGraphAStarCUDA, num_gpus=1, num_trials=10)
    
    print("\nGenerating visualizations and saving results...")
    summary = save_and_visualize_results(results)
    
    print("\nBenchmark Summary:")
    print(summary)