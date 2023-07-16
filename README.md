# gpu_cuda
Cuda on Google Collab 
- https://www.youtube.com/watch?v=XbZL0C9DvGQ

## Write File in Google Collab 
`%%writefile thread_hierarchy.cu` 

## Compile with nvcc 
`!nvcc thread_hierarchy.cu -o thread_hierarchy`

## Execute 
`!./thread_hierarchy`

# Resources 
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

# ChatGPT Visualizing Thread Hierarchy
## Visualization
- Let's consider a simple example with 1D blocks in a 1D grid first. This is easy to visualize. If we have 2 blocks of 4 threads each, we might have a layout like this:
  ```
  Blocks: B0   B1
  Threads: t0 t1 t2 t3 | t0 t1 t2 t3
  Global Thread ID: 0  1  2  3  | 4  5  6  7 
  ```
  As you can see, each block (B0 and B1) contains 4 threads (t0 - t3). In the context of each block, threads are numbered from 0 to 3 (this is threadIdx.x). But in the global context, each thread has a unique global thread ID ranging from 0 to 7.
-----
- Let's imagine we have a 2D grid containing 2D blocks. Each block contains 2x2 threads and the grid contains 2x2 blocks. In a 2D representation, it could look like this:
  ```
  Blocks:  B00  B01  
           B10  B11
  
  Threads: t00 t01 | t00 t01
           t10 t11 | t10 t11
           ------------------
           t00 t01 | t00 t01
           t10 t11 | t10 t11
  
  Global Thread ID (x,y):
           (0,0) (1,0) | (2,0) (3,0)
           (0,1) (1,1) | (2,1) (3,1)
           ---------------------------
           (0,2) (1,2) | (2,2) (3,2)
           (0,3) (1,3) | (2,3) (3,3)
  ```
The 2D grid contains 4 blocks arranged in 2 rows and 2 columns. Each block (for example, B00, B01, B10, and B11) contains 4 threads arranged in 2 rows and 2 columns. Inside each block, the threads are indexed from (0,0) to (1,1) using (threadIdx.x, threadIdx.y). But in the global context, each thread has a unique 2D global thread ID ranging from (0,0) to (3,3).

## Thread ID 
- **One-dimensional case:**
If you have a block of threads in one dimension, say 10 threads, then the thread ID and the index are the same. So if you're in thread 7, then your thread ID is 7 and your index is also 7. This is straightforward as there's only one dimension to worry about.

- **Two-dimensional case**
If you have a block of threads in two dimensions, things get a bit more interesting. Let's say you have a 3x3 block of threads, so Dx=3, Dy=3. If you're in the thread at position (1,2), then your thread ID is (1 + 2 * 3 = 7). The general formula is thread ID = x + y * Dx.

To visualize this, here's how the thread IDs map to (x, y) coordinates in a 3x3 block:
  ```
  Thread ID | (x, y) coordinates
  ----------|-------------------
      0     | (0, 0)
      1     | (1, 0)
      2     | (2, 0)
      3     | (0, 1)
      4     | (1, 1)
      5     | (2, 1)
      6     | (0, 2)
      7     | (1, 2)
      8     | (2, 2)
  ``` 
