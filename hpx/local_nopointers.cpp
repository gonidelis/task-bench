/* Copyright 2020 Los Alamos National Laboratory
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "hpx/hpx.hpp"
#include <hpx/hpx_main.hpp>
#include <hpx/parallel/algorithms/for_loop.hpp>
#include <hpx/async_local/dataflow.hpp>


#include <stdarg.h>
#include <assert.h>
#include <string.h>
#include <algorithm> 
#include <unistd.h>
#include "../core/core.h"
#include "../core/timer.h"

#define VERBOSE_LEVEL 0

#define USE_CORE_VERIFICATION 1

#define MAX_NUM_ARGS 10

typedef struct tile_s {
  float dep;
  std::string  output_buff;
}tile_t;

typedef struct payload_s {
  int x;
  int y;
  TaskGraph graph;
}payload_t;

typedef struct task_args_s {
  int x;
  int y;
}task_args_t;

typedef struct matrix_s {
  std::vector<tile_t> data;
  int M;
  int N;
}matrix_t;

char **extra_local_memory;

// matrix_t matrix = NULL;
std::vector<matrix_t> matrix;

struct OpenMPApp : public App {
  OpenMPApp(int argc, char **argv);
  ~OpenMPApp();
  void execute_main_loop();
  void execute_timestep(size_t idx, long t);
private:
  void insert_task(task_args_t *args, int num_args, payload_t payload, size_t graph_id);
//   void debug_printf(int verbose_level, const char *format, ...);
private:
  int nb_workers;
//  matrix_t *matrix;
};

OpenMPApp::OpenMPApp(int argc, char **argv)
  : App(argc, argv)
{ 
  nb_workers = 1;
  
  for (int k = 1; k < argc; k++) {
    if (!strcmp(argv[k], "-worker")) {
      nb_workers = atol(argv[++k]);
    }
  }
  
//   matrix = (matrix_t *)malloc(sizeof(matrix_t) * graphs.size());
  
  matrix.resize(graphs.size());
  
  size_t max_scratch_bytes_per_task = 0;
  
  for (unsigned i = 0; i < graphs.size(); i++) {
    TaskGraph &graph = graphs[i];
    
    matrix[i].M = graph.nb_fields;
    matrix[i].N = graph.max_width;
    // matrix[i].data = (tile_t*)malloc(sizeof(tile_t) * matrix[i].M * matrix[i].N);
    matrix[i].data = std::vector<tile_t>(matrix[i].M * matrix[i].N);
  
    for (int j = 0; j < matrix[i].M * matrix[i].N; j++) {
        matrix[i].data[j].output_buff.resize(graph.output_bytes_per_task);
    }
    
    if (graph.scratch_bytes_per_task > max_scratch_bytes_per_task) {
      max_scratch_bytes_per_task = graph.scratch_bytes_per_task;
    }
    
    // printf("graph id %d, M = %d, N = %d, data %p, nb_fields %d\n", i, matrix[i].M, matrix[i].N, matrix[i].data, graph.nb_fields);
  }
  
  extra_local_memory = (char**)malloc(sizeof(char*) * nb_workers);
  assert(extra_local_memory != NULL);
  for (int k = 0; k < nb_workers; k++) {
    if (max_scratch_bytes_per_task > 0) {
      extra_local_memory[k] = (char*)malloc(sizeof(char)*max_scratch_bytes_per_task);
      TaskGraph::prepare_scratch(extra_local_memory[k], sizeof(char)*max_scratch_bytes_per_task);
    } else {
      extra_local_memory[k] = NULL;
    }
  }
  
//   omp_set_num_threads(nb_workers);
  
  if (max_scratch_bytes_per_task > 0) {
    #pragma omp parallel
    {
      int tid = hpx::get_worker_thread_num();
      // printf("im tid %d\n", tid);
      TaskGraph::prepare_scratch(extra_local_memory[tid], sizeof(char)*max_scratch_bytes_per_task);
    }
  }

}

OpenMPApp::~OpenMPApp()
{
//   for (unsigned i = 0; i < graphs.size(); i++) {
//     for (int j = 0; j < matrix[i].M * matrix[i].N; j++) {
//       free(matrix[i].data[j].output_buff);
//       matrix[i].data[j].output_buff = NULL;
//     }
//     free(matrix[i].data);
//     matrix[i].data = NULL;
//   }
  
//   free(matrix);
//   matrix = NULL;
  
  for (int j = 0; j < nb_workers; j++) {
    if (extra_local_memory[j] != NULL) {
      free(extra_local_memory[j]);
      extra_local_memory[j] = NULL;
    }
  }
  free(extra_local_memory);
  extra_local_memory = NULL;
}

void OpenMPApp::execute_main_loop()
{ 
  display();
  
  Timer::time_start();
  
    for (unsigned i = 0; i < graphs.size(); i++) {
       const TaskGraph &g = graphs[i];
        for (int y = 0; y < g.timesteps; y++) {
            execute_timestep(i, y);
        }      
    }
  
  double elapsed = Timer::time_end();
  report_timing(elapsed);
}

void OpenMPApp::execute_timestep(size_t idx, long t)
{
  const TaskGraph &g = graphs[idx];
  long offset = g.offset_at_timestep(t);
  long width = g.width_at_timestep(t);
  long dset = g.dependence_set_at_timestep(t);
  int nb_fields = g.nb_fields;
  
  task_args_t args[MAX_NUM_ARGS];
  payload_t payload;
  int num_args = 0;
  int ct = 0;  
  
  for (int x = offset; x <= offset+width-1; x++) {
    std::vector<std::pair<long, long> > deps = g.dependencies(dset, x);   
    num_args = 0;
    ct = 0;    
    
    if (deps.size() == 0) {
      num_args = 1;
    //   debug_printf(1, "%d[%d] ", x, num_args);
      args[ct].x = x;
      args[ct].y = t % nb_fields;
      ct ++;
    } else {
      if (t == 0) {
        num_args = 1;
        // debug_printf(1, "%d[%d] ", x, num_args);
        args[ct].x = x;
        args[ct].y = t % nb_fields;
        ct ++;
      } else {
        num_args = 1;
        args[ct].x = x;
        args[ct].y = t % nb_fields;
        ct ++;
        long last_offset = g.offset_at_timestep(t-1);
        long last_width = g.width_at_timestep(t-1);
        for (std::pair<long, long> dep : deps) {
          num_args += dep.second - dep.first + 1;
        //   debug_printf(1, "%d[%d, %d, %d] ", x, num_args, dep.first, dep.second); 
          for (int i = dep.first; i <= dep.second; i++) {
            if (i >= last_offset && i < last_offset + last_width) {
              args[ct].x = i;
              args[ct].y = (t-1) % nb_fields;
              ct ++;
            } else {
              num_args --;
            }
          }
        }
      }
    }
    
    assert(num_args == ct);
    
    payload.y = t;
    payload.x = x;
    payload.graph = g;
    insert_task(args, num_args, payload, idx);
  }
}

static inline void task1(tile_t tile_out, payload_t payload)
{
  int tid = hpx::get_worker_thread_num();
  TaskGraph graph = payload.graph;
  std::string output_ptr = tile_out.output_buff;
  size_t output_bytes= graph.output_bytes_per_task;
  std::vector<std::string> input_ptrs;
  std::vector<size_t> input_bytes;

  input_ptrs.push_back(tile_out.output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  
  graph.execute_point(payload.y, payload.x, reinterpret_cast<char*>(output_ptr.data()), output_bytes,
                      reinterpret_cast<const char**>(input_ptrs.data()), input_bytes.data(), input_ptrs.size(), extra_local_memory[tid], graph.scratch_bytes_per_task);
 
}

static inline void task2(tile_t tile_out, tile_t tile_in1, payload_t payload)
{
  int tid = hpx::get_worker_thread_num();
  TaskGraph graph = payload.graph;
  std::string output_ptr = tile_out.output_buff;
  size_t output_bytes= graph.output_bytes_per_task;
  std::vector<std::string> input_ptrs;
  std::vector<size_t> input_bytes;

  input_ptrs.push_back(tile_in1.output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);

  
  graph.execute_point(payload.y, payload.x, reinterpret_cast<char*>(output_ptr.data()), output_bytes,
                      reinterpret_cast<const char**>(input_ptrs.data()), input_bytes.data(), input_ptrs.size(), extra_local_memory[tid], graph.scratch_bytes_per_task);

}


static inline void task3(tile_t tile_out, tile_t tile_in1, tile_t tile_in2, payload_t payload)
{
  int tid = hpx::get_worker_thread_num();
  
  TaskGraph graph = payload.graph;
  std::string output_ptr = tile_out.output_buff;
  size_t output_bytes= graph.output_bytes_per_task;
  std::vector<std::string> input_ptrs;
  std::vector<size_t> input_bytes;

  input_ptrs.push_back(tile_in1.output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back(tile_in2.output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);

  graph.execute_point(payload.y, payload.x, reinterpret_cast<char*>(output_ptr.data()),
    output_bytes, reinterpret_cast<const char**>(input_ptrs.data()),
    input_bytes.data(), input_ptrs.size(), extra_local_memory[tid],
    graph.scratch_bytes_per_task);
}

static inline void task4(tile_t tile_out, tile_t tile_in1, tile_t tile_in2,
  tile_t tile_in3, payload_t payload)
{
  int tid = hpx::get_worker_thread_num();
  
  TaskGraph graph = payload.graph;
  std::string output_ptr = tile_out.output_buff;
  size_t output_bytes= graph.output_bytes_per_task;
  std::vector<std::string> input_ptrs;
  std::vector<size_t> input_bytes;

  input_ptrs.push_back(tile_in1.output_buff);
  // input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back(tile_in2.output_buff);
  input_ptrs.push_back(tile_in3.output_buff);
  for(int i = 0 ; i < 3 ; ++i)
  {
    input_bytes.push_back(graph.output_bytes_per_task);
  }

  graph.execute_point(payload.y, payload.x, reinterpret_cast<char*>(output_ptr.data()),
    output_bytes, reinterpret_cast<const char**>(input_ptrs.data()),
    input_bytes.data(), input_ptrs.size(), extra_local_memory[tid],
    graph.scratch_bytes_per_task);
}

// static inline void task5(tile_t tile_out, tile_t tile_in1, tile_t tile_in2,
//   tile_t tile_in3, tile_t tile_in4, payload_t payload)
// {
//   int tid = hpx::get_worker_thread_num();
  
//   TaskGraph graph = payload.graph;
//   std::vector<char> output_ptr = tile_out.output_buff;
//   size_t output_bytes= graph.output_bytes_per_task;
//   std::vector<std::s> input_ptrs;
//   std::vector<size_t> input_bytes;

//   input_ptrs.push_back(tile_in1.output_buff);
//   // input_bytes.push_back(graph.output_bytes_per_task);
//   input_ptrs.push_back(tile_in2.output_buff);
//   input_ptrs.push_back(tile_in3.output_buff);
//   input_ptrs.push_back(tile_in4.output_buff);

//   for(int i = 0 ; i < 4 ; ++i)
//   {
//     input_bytes.push_back(graph.output_bytes_per_task);
//   }

//   graph.execute_point(payload.y, payload.x, reinterpret_cast<char*>(output_ptr.data()),
//     output_bytes, reinterpret_cast<const char**>(input_ptrs.data()),
//     input_bytes.data(), input_ptrs.size(), extra_local_memory[tid],
//     graph.scratch_bytes_per_task);
// }

void OpenMPApp::insert_task(task_args_t *args, int num_args, payload_t payload, size_t graph_id)
{
    std::vector<tile_t> mat = matrix[graph_id].data;
    int x0 = args[0].x;
    int y0 = args[0].y;

    switch(num_args) {
    case 1:
    {
      task1(mat[y0 * matrix[graph_id].N + x0], payload);
    break;
    }
    case 2:
    {
      int x1 = args[1].x;
      int y1 = args[1].y;
      task2(mat[y0 * matrix[graph_id].N + x0], 
        mat[y1 * matrix[graph_id].N + x1], payload);
    break;
    }
    case 3:
    {
      int x1 = args[1].x;
      int y1 = args[1].y;
      int x2 = args[2].x;
      int y2 = args[2].y;
      task3(mat[y0 * matrix[graph_id].N + x0], 
            mat[y1 * matrix[graph_id].N + x1], 
            mat[y2 * matrix[graph_id].N + x2], payload);
    break;
    }
    case 4:
    {
      int x1 = args[1].x;
      int y1 = args[1].y;
      int x2 = args[2].x;
      int y2 = args[2].y;
      int x3 = args[3].x;
      int y3 = args[3].y;
      task4(mat[y0 * matrix[graph_id].N + x0], 
            mat[y1 * matrix[graph_id].N + x1], 
            mat[y2 * matrix[graph_id].N + x2], 
            mat[y3 * matrix[graph_id].N + x3], payload);
    break;
    }
    }
}


// void OpenMPApp::debug_printf(int verbose_level, const char *format, ...)
// {
//   if (verbose_level > VERBOSE_LEVEL) {
//     return;
//   }
//   va_list args;
//   va_start(args, format);
//   vprintf(format, args);
//   va_end(args);
// }

int main(int argc, char ** argv)
{
  OpenMPApp app(argc, argv);
  app.execute_main_loop();

  return 0;
}
