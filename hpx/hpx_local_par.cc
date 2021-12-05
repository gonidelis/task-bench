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
// #include <hpx/parallel/algorithms/for_loop.hpp>
// #include <hpx/async_local/dataflow.hpp>
#include <chrono>

#include <stdarg.h>
#include <assert.h>
#include <string.h>
#include <algorithm> 
#include <unistd.h>
#include "../core/core.h"
#include "../core/timer.h"

#define VERBOSE_LEVEL 0

#define USE_CORE_VERIFICATION

#define MAX_NUM_ARGS 10

typedef struct tile_s {   // does this thing represent the dependency or the data?
  float dep; // might turn that into a future
  char *output_buff;
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
  tile_t *data;   // single tile or array of tiles?: It's an array of tiles
  
  int M;
  int N;
}matrix_t;

char **extra_local_memory;

// static inline void task(tile_t *tile_out, payload_t payload, int num_args)
// {
//   int tid = hpx::get_worker_thread_num();

//   TaskGraph graph = payload.graph;
//   char *output_ptr = (char*)tile_out->output_buff; // that's the output - dependecy is being created on that thingy
//   size_t output_bytes= graph.output_bytes_per_task;
//   std::vector<const char *> input_ptrs;
//   std::vector<size_t> input_bytes;
//   input_ptrs.push_back((char*)tile_out->output_buff);
//   input_bytes.push_back(graph.output_bytes_per_task);
  
//   graph.execute_point(payload.y, payload.x, output_ptr, output_bytes,
//                       input_ptrs.data(), input_bytes.data(), input_ptrs.size(), extra_local_memory[tid], graph.scratch_bytes_per_task);
// }

static inline void task(tile_t *tile_out, std::vector<tile_t *> const &tile_ins, payload_t payload, int num_args)
{
  int tid = hpx::get_worker_thread_num();
  TaskGraph graph = payload.graph;
  char *output_ptr = (char*)tile_out->output_buff;
  size_t output_bytes= graph.output_bytes_per_task;
  std::vector<const char *> input_ptrs;
  std::vector<size_t> input_bytes;

  if(num_args > 1)
  {
    input_ptrs.reserve(num_args);
    input_bytes.reserve(num_args);
    for(int i = 1 ; i < num_args ; ++i)
    {
      input_ptrs.push_back((char*)tile_ins[i]->output_buff);
      input_bytes.push_back(graph.output_bytes_per_task);
    }
  }
  else
  {
    input_ptrs.push_back((char*)tile_out->output_buff);
    input_bytes.push_back(graph.output_bytes_per_task);
  }
  
  graph.execute_point(payload.y, payload.x, output_ptr, output_bytes,
                      input_ptrs.data(), input_bytes.data(), input_ptrs.size(), extra_local_memory[tid], graph.scratch_bytes_per_task);
}



struct OpenMPApp : public App {
  OpenMPApp(int argc, char **argv);
  ~OpenMPApp();
  void execute_main_loop();
  void execute_timestep(size_t idx, long t);
private:
  void insert_task(task_args_t *args, int num_args, payload_t payload, size_t graph_id);
  void debug_printf(int verbose_level, const char *format, ...);
private:
  int nb_workers;
//  matrix_t *matrix;
};

matrix_t *matrix = NULL;

OpenMPApp::OpenMPApp(int argc, char **argv)
  : App(argc, argv)
{ 
  nb_workers = 1;
  
  for (int k = 1; k < argc; k++) {
    if (!strcmp(argv[k], "-worker")) {
      nb_workers = atol(argv[++k]);
    }
  }
  
  matrix = (matrix_t *)malloc(sizeof(matrix_t) * graphs.size());
  
  size_t max_scratch_bytes_per_task = 0;
  
  for (unsigned i = 0; i < graphs.size(); i++) {
    TaskGraph &graph = graphs[i];
    
    matrix[i].M = graph.nb_fields;
    matrix[i].N = graph.max_width;
    matrix[i].data = (tile_t*)malloc(sizeof(tile_t) * matrix[i].M * matrix[i].N);
  
    for (int j = 0; j < matrix[i].M * matrix[i].N; j++) {
      matrix[i].data[j].output_buff = (char *)malloc(sizeof(char) * graph.output_bytes_per_task);
    }
    
    if (graph.scratch_bytes_per_task > max_scratch_bytes_per_task) {
      max_scratch_bytes_per_task = graph.scratch_bytes_per_task;
    }
    
    printf("graph id %d, M = %d, N = %d, data %p, nb_fields %d\n", i, matrix[i].M, matrix[i].N, matrix[i].data, graph.nb_fields);
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
  
  // omp_set_dynamic(1);
  // omp_set_num_threads(nb_workers);
  if (max_scratch_bytes_per_task > 0) {
    // #pragma omp parallel

      int tid = hpx::get_worker_thread_num();
      // printf("Im tid %d\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", tid);
      TaskGraph::prepare_scratch(extra_local_memory[tid], sizeof(char)*max_scratch_bytes_per_task);
  }


}

OpenMPApp::~OpenMPApp()
{
  for (unsigned i = 0; i < graphs.size(); i++) {
    for (int j = 0; j < matrix[i].M * matrix[i].N; j++) {
      free(matrix[i].data[j].output_buff);
      matrix[i].data[j].output_buff = NULL;
    }
    free(matrix[i].data);
    matrix[i].data = NULL;
  }
  
  free(matrix);
  matrix = NULL;
  
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
  

  hpx::for_loop(hpx::execution::seq,
    int(0), graphs.size(), [&](int i) {
    const TaskGraph &g = graphs[i];
    hpx::for_loop(hpx::execution::seq, // try doing that with a dataflow/taskblock
    int(0), g.timesteps, [&](int y) {
      execute_timestep(i, y);
    });

  });    

  double elapsed = Timer::time_end();
  report_timing(elapsed);
}


// int flag = 1;
void OpenMPApp::execute_timestep(size_t idx, long t)
{
  tile_t *mat = matrix[idx].data;

  const TaskGraph &g = graphs[idx];
  long offset = g.offset_at_timestep(t);
  long width = g.width_at_timestep(t);
  long dset = g.dependence_set_at_timestep(t);
  int nb_fields = g.nb_fields;
   
  // std::vector<hpx::future<void>> futures(width);
  std::size_t work = (1 * hpx::get_num_worker_threads());
  hpx::execution::static_chunk_size cs(width/work);
  // std::vector<double> timings(width);

  // for(int x = offset; x <= offset+width-1; x++) 
  hpx::for_loop(hpx::execution::par.with(cs), offset, offset+width, 
    [&](int x)
    {
      // auto start = std::chrono::steady_clock::now();

      int num_args = 0;
      int ct = 0;  
      task_args_t args[MAX_NUM_ARGS];

      std::vector<std::pair<long, long> > deps = g.dependencies(dset, x);   
      num_args = 0;
      ct = 0;    
      payload_t payload;  // @Giannis: try moving that inside the for_loop
      
      if (deps.size() == 0) {
        num_args = 1;
        debug_printf(1, "%d[%d] ", x, num_args);
        args[ct].x = x;
        args[ct].y = t % nb_fields;
        ct ++;
      } else {
        if (t == 0) {
          num_args = 1;
          debug_printf(1, "%d[%d] ", x, num_args);
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
            debug_printf(1, "%d[%d, %d, %d] ", x, num_args, dep.first, dep.second); 
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

      
      // hpx::shared_future<int> num_args_f = hpx::make_ready_future(num_args);
      // hpx::shared_future<task_args_t> args_f = hpx::make_ready_future(&args);
      // hpx::shared_future<payload_t> payload_f = hpx::make_ready_future(payload);
      // hpx::shared_future<size_t> idx_f = hpx::make_ready_future(idx);

      payload.y = t;
      payload.x = x;
      payload.graph = g;
      // insert_task(args, num_args, payload, idx);
      std::vector<tile_t*> tiles(num_args);
      tile_t *tile_out = &mat[args[0].y * matrix[idx].N + args[0].x];
      for(int i = 1 ; i < num_args ; ++i)
      {
        tiles[i] = &mat[args[i].y * matrix[idx].N + args[i].x];
      }

      // futures[x] = hpx::async(task, tile_out, std::move(tiles), payload, num_args);
      task(tile_out, tiles, payload, num_args);

      // 1. Task Block Implementation
      // hpx::define_task_block(
      //   hpx::execution::par,
      //   [&](hpx::task_block<>& tb) {
      //     tb.run([&] {task(tile_out, tiles, payload, num_args);} );
      // });
      // auto finish = std::chrono::steady_clock::now();
      // timings[x] = std::chrono::duration_cast<
      // std::chrono::duration<double> >(finish - start).count();
    });
  // if(flag)
  // {
  //   for(int i = 0 ; i < timings.size() ; ++i)
  //   {
  //     std::cout << timings[i] << " ";
  //   }
  //   std::cout << std::endl;
  //   flag = 0;
  // }
}

void OpenMPApp::debug_printf(int verbose_level, const char *format, ...)
{
  if (verbose_level > VERBOSE_LEVEL) {
    return;
  }
  va_list args;
  va_start(args, format);
  vprintf(format, args);
  va_end(args);
}

int main(int argc, char ** argv)
{
  OpenMPApp app(argc, argv);
  app.execute_main_loop();

  return 0;
}
