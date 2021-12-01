//  Copyright (c)      2021 Nanmiao Wu
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "../core/core.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string> 
#include <tuple>

#include "hpx/hpx_init.hpp"
#include "hpx/hpx.hpp"
#include "hpx/local/chrono.hpp"
#include "hpx/modules/async_mpi.hpp"

#include "mpi.h"

///////////////////////////////////////////////////////////////////////////////

//char const* const barrier_name = "hpx_barrier_task";
///////////////////////////////////////////////////////////////////////////////
int hpx_main(int argc, char *argv[]) 
{ 
  
  int n_ranks, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  App app(argc, argv);
  if (rank == 0) app.display();

  std::vector<std::vector<char> > scratch;

  hpx::mpi::experimental::enable_user_polling enable_polling;
  hpx::mpi::experimental::executor mpi_exec(MPI_COMM_WORLD);

  hpx::execution::static_chunk_size fixed(1);

  auto policy = hpx::execution::par.with(fixed); 

  std::vector<hpx::future<int>> requests;

  //hpx::lcos::barrier HPX_barrier(barrier_name);

  for (auto graph : app.graphs) {
    long first_point = rank * graph.max_width / n_ranks;
    long last_point = (rank + 1) * graph.max_width / n_ranks - 1;
    long n_points = last_point - first_point + 1;

    size_t scratch_bytes = graph.scratch_bytes_per_task;
    scratch.emplace_back(scratch_bytes * n_points);
    TaskGraph::prepare_scratch(scratch.back().data(), scratch.back().size());
  }
  
  double elapsed = 0.0;

  for (int iter = 0; iter < 2; ++iter) {
    MPI_Barrier(MPI_COMM_WORLD);
    hpx::chrono::high_resolution_timer timer;

    for (auto graph : app.graphs) {
      long first_point = rank * graph.max_width / n_ranks;
      long last_point = (rank + 1) * graph.max_width / n_ranks - 1;
      long n_points = last_point - first_point + 1;

      size_t scratch_bytes = graph.scratch_bytes_per_task;
      char *scratch_ptr = scratch[graph.graph_index].data();

      std::vector<int> locality_by_point(graph.max_width);
      std::vector<int> tag_bits_by_point(graph.max_width);

      for (int r = 0; r < n_ranks; ++r) {
        long r_first_point = r * graph.max_width / n_ranks;
        long r_last_point = (r + 1) * graph.max_width / n_ranks - 1;
        for (long p = r_first_point; p <= r_last_point; ++p) {
          locality_by_point[p] = r;
          tag_bits_by_point[p] = p - r_first_point;
        }
      }

      long max_deps = 0;
      for (long dset = 0; dset < graph.max_dependence_sets(); ++dset) {
        for (long point = first_point; point <= last_point; ++point) {
          long deps = 0;
          for (auto interval : graph.dependencies(dset, point)) {
            deps += interval.second - interval.first + 1;
          }
          max_deps = std::max(max_deps, deps);
        }
      }

      // Create input and output buffers.
      std::vector<std::vector<std::vector<char> > > inputs(n_points);
      std::vector<std::vector<const char *> > input_ptr(n_points);
      std::vector<std::vector<size_t> > input_bytes(n_points);
      std::vector<long> n_inputs(n_points);
      std::vector<std::vector<char> > outputs(n_points);

      for (long point = first_point; point <= last_point; ++point) {
        long point_index = point - first_point;

        auto &point_inputs = inputs[point_index];
        auto &point_input_ptr = input_ptr[point_index];
        auto &point_input_bytes = input_bytes[point_index];

        point_inputs.resize(max_deps);
        point_input_ptr.resize(max_deps);
        point_input_bytes.resize(max_deps);

        for (long dep = 0; dep < max_deps; ++dep) {
          point_inputs[dep].resize(graph.output_bytes_per_task);
          point_input_ptr[dep] = point_inputs[dep].data();
          point_input_bytes[dep] = point_inputs[dep].size();
        }

        auto &point_outputs = outputs[point_index];
        point_outputs.resize(graph.output_bytes_per_task);
      }

      // Cache dependencies.
      std::vector<std::vector<std::vector<std::pair<long, long> > > > dependencies(graph.max_dependence_sets());
      std::vector<std::vector<std::vector<std::pair<long, long> > > > reverse_dependencies(graph.max_dependence_sets());
      for (long dset = 0; dset < graph.max_dependence_sets(); ++dset) {
        dependencies[dset].resize(n_points);
        reverse_dependencies[dset].resize(n_points);

        for (long point = first_point; point <= last_point; ++point) {
          long point_index = point - first_point;

          dependencies[dset][point_index] = graph.dependencies(dset, point);
          reverse_dependencies[dset][point_index] = graph.reverse_dependencies(dset, point);
        }
      }
    
      
      for (long timestep = 0; timestep < graph.timesteps; ++timestep) {
        //std::cout << "timestep: " << timestep << "\n";
        long offset = graph.offset_at_timestep(timestep);
        long width = graph.width_at_timestep(timestep);

        long last_offset = graph.offset_at_timestep(timestep-1);
        long last_width = graph.width_at_timestep(timestep-1);

        long dset = graph.dependence_set_at_timestep(timestep);
        auto &deps = dependencies[dset];
        auto &rev_deps = reverse_dependencies[dset];

        requests.clear();

        for (long point = first_point; point <= last_point; ++point) {
          long point_index = point - first_point;

          auto &point_inputs = inputs[point_index];
          auto &point_n_inputs = n_inputs[point_index];
          auto &point_output = outputs[point_index];

          auto &point_deps = deps[point_index];
          auto &point_rev_deps = rev_deps[point_index];

          // Send 
          if (point >= last_offset && point < last_offset + last_width) {
            for (auto interval : point_rev_deps) {
              for (long dep = interval.first; dep <= interval.second; dep++) {
                if (dep < offset || dep >= offset + width || (first_point <= dep && dep <= last_point)) {
                  continue;
                }
                int from = tag_bits_by_point[point];
                int to = tag_bits_by_point[dep];
                int tag = (from << 1) | to;

                requests.push_back(hpx::async(mpi_exec, MPI_Isend, 
                    point_output.data(), point_output.size(), MPI_BYTE, 
                    locality_by_point[dep], tag));
                
              }
            }
          }  // send
          //std::cout << "Done send \n";

          // Receive 
          point_n_inputs = 0;
          if (point >= offset && point < offset + width) {
            for (auto interval : point_deps) {
              for (long dep = interval.first; dep <= interval.second; ++dep) {
                if (dep < last_offset || dep >= last_offset + last_width) {
                  continue;
                }
                // Use shared memory for on-node data.
                if (first_point <= dep && dep <= last_point) {
                  auto &output = outputs[dep - first_point];
                  point_inputs[point_n_inputs].assign(output.begin(), output.end());
                } else {
                  int from = tag_bits_by_point[dep];
                  int to = tag_bits_by_point[point];
                  int tag = (from << 1) | to;

                  requests.push_back(hpx::async(
                      mpi_exec, MPI_Irecv, point_inputs[point_n_inputs].data(),
                      point_inputs[point_n_inputs].size(), MPI_BYTE,
                      locality_by_point[dep], tag));
                }
                point_n_inputs++;
              }
            }
          } // receive

          //std::cout << "Done rec \n";
        
        } // for loop for exchange
        //std::cout << "timestep: " << timestep << ", this rank: " << rank
        //          << ", after collecting requests for exchanging data and requests size: "
        //          << requests.size() << "\n";
        hpx::wait_all(requests);
        //std::cerr << "timestep: " << timestep << ", requests size: " << requests.size() << "\n";
        //for (auto& f : requests) {
        //    //std::cout << "f \n";
        //    f.get();
        //}
        
        //HPX_barrier.wait();
        //std::cout << "timestep: " << timestep << ", this rank: " << rank
        //          << ", after reaching barrier ~~~~~ \n";
        long start = std::max(first_point, offset);
        long end = std::min(last_point + 1, offset + width);

        hpx::for_loop(policy, start, end, [&](int point) {
          long point_index = point - first_point;

          auto &point_input_ptr = input_ptr[point_index];
          auto &point_input_bytes = input_bytes[point_index];
          auto &point_n_inputs = n_inputs[point_index];
          auto &point_output = outputs[point_index];

          graph.execute_point(
              timestep, point, point_output.data(), point_output.size(),
              point_input_ptr.data(), point_input_bytes.data(), point_n_inputs,
              scratch_ptr + scratch_bytes * point_index, scratch_bytes);
        });  // hpx_for loop
        //std::cout << "timestep: " << timestep << ", this rank: " << rank
        //          << ", Done execute points ==== \n";
          
      } // for time steps loop 
      

    } // for graphs loop
    //HPX_barrier.wait();
    MPI_Barrier(MPI_COMM_WORLD);
    elapsed = timer.elapsed(); 

  } // for 2-time iter

  

  if (rank == 0) {
    app.report_timing(elapsed);
  }

  return hpx::finalize();
}
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // all ranks run their main function
    std::vector<std::string> const cfg = {
        "hpx.run_hpx_main!=1",
        "--hpx:ini=hpx.commandline.allow_unknown!=1",
        "--hpx:ini=hpx.commandline.aliasing!=0",
        "--hpx:ini=hpx.stacks.small_size!=0x20000"
    };

    // Init MPI
    int provided = MPI_THREAD_MULTIPLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE)
    {
        std::cout << "Provided MPI is not : MPI_THREAD_MULTIPLE " << provided
                  << std::endl;
    }

    // Initialize and run HPX.
    hpx::init_params init_args;
    init_args.cfg = cfg;

    auto result = hpx::init(argc, argv, init_args);

    // Finalize MPI
    MPI_Finalize();

    return result;
}

