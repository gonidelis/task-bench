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

#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include "hpx/hpx.hpp"
#include "hpx/hpx_init.hpp"
#include "hpx/include/compute.hpp"
#include "hpx/local/chrono.hpp"
#include "hpx/modules/collectives.hpp"

///////////////////////////////////////////////////////////////////////////////
constexpr char const* channel_communicator_name =
  "hpx_comm_task";

namespace detail{
  std::tuple<long, long, long> tile(std::uint32_t loc_idx, long column, std::uint32_t numlocs)
  {
    long first_point;
    long n_points = column / numlocs;
    long rest = column % numlocs;
    if (loc_idx < rest){
      n_points ++;
    }

    if (rest != 0 && loc_idx >= rest){
      first_point = (n_points + 1) * rest + n_points * (loc_idx - rest);
    } else {
      first_point = n_points * loc_idx;
    }
    long last_point = first_point + n_points - 1;
    return std::make_tuple(first_point, last_point, n_points);
  }
} // namespace detail

///////////////////////////////////////////////////////////////////////////////
int hpx_main(int argc, char *argv[]) 
{    
  // get number of localities and this locality
  std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);
  std::uint32_t this_locality = hpx::get_locality_id();

  App app(argc, argv);
  if (this_locality == 0) app.display();

  std::vector<std::vector<char> > scratch;

  // allocate channel communicator
  auto comm = hpx::collectives::create_channel_communicator(hpx::launch::sync,
      channel_communicator_name, hpx::collectives::num_sites_arg(num_localities),
      hpx::collectives::this_site_arg(this_locality));
      
  using data_type = std::vector<char>;

  std::vector<hpx::future<void>> sets;

  hpx::execution::static_chunk_size fixed(1);

  using executor = hpx::execution::experimental::fork_join_executor;
  executor exec(hpx::threads::thread_priority::default_, hpx::threads::thread_stacksize::small_,
                executor::loop_schedule::static_, std::chrono::microseconds(10));
  
  auto policy = hpx::execution::par(hpx::execution::task).on(exec).with(fixed); // this will hang using 4 or 8 nodes.

  //auto policy = hpx::execution::par.with(fixed);   // this will not hang

  for (auto graph : app.graphs) {
    //long first_point = this_locality * graph.max_width / num_localities;
    //long last_point = (this_locality + 1) * graph.max_width / num_localities - 1;
    //long n_points = last_point - first_point + 1;
    long first_point, last_point, n_points;
    std::tie(first_point, last_point, n_points) = detail::tile(this_locality,graph.max_width, num_localities);

    size_t scratch_bytes = graph.scratch_bytes_per_task;
    scratch.emplace_back(scratch_bytes * n_points);

    char *scratch_ptr = scratch.back().data();

    hpx::for_loop(policy, first_point, last_point + 1,
      [&](long point)
      {
        long point_index = point - first_point;
        TaskGraph::prepare_scratch(scratch_ptr + scratch_bytes * point_index, scratch_bytes);
      }); //
  }
  
  double elapsed = 0.0;

  for (int iter = 0; iter < 2; ++iter) {
    
    hpx::chrono::high_resolution_timer timer;
    auto duration_lambda_execute = 0.0;  

    for (auto graph : app.graphs) {
   
      //long first_point = this_locality * graph.max_width / num_localities;
      //long last_point = (this_locality + 1) * graph.max_width / num_localities - 1;
      //long n_points = last_point - first_point + 1;
      long first_point, last_point, n_points;
      std::tie(first_point, last_point, n_points) = detail::tile(this_locality,graph.max_width, num_localities);

      size_t scratch_bytes = graph.scratch_bytes_per_task;
      char *scratch_ptr = scratch[graph.graph_index].data();

      std::vector<int> locality_by_point(graph.max_width);
      for (int r = 0; r < num_localities; ++r) {
        long r_first_point = r * graph.max_width / num_localities;
        long r_last_point = (r + 1) * graph.max_width / num_localities - 1;
        for (long p = r_first_point; p <= r_last_point; ++p) {
          locality_by_point[p] = r;
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

        long offset = graph.offset_at_timestep(timestep);
        long width = graph.width_at_timestep(timestep);

        long last_offset = graph.offset_at_timestep(timestep-1);
        long last_width = graph.width_at_timestep(timestep-1);

        long dset = graph.dependence_set_at_timestep(timestep);
        auto &deps = dependencies[dset];
        auto &rev_deps = reverse_dependencies[dset];

        sets.clear();
        
        for (long point = first_point; point <= last_point; ++point) {
          
          long point_index = point - first_point;

          auto &point_inputs = inputs[point_index];
          auto &point_n_inputs = n_inputs[point_index];
          auto &point_output = outputs[point_index];

          auto &point_deps = deps[point_index];
          auto &point_rev_deps = rev_deps[point_index];

          // Send 
          if (point >= last_offset && point < last_offset + last_width) {
            sets.clear();
            for (auto interval : point_rev_deps) {
              for (long dep = interval.first; dep <= interval.second; dep++) {
                if (dep < offset || dep >= offset + width || (first_point <= dep && dep <= last_point)) {
                  continue;
                }
                sets.push_back(hpx::collectives::set(comm, 
                    hpx::collectives::that_site_arg(locality_by_point[dep]), point_output));
              }
            }
          }  // send

          //std::cout << "After send and before wait all futures, this_locality is: " << this_locality
          //          << ", timestep is: " << timestep << "," << "iter is: " << iter
          //          << std::endl;

          hpx::wait_all(sets);

          //std::cout << "Done send, this_locality is: " << this_locality
          //          << ", timestep is: " << timestep << "," << "iter is: " << iter
          //          << std::endl;
              
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
                  auto got_msg = hpx::collectives::get<data_type>(comm, 
                          hpx::collectives::that_site_arg(locality_by_point[dep]));
                  data_type rec_msg = got_msg.get();
                  point_inputs[point_n_inputs].assign(rec_msg.begin(), rec_msg.end());

                  //std::cout << "After receive, this_locality is: " << this_locality
                  //          << ", timestep is: " << timestep << "," << "iter is: " << iter
                  //          << std::endl;

                }
                point_n_inputs++;
              }
            }
          } // receive

          //std::cout << "Done receive, this_locality is: " << this_locality
          //                  << ", timestep is: " << timestep << "," << "iter is: " << iter
          //                  << std::endl;
          
        } // for loop for exchange

        //std::cerr << "++ Done for exchange data"
        //          << ", iter is: " << iter << ", this_locality is: " 
        //          << this_locality 
        //          << ", timestep is: " << timestep
        //          << "\n";

        hpx::for_loop(
            policy, std::max(first_point, offset),
            std::min(last_point, offset + width - 1) + 1, [&](long point) {
              long point_index = point - first_point;

              auto &point_input_ptr = input_ptr[point_index];
              auto &point_input_bytes = input_bytes[point_index];
              auto &point_n_inputs = n_inputs[point_index];
              auto &point_output = outputs[point_index];

              //std::cerr << "===> Inside for_loop, before execute point"
              //    << ", iter is: " << iter << ", this_locality is: " 
              //    << this_locality 
              //    << ", timestep is: " << timestep
              //    << "\n";

              graph.execute_point(timestep, point, point_output.data(),
                                  point_output.size(), point_input_ptr.data(),
                                  point_input_bytes.data(), point_n_inputs,
                                  scratch_ptr + scratch_bytes * point_index,
                                  scratch_bytes);

              //std::cerr << "<===Inside for_loop, after execute point"
              //    << ", iter is: " << iter << ", this_locality is: " 
              //    << this_locality 
              //    << ", timestep is: " << timestep
              //    << "\n";

            });  // hpx_for loop
          
        //std::cerr << "***Done for for_loop execute point"
        //          << ", iter is: " << iter << ", this_locality is: " 
        //          << this_locality 
        //          << ", timestep is: " << timestep
        //          << "\n";

      } // for time steps loop 
      

    } // for graphs loop
    elapsed = timer.elapsed(); 

    //if (iter == 1) {
    //  std::cout << "Time by executing lambda function is: "
    //            << duration_lambda_execute << " nanoseconds" << std::endl;
    //}

  } // for 2-time iter

  

  if (this_locality == 0) {
    app.report_timing(elapsed);
  }

  return hpx::finalize();;
}
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Initialize and run HPX, this example requires to run hpx_main on all
    // localities
    std::vector<std::string> const cfg = {
        "hpx.run_hpx_main!=1",
        "--hpx:ini=hpx.commandline.allow_unknown!=1",
        "--hpx:ini=hpx.commandline.aliasing!=0"
    };
    hpx::init_params init_args;
    init_args.cfg = cfg;  
    
    return hpx::init(argc, argv, init_args);
}
#endif
