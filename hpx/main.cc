//  Copyright (c)      2021 Nanmiao Wu
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "../core/core.h"

#include <cstdlib>
#include <string> 
#include <tuple>

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include "hpx/hpx.hpp"
#include "hpx/hpx_init.hpp"
#include "hpx/include/actions.hpp"
#include "hpx/local/chrono.hpp"
#include "hpx/modules/collectives.hpp"

///////////////////////////////////////////////////////////////////////////////
constexpr char const* channel_communicator_basename =
  "hpx_block_task";

///////////////////////////////////////////////////////////////////////////////
std::tuple<std::vector<long>, std::vector<std::vector<std::vector<char>>> >  each_time_get_set_comm(
      hpx::collectives::channel_communicator comm,
      long first_point, long last_point, std::vector<long> n_inputs, 
      std::vector<std::vector<std::vector<char> > > inputs, 
      std::vector<std::vector<char> > outputs, long dset,
      std::vector<std::vector<std::vector<std::pair<long, long> > > > dependencies,
      std::vector<std::vector<std::vector<std::pair<long, long> > > > reverse_dependencies,
      long offset, long width, long last_offset, long last_width,
      std::vector<int> locality_by_point)
  {    
    auto &deps = dependencies[dset];
    auto &rev_deps = reverse_dependencies[dset];

    std::vector<std::size_t> point_n_inputs_future_vec;
    std::vector<std::size_t> point_inputs_future_vec;

    using data_type = std::vector<char>;
    std::vector<hpx::future<data_type>> gets;
    std::vector<hpx::future<void>> sets;

    for (long point = first_point; point <= last_point; ++point) {

      long point_index = point - first_point;

      auto &point_inputs = inputs[point_index];
      auto &point_n_inputs = n_inputs[point_index];
      auto &point_output = outputs[point_index];

      auto &point_deps = deps[point_index];
      auto &point_rev_deps = rev_deps[point_index];

      //Receive 
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
              //MPI_Irecv(point_inputs[point_n_inputs].data(),
              //          point_inputs[point_n_inputs].size(), MPI_BYTE,
              //          rank_by_point[dep], tag, MPI_COMM_WORLD, &req);
              gets.push_back(hpx::collectives::get<data_type>(comm, hpx::collectives::that_site_arg(locality_by_point[dep])));
              point_n_inputs_future_vec.push_back(point_n_inputs);
              point_inputs_future_vec.push_back(point_index);
            }
            point_n_inputs++;
          }
        }
      }

      // Send
      if (point >= last_offset && point < last_offset + last_width) {
        for (auto interval : point_rev_deps) {
          for (long dep = interval.first; dep <= interval.second; dep++) {
            if (dep < offset || dep >= offset + width || (first_point <= dep && dep <= last_point)) {
              continue;
            }
            //MPI_Isend(point_output.data(), point_output.size(), MPI_BYTE,
            //          rank_by_point[dep], tag, MPI_COMM_WORLD, &req);
            sets.push_back(hpx::collectives::set(comm, hpx::collectives::that_site_arg(locality_by_point[dep]), point_output));
          }
        }
      }
    }

    hpx::wait_all(sets, gets);


    for (std::size_t i = 0; i != gets.size(); ++i) {
      auto point_index = point_inputs_future_vec[i];

      auto &point_inputs = inputs[point_index];
      auto &point_n_inputs = n_inputs[point_index];

      point_n_inputs = point_n_inputs_future_vec[i];
      point_inputs[point_n_inputs] = gets[i].get();

    }

    return std::make_tuple(n_inputs, inputs);

  }
///////////////////////////////////////////////////////////////////////////////

HPX_PLAIN_ACTION(each_time_get_set_comm, each_time_get_set_comm_action);

///////////////////////////////////////////////////////////////////////////////

  std::tuple<std::vector<long>, std::vector<std::vector<std::vector<char>>> > assign_tasks(
      long first_point, long last_point, std::vector<long> &n_inputs, 
      std::vector<std::vector<std::vector<char> > > &inputs, 
      std::vector<std::vector<char> > outputs, long dset,
      std::vector<std::vector<std::vector<std::pair<long, long> > > > dependencies,
      std::vector<std::vector<std::vector<std::pair<long, long> > > > reverse_dependencies,
      long offset, long width, long last_offset, long last_width,
      std::vector<int> locality_by_point)
  {
    std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);

    // for each site, create new channel_communicator
    std::vector<hpx::collectives::channel_communicator> comms;
    // need to know number of tasks first? # use deps and max_deps
    comms.reserve(num_localities);

    // not general 
    for (std::size_t i = 0; i != num_localities; ++i)
    {
        comms.push_back(hpx::collectives::create_channel_communicator(hpx::launch::sync,
            channel_communicator_basename,
            hpx::collectives::num_sites_arg(num_localities),
            hpx::collectives::this_site_arg(i)));
    }

    //using data_type = std::vector<std::vector<std::vector<char> > >;
    using data_type = std::tuple<std::vector<long>, std::vector<std::vector<std::vector<char>>> >;

    std::vector<hpx::future<data_type>> tasks;
    tasks.reserve(num_localities);

    for (std::size_t i = 0; i != num_localities; ++i)
    {
        std::cout << "task: " << i << '\n';
        tasks.push_back(hpx::async(
          each_time_get_set_comm, comms[i], 
          first_point, last_point, n_inputs, inputs, outputs, dset,
          dependencies, reverse_dependencies, offset, width, last_offset, 
          last_width, locality_by_point));
    }
    hpx::wait_all(tasks);

    for (std::size_t i = 0; i != num_localities; ++i)
    {
        if (hpx::get_locality_id() == i) {
          auto data = tasks[i].get();
          n_inputs = std::get<0>(data);
          inputs = std::get<1>(data);
        }
        
    }

    return std::make_tuple(n_inputs, inputs);

  }


  void execute_point(long timestep, long first_point, long last_point, std::vector<long> &n_inputs, 
      std::vector<std::vector<std::vector<char> > > &inputs, 
      std::vector<std::vector<const char *> > &input_ptr,
      std::vector<std::vector<size_t> > &input_bytes,
      std::vector<std::vector<char> > &outputs,
      size_t scratch_bytes,
      char *scratch_ptr,
      long offset, long width, TaskGraph graph)
  {
    for (long point = std::max(first_point, offset); point <= std::min(last_point, offset + width - 1); ++point) {
      long point_index = point - first_point;

      auto &point_input_ptr = input_ptr[point_index];
      auto &point_input_bytes = input_bytes[point_index];
      auto &point_n_inputs = n_inputs[point_index];
      auto &point_output = outputs[point_index];

      graph.execute_point(timestep, point,
                          point_output.data(), point_output.size(),
                          point_input_ptr.data(), point_input_bytes.data(), point_n_inputs,
                          scratch_ptr + scratch_bytes * point_index, scratch_bytes);
    }   

  }

///////////////////////////////////////////////////////////////////////////////
int hpx_main(int argc, char *argv[]) 
{    
  // get number of localities and this locality
  std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);
  std::uint32_t this_locality = hpx::get_locality_id();

  App app(argc, argv);
  if (this_locality == 0) app.display();

  std::vector<std::vector<char> > scratch;

  for (auto graph : app.graphs) {
    long first_point = this_locality * graph.max_width / num_localities;
    long last_point = (this_locality + 1) * graph.max_width / num_localities - 1;
    long n_points = last_point - first_point + 1;

    size_t scratch_bytes = graph.scratch_bytes_per_task;
    scratch.emplace_back(scratch_bytes * n_points);
    TaskGraph::prepare_scratch(scratch.back().data(), scratch.back().size());
  }
  
  double elapsed = 0.0;
  for (int iter = 0; iter < 1; ++iter) {
    
    hpx::chrono::high_resolution_timer timer;

    for (auto graph : app.graphs) {
   
      long first_point = this_locality * graph.max_width / num_localities;
      long last_point = (this_locality + 1) * graph.max_width / num_localities - 1;
      long n_points = last_point - first_point + 1;

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

        std::tie(n_inputs, inputs) = assign_tasks(first_point, last_point, n_inputs, inputs, outputs, dset,
            dependencies, reverse_dependencies, offset, width, last_offset, 
            last_width, locality_by_point);
        
        for (long point = first_point; point <= last_point; ++point) {
          long point_index = point - first_point;

          auto &point_inputs = inputs[point_index];
          auto &point_input_ptr = input_ptr[point_index];
          auto &point_input_bytes = input_bytes[point_index];

          for (long dep = 0; dep < max_deps; ++dep) {
            point_input_ptr[dep] = point_inputs[dep].data();
            point_input_bytes[dep] = point_inputs[dep].size();
          }
        }
        
        execute_point(timestep, first_point, last_point, n_inputs, inputs, input_ptr,
            input_bytes, outputs, scratch_bytes, scratch_ptr, offset, width, graph);       
      }    


    }
  
    elapsed = timer.elapsed(); 
    std::cout << "-----------------------------" << std::endl;
  }



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
        "--hpx:init=hpx.commandline.aliasing!=0"
    };
    hpx::init_params init_args;
    init_args.cfg = cfg;  
    
    return hpx::init(argc, argv, init_args);
}
#endif