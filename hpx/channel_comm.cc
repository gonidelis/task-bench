// Copyright (c) 2021 Hartmut Kaiser
// Copyright (c) 2021 Nanmiao Wu
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/actions_base/plain_action.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/modules/collectives.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/serialization.hpp>

#include <cstddef>
#include <utility>
#include <vector>

using namespace hpx::collectives;

///////////////////////////////////////////////////////////////////////////////
constexpr char const* channel_communicator_basename =
"lalalalala_hpx_hpx";

///////////////////////////////////////////////////////////////////////////////
int channel_comm(
    hpx::collectives::channel_communicator comm,
    int output_vec)
{    
  
  std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);
  
  using data_type = int;

  // receive the values sent above
  hpx::future<data_type> gets;
  
  
  std::uint32_t another_loc = (hpx::get_locality_id() + 1) % num_localities;
  std::cout << "Entering beginning: this loc is: " << hpx::get_locality_id() << " " << ", another loc is: "
            << another_loc << '\n';  
  
  gets = get<data_type>(comm, that_site_arg(another_loc));
  
  // send a value to each of the participating sites
  hpx::future<void> sets;
  
  sets = set(comm, that_site_arg(another_loc), output_vec);
  
  hpx::wait_all(sets, gets);
  
  int inputs = gets.get();

  std::cout << "Testing: this loc is: " << hpx::get_locality_id() << " " << ", another loc is: "
            << another_loc << ", "
            << inputs << '\n';
  return inputs;
}
////////////////////////////////////////////////////////////////////////
HPX_PLAIN_ACTION(channel_comm, channel_comm_action);

////////////////////////////////////////////////////////////////////////

int hpx_main()
{
  //////////////////////////////////////////////////////////////////////
  std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);
  int output_loc0 = 42;
  int output_loc1 = 4422;
  std::vector<int> output_vec = {output_loc0, output_loc1};
  std::vector<int> inputs(num_localities);

  std::vector<hpx::id_type> localities = hpx::find_all_localities();
  
  // for each site, create new channel_communicator
  std::vector<hpx::collectives::channel_communicator> comms;
  comms.reserve(num_localities);
  
  for (std::size_t i = 0; i != num_localities; ++i)
  {
      comms.push_back(create_channel_communicator(hpx::launch::sync,
          channel_communicator_basename,
          num_sites_arg(num_localities),
          this_site_arg(i)));
  }

  using data_ = int;
  std::vector<hpx::future<data_>> tasks;

  channel_comm_action channel_comm;

  for(std::size_t i = 0; i < localities.size(); ++i)
  {
      std::cout << "task: " << i << '\n';
      tasks.push_back(hpx::async(channel_comm,
        localities[i], comms[i], output_vec[i]));
  }

  hpx::wait_all(tasks);
  
  return hpx::finalize();
}

int main(int argc, char* argv[])
{
  HPX_TEST_EQ(hpx::init(argc, argv), 0);
  return hpx::util::report_errors();
}
#endif
