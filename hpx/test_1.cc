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
#include <hpx/modules/collectives.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <utility>
#include <vector>

using namespace hpx::collectives;

///////////////////////////////////////////////////////////////////////////////
constexpr char const* channel_communicator_basename =
"lalalalala";

///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
std::vector<std::vector<char>> test_channel_communicator_get_first_comm(
    std::size_t site, hpx::collectives::channel_communicator comm,
    std::vector<std::vector<char>> output_vec,
    std::vector<std::vector<char>> inputs)
{       
  std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);
  
  using data_type = std::vector<char>;
  auto output_vector_to_send = output_vec[site];
  
  
  // receive the values sent above
  hpx::future<data_type> gets;
  
  
  std::uint32_t another_loc = (site + 1) % num_localities;
  
  
  gets = get<data_type>(comm, that_site_arg(another_loc));
  
  
  // send a value to each of the participating sites
  hpx::future<void> sets;
  
  
  sets = set(comm, that_site_arg(another_loc), output_vector_to_send);
  
  
  hpx::wait_all(sets, gets);
  
  //std::vector<std::vector<char>> inputs(num_localities);
  auto input_vector_when_receive = inputs[site];
  input_vector_when_receive = gets.get();

  std::cout << "Testing: this loc is: " << site << " " << ", another loc is: "
            << another_loc << ", "
            << input_vector_when_receive[0] << input_vector_when_receive[1] << input_vector_when_receive[2]<< '\n';
  return inputs;
}

void test_main()
{
  std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);
  
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

  using data_ = std::vector<std::vector<char>>;
  
  std::vector<hpx::future<data_>> tasks;
  tasks.reserve(num_localities);

  std::vector<char> output_loc0 ={'m', 'p', 'i'};
  std::vector<char> output_loc1 ={'h', 'p', 'x'};
  std::vector<std::vector<char>> output_vec = {output_loc0, output_loc1};

  std::vector<std::vector<char>> inputs(num_localities);
  
  for (std::size_t i = 0; i != num_localities; ++i)
  {
      std::cout << "task: " << i << '\n';
      tasks.push_back(hpx::async(
          test_channel_communicator_get_first_comm, i, comms[i], output_vec,
          inputs));
  }
  
  hpx::wait_all(tasks);

}

int hpx_main()
{

  test_main();
  
  return hpx::finalize();
}

int main(int argc, char* argv[])
{
  HPX_TEST_EQ(hpx::init(argc, argv), 0);
  return hpx::util::report_errors();
}
#endif
