//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/local/execution.hpp>
#include <hpx/local/future.hpp>
#include <hpx/local/init.hpp>
#include <hpx/modules/async_mpi.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/program_options.hpp>

#include <array>
#include <atomic>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <mpi.h>

// This test send a message from rank 0 to rank 1 and from R1->R2 in a ring
// until the last rank which sends it back to R0.and completes an iteration
//
// For benchmarking the test has been extended to allow many iterations
// but unfortunately, if we prepost too many receives, MPI has problems
// so we do 1000 iterations per main loop and another loop around that.

// this is called on an hpx thread after the runtime starts up
int hpx_main(hpx::program_options::variables_map& vm)
{
    int rank, size;
    //
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    {
        hpx::mpi::experimental::enable_user_polling enable_polling;

        hpx::mpi::experimental::executor exec(MPI_COMM_WORLD);

        hpx::execution::experimental::limiting_executor<
            hpx::mpi::experimental::executor>
            limexec(exec, 32, 64, true);
        
        int next_rank = (rank + 1) % size;

        int msg = (rank == 0)? 1: -1;

        std::atomic<std::uint64_t> counter(2);
        int tag = 345;

        // pre-post a receive    
        hpx::future<int> f_recv =
            hpx::async(limexec, MPI_Irecv, &msg, 1, MPI_INT, next_rank, tag);

        // when recv completes
        f_recv.then([=, &exec, &msg, &counter](auto&&) {
          if (rank > 0) {
              std::cout << "rank 1 Done send \n";
              ++msg;
              --counter;
              hpx::future<int> f_send =
                  hpx::async(exec, MPI_Isend, &msg, 1, MPI_INT, next_rank, tag);
              f_send.then([=, &counter](auto&&) {
                std::cout << "Rank 0 Done rec \n";
                --counter;
              });
          }
          else {
              std::cout << "rank 0 Done send \n";
              --counter;
          }
        });

        if (rank == 0) {
          hpx::future<int> f_send = hpx::async
            (limexec, MPI_Isend, &msg, 1, MPI_INT, next_rank, tag);
          f_send.then([=, &counter](auto&&) {
            std::cout << "Done rank 0 send \n";
            --counter;
          });
        }

        hpx::mpi::experimental::wait([&]() { 
            //std::cout << "wait, rank: " << rank << ", counter: " << counter << "\n"; 
            return counter != 0; 
        });

        std::cout << "msg: " << msg << "\n";

    }
    
    return hpx::local::finalize();
}

// the normal int main function that is called at startup and runs on an OS
// thread the user must call hpx::local::init to start the hpx runtime which
// will execute hpx_main on an hpx thread
int main(int argc, char* argv[])
{
    // if this test is run with distributed runtime, we need to make sure
    // that all ranks run their main function
    std::vector<std::string> cfg = {"hpx.run_hpx_main!=1"};

    // Init MPI
    int provided = MPI_THREAD_MULTIPLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE)
    {
        std::cout << "Provided MPI is not : MPI_THREAD_MULTIPLE " << provided
                  << std::endl;
    }


    // Initialize and run HPX.
    hpx::local::init_params init_args;
    init_args.cfg = cfg;

    auto result = hpx::local::init(hpx_main, argc, argv, init_args);

    // Finalize MPI
    MPI_Finalize();

    return result || hpx::util::report_errors();
}