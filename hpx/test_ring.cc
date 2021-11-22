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

using hpx::program_options::options_description;
using hpx::program_options::value;
using hpx::program_options::variables_map;

static bool output = true;



// this is called on an hpx thread after the runtime starts up
int hpx_main(hpx::program_options::variables_map& vm)
{
    int rank, size;
    //
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const std::uint64_t iterations = vm["iterations"].as<std::uint64_t>();


    {
        // this needs to scope all uses of hpx::mpi::experimental::executor
        hpx::mpi::experimental::enable_user_polling enable_polling;

        // Ring send/recv around N ranks
        // Rank 0      : Send then Recv
        // Rank 1->N-1 : Recv then Send

        hpx::mpi::experimental::executor exec(MPI_COMM_WORLD);

        // mpi chokes if we put too many messages into the system at once
        // we will use a limiting executor with N 'in flight' at once
        hpx::execution::experimental::limiting_executor<
            hpx::mpi::experimental::executor>
            limexec(exec, 32, 64, true);

        std::vector<int> tokens(iterations, -1);

        hpx::chrono::high_resolution_timer t;

        std::atomic<std::uint64_t> counter(iterations*2);
        std::cout << "rank: "<< rank << ", counter is: " << counter << "\n";
        for (std::uint64_t i = 0; (i != iterations); ++i)
        {
            tokens[i] = (rank == 0) ? 1 : -1;
            int rank_from = (size + rank - 1) % size;
            int rank_to = (rank + 1) % size;

            // all ranks pre-post a receive
            hpx::future<int> f_recv = hpx::async(
                limexec, MPI_Irecv, &tokens[i], 1, MPI_INT, rank_from, i);

            // when the recv completes,
            f_recv.then([=, &exec, &tokens, &counter](auto&&) {
                if (rank > 0)
                {
                    --counter;
                    // send the incremented token to the next rank
                    ++tokens[i];
                    hpx::future<int> f_send = hpx::async(
                        exec, MPI_Isend, &tokens[i], 1, MPI_INT, rank_to, i);
                    // when the send completes
                    f_send.then([=, &tokens, &counter](auto&&) {
                        // ranks > 0 are done when they have sent their token
                        std::cout << "rank = 1, counter: " << counter << "\n";
                        --counter;
                        std::cout << "rank = 1, after --counter: " << counter << "\n";
                    });
                }
                else
                {
                    // rank 0 is done when it receives its token
                    std::cout << "rank = 0, counter: " << counter << "\n";
                    --counter;
                    std::cout << "rank = 0, after --counter: " << counter << "\n";
                }
            });

            // rank 0 starts the process with a send
            if (rank == 0)
            {
                auto f_send = hpx::async(
                    limexec, MPI_Isend, &tokens[i], 1, MPI_INT, rank_to, i);
                f_send.then([=, &tokens, &counter](auto&&) {
                    --counter;
                });
            }
        }

        std::cout << "Reached end of test " << counter << std::endl;
        // Our simple counter should reach zero when all send/recv pairs are done
        hpx::mpi::experimental::wait([&]() { 
            std::cout << "wait, rank: " << rank << ", counter: " << counter << "\n"; 
            return counter != 0; 
        });

        if (rank == 0)
        {
            std::cout << "time " << t.elapsed() << std::endl;
        }

        // let the user polling go out of scope
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

    // Configure application-specific options.
    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");

    // clang-format off
    cmdline.add_options()(
        "iterations",
        value<std::uint64_t>()->default_value(1),
        "number of iterations to test")

        ("output", "display messages during test");
    // clang-format on

    // Initialize and run HPX.
    hpx::local::init_params init_args;
    init_args.desc_cmdline = cmdline;
    init_args.cfg = cfg;

    auto result = hpx::local::init(hpx_main, argc, argv, init_args);

    // Finalize MPI
    MPI_Finalize();

    return result || hpx::util::report_errors();
}