#ifndef __THREAD_POOL_H__
#define __THREAD_POOL_H__
 
#include <atomic>
#include <functional>
#include <iostream>
#include <thread>


// Thread pool of size N
template <size_t N>
class ThreadPool
{
public:

    ThreadPool():   
        thread_done_flags_({})
    {
        //initialize threads
        for(size_t i = 0; i < N; i++)
        {
            threads_[i] = std::thread(&ThreadPool::thread_loop, std::ref(*this), i);
        }
    }

    ~ThreadPool()
    {
        alive_ = false;
        for(size_t i = 0; i < N; i++)
        {
            threads_[i].join();
        }
    }

    void set_function(const std::function<void(size_t thread_index, size_t num_threads)>& func)
    {
        func_ = func;
    }

    // Start tasks and return when all are complete
    void execute()
    {
        if(!func_)
        {
            // There is no task!
            return;
        }

        // No threads are done
        clear_done_flags();

        // This will kick off the worker threads, which continuously check this in their loops
        ready_ = true;

        // wait for all threads to finish
        while(!check_all_done_flags())
        {}

        // All threads are now done, set ready_ to false
        ready_ = false;
    }

private:
    // Thread will continuously run and execute tasks when prompted
    // Prompt is triggered by setting ready_ = true;
    void thread_loop(const size_t thread_index)
    {
        while(alive_)
        {
            if(ready_ && func_ && !thread_done_flags_[thread_index])
            {
                func_(thread_index, N);
                // guarantee that the loop executes at most once
                thread_done_flags_[thread_index] = true;
            }
        }
    }

    void clear_done_flags()
    {
        std::memset(thread_done_flags_, 0, N);
    }

    bool check_all_done_flags()
    {
        for(size_t i = 0; i < N; i++)
        {
            if(!thread_done_flags_[i])
            {
                return false;
            }
        }

        return true;
    }

    // Thread pool is alive - set false upon destruction
    std::atomic<bool> alive_ = true;

    // Flag to tell threads to start executing the task
    std::atomic<bool> ready_ = false;

    // Function for each thread to execute
    std::function<void(size_t thread_index, size_t num_threads)> func_ = nullptr;

    // Running threads
    std::thread threads_[N];

    // Flags set by each thread to signal that they are done
    std::atomic<bool> thread_done_flags_[N];
};


#endif