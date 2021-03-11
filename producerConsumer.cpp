#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <chrono>
#include <condition_variable>

// Thread work where we can pass gradients start, end and count
struct ThreadWork {
    int data1{0};
    int data2{0};
};

// Thread control structures to assign work to thread
struct ThreadControl {
    int threadId;
    std::mutex m;
    std::condition_variable cv;
    bool ready{false};
    ThreadWork work;
};

// Max workers
static constexpr int maxWorkers = 4;

// ProducerControl
struct ProducerControl {
    std::mutex m;
    bool processed{false};
    std::condition_variable cv;
    std::atomic<int> workCompleted;
    ThreadControl threadCntrl[maxWorkers];
    bool exitWorker{false};
};

void worker(ThreadControl *threadCntrl, ProducerControl *pCntrl)
{
    // Loop for ever
    while (1) {
        {
            // Wait until producer sends data
            std::unique_lock<std::mutex> lk(threadCntrl->m);
            threadCntrl->cv.wait(lk, [&]{return threadCntrl->ready;});

            // Do the work (TX followed by RX of ML packets)
            std::cout << "Worker thread is processing data - " << threadCntrl->threadId << std::endl;
            threadCntrl->work.data1++;
            threadCntrl->work.data2++;
            std::cout << "Worker thread is completed processing - " << threadCntrl->threadId << std::endl;

            // Once the work is completed by all the workers, inform producer.
            // If not, go back and wait for next work.
            auto workCount = pCntrl->workCompleted.fetch_add(1, std::memory_order_relaxed);
            std::cout << "Job number "<< workCount+1 << " completed" << std::endl;
            if (workCount == maxWorkers - 1) {
                std::unique_lock<std::mutex> lk(pCntrl->m);
                pCntrl->processed = true;
                lk.unlock();
                pCntrl->cv.notify_one();
                std::cout << "Completed work" << std::endl;
            }
            threadCntrl->ready = false;
            lk.unlock();

            // Check if we have to terminate the worker
            if (pCntrl->exitWorker) {
                return;
            }
        }
    }
}

int main()
{
    ProducerControl pCntrl;

    // Start the workers
    std::vector<std::thread> ThreadVector;
    std::cout << "Starting workers..." << std::endl;
    for(auto i = 0; i< maxWorkers; i++) {
        pCntrl.threadCntrl[i].threadId = i+1;
        ThreadVector.emplace_back([&]() { worker(&pCntrl.threadCntrl[i], &pCntrl); });
        // Wait for threads to start
        // This is shortcut instead of adding logic to verify whether threads are already started
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    std::cout << "Got callback from GPU..." << std::endl;
    // Got callback from GPU, so assign work to all the workers
    auto count = 8;
    while (count) {
        pCntrl.workCompleted.store(0);

        // Notify each worker thread work is available for processing
        for(auto i = 0; i< maxWorkers; i++) {
            {
                std::lock_guard<std::mutex> lk(pCntrl.threadCntrl[i].m);
                pCntrl.threadCntrl[i].ready = true;
            }
            pCntrl.threadCntrl[i].cv.notify_one();
        }

        // wait for the workers to complete the work
        {
            std::unique_lock<std::mutex> lk(pCntrl.m);
            pCntrl.cv.wait(lk, [&]{return pCntrl.processed;});
            pCntrl.processed = false;
        }

        // Inform GPU
        for(auto i = 0; i< maxWorkers; i++) {
            std::cout << "Worker - " << i+1 << "Data1 -" << pCntrl.threadCntrl[i].work.data1 <<
                " Data2 - " << pCntrl.threadCntrl[i].work.data2 << std::endl;
        }

        // Are we done with work?
        count--;
        if (count == 1) {
            pCntrl.exitWorker = true;
        }
    }

    // Stop all the threads
    std::cout << "Stopping threads..." << std::endl;
    for(auto i = 0; i< maxWorkers; i++) {
        ThreadVector[i].join();
    }
}
