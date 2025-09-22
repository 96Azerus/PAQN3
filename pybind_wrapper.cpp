#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <thread>
#include <atomic>
#include <vector>
#include <memory>
#include <iostream>
#include "cpp_src/DeepMCCFR.hpp"
#include "cpp_src/SharedReplayBuffer.hpp"
#include "cpp_src/constants.hpp"
#include "cpp_src/hand_evaluator.hpp"
#include "cpp_src/InferenceQueue.hpp"

namespace py = pybind11;

class SolverManagerImpl {
public:
    SolverManagerImpl(
        size_t num_workers,
        size_t action_limit,
        ofc::SharedReplayBuffer* policy_buffer, 
        ofc::SharedReplayBuffer* value_buffer,
        py::object request_queue,
        py::object result_queue,
        py::object log_queue
    ) : num_workers_(num_workers),
        action_limit_(action_limit),
        policy_buffer_(policy_buffer),
        value_buffer_(value_buffer),
        request_queue_(request_queue), 
        result_queue_(result_queue), 
        log_queue_(log_queue) 
    {
        stop_flag_.store(true);
    }

    ~SolverManagerImpl() {
        stop();
    }

    void start() {
        if (!stop_flag_.exchange(false)) {
            return;
        }
        for (size_t i = 0; i < num_workers_; ++i) {
            auto solver = std::make_unique<ofc::DeepMCCFR>(
                action_limit_, policy_buffer_, value_buffer_, &request_queue_, &result_queue_, &log_queue_
            );
            threads_.emplace_back(&SolverManagerImpl::worker_loop, this, std::move(solver));
        }
    }

    void stop() {
        if (!stop_flag_.exchange(true)) {
            for (auto& t : threads_) {
                if (t.joinable()) {
                    t.join();
                }
            }
            threads_.clear();
        }
    }

private:
    void worker_loop(std::unique_ptr<ofc::DeepMCCFR> solver) {
        while (!stop_flag_.load()) {
            solver->run_traversal();
        }
    }

    size_t num_workers_;
    size_t action_limit_;
    ofc::SharedReplayBuffer* policy_buffer_;
    ofc::SharedReplayBuffer* value_buffer_;

    std::vector<std::thread> threads_;
    std::atomic<bool> stop_flag_;
    py::object request_queue_;
    py::object result_queue_;
    py::object log_queue_;
};

class PySolverManager {
public:
    PySolverManager(
        size_t num_workers,
        size_t action_limit,
        ofc::SharedReplayBuffer* policy_buffer, 
        ofc::SharedReplayBuffer* value_buffer,
        py::object request_queue,
        py::object result_obj,
        py::object log_queue
    ) {
        impl_ = std::make_unique<SolverManagerImpl>(
            num_workers, action_limit, policy_buffer, value_buffer,
            request_queue, result_obj, log_queue
        );
    }

    ~PySolverManager() {
        py::gil_scoped_acquire acquire;
        if (impl_) {
            impl_->stop();
        }
    }

    void start() {
        py::gil_scoped_release release;
        if (impl_) {
            impl_->start();
        }
    }

    void stop() {
        py::gil_scoped_release release;
        if (impl_) {
            impl_->stop();
        }
    }

private:
    std::unique_ptr<SolverManagerImpl> impl_;
};


PYBIND11_MODULE(ofc_engine, m) {
    m.doc() = "OFC Engine with C++ Thread Manager and Queue-based Inference";

    m.def("initialize_evaluator", []() {
        omp::HandEvaluator::initialize();
    }, "Initializes the static hand evaluator lookup tables.", py::call_guard<py::gil_scoped_release>());

    py::class_<ofc::SharedReplayBuffer>(m, "ReplayBuffer")
        .def(py::init<uint64_t>(), py::arg("capacity"))
        .def("size", &ofc::SharedReplayBuffer::size)
        .def("total_generated", &ofc::SharedReplayBuffer::total_generated)
        .def("sample", [](ofc::SharedReplayBuffer &buffer, int batch_size) -> py::object {
            auto infosets_np = py::array_t<float>({batch_size, ofc::INFOSET_SIZE});
            auto actions_np  = py::array_t<float>({batch_size, ofc::ACTION_VECTOR_SIZE});
            auto targets_np  = py::array_t<float>({batch_size});

            auto infos_bi = infosets_np.request();
            auto act_bi   = actions_np.request();
            auto targ_bi  = targets_np.request();

            bool success = false;
            {
                py::gil_scoped_release release;
                success = buffer.sample_to_ptr(
                    batch_size,
                    static_cast<float*>(infos_bi.ptr),
                    static_cast<float*>(act_bi.ptr),
                    static_cast<float*>(targ_bi.ptr)
                );
            }

            if (!success) {
                return py::none();
            }
            
            return py::make_tuple(infosets_np, actions_np, targets_np);
        }, py::arg("batch_size"), "Samples a batch from the buffer.");
        
    py::class_<PySolverManager>(m, "SolverManager")
        .def(py::init<size_t, size_t, ofc::SharedReplayBuffer*, ofc::SharedReplayBuffer*, py::object, py::object, py::object>(),
             py::arg("num_workers"),
             py::arg("action_limit"),
             py::arg("policy_buffer"), py::arg("value_buffer"),
             py::arg("request_queue"), py::arg("result_queue"),
             py::arg("log_queue")
        )
        .def("start", &PySolverManager::start)
        .def("stop", &PySolverManager::stop);
}
