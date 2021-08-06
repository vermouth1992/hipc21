//
// Created by Chi Zhang on 8/5/21.
//

#include "scheduler.h"

ConstantScheduler::ConstantScheduler(double value) : m_value(value) {

}

double ConstantScheduler::value(int64_t t) {
    return m_value;
}

LinearSchedule::LinearSchedule(int64_t scheduleTimesteps, double finalP, double initialP)
        : schedule_timesteps(scheduleTimesteps),
          final_p(finalP),
          initial_p(initialP) {

}

double LinearSchedule::value(int64_t t) {
    double fraction = std::min((double) t / (double) schedule_timesteps, 1.0);
    return initial_p + fraction * (final_p - initial_p);
}
