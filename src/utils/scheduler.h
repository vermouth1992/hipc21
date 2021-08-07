//
// Created by Chi Zhang on 8/5/21.
//

#ifndef HIPC21_SCHEDULER_H
#define HIPC21_SCHEDULER_H


#include <cstdint>

class Scheduler {
    virtual double value(int64_t t) = 0;
};

class ConstantScheduler : public Scheduler {
public:
    explicit ConstantScheduler(double value);

    double value(int64_t t) override;

private:
    double m_value;
};


class LinearSchedule : public Scheduler {
public:
    explicit LinearSchedule(int64_t scheduleTimesteps, double finalP, double initialP);

    double value(int64_t t) override;

private:
    int64_t schedule_timesteps;
    double final_p;
    double initial_p;
};


class PiecewiseSchedule : public Scheduler {
    // TODO
};


#endif //HIPC21_SCHEDULER_H
