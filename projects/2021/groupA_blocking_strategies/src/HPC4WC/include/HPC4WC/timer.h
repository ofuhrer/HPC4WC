#pragma once

#ifdef WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#endif

#ifndef WIN32
#include <sys/time.h>
#endif

namespace HPC4WC {

/**
 * @brief Simple timer class.
 * 
 * Source: https://github.com/cmm-21/a2/blob/main/src/libs/utils/include/utils/timer.h
 * and slightly changed.
 * 
 * This is a simple timer class, that can be reset at any point, and can be used
 * to get the ellapsed time since the last reset. In windows, it will be
 * implemented using the methods QueryPerformanceFrequency() and
 * QueryPerformanceCounter(). In Linux, it can be implemented using the method
 * gettimeofday(). The timeEllapsed method will return the time, in seconds but
 * expressed as a double, since the timer was reset.
 */
class Timer {
private:
#ifdef WIN32
    // this is the start time of the timer, in milliseconds.
    long long int startTime;
    // this is the frequency of the performance counter
    long long int frequency;
    double countsPerMillisecond;
#else
    //! start time of reset
    struct timeval startTime;
#endif

public:
    /**
     * @brief Default constructor - resets the timer for the first time.
    */
    Timer();

    /**
     * @brief Default destructor - doesn't do much.
    */
    ~Timer() {}

    /**
     * @brief This method resets the starting time. All the time Ellapsed function calls will use the reset start time for their time evaluations.
    */
    void restart();

    /**
     * @brief This method returns the number of seconds that has ellapsed since the timer was reset.
    */
    double timeElapsed();

    void wait(double t);

#ifdef WIN32
#else
protected:
    /*! Subtract two timevals.
     *
     * @param result	the result
     * @param x			first timeval
     * @param y			second timeval
     * @return			1 if result is negative
     */
    int timeval_subtract(struct timeval *result, struct timeval *x, struct timeval *y) {
        /* Perform the carry for the later subtraction by updating y. */
        if (x->tv_usec < y->tv_usec) {
            int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
            y->tv_usec -= 1000000 * nsec;
            y->tv_sec += nsec;
        }
        if (x->tv_usec - y->tv_usec > 1000000) {
            int nsec = (x->tv_usec - y->tv_usec) / 1000000;
            y->tv_usec += 1000000 * nsec;
            y->tv_sec -= nsec;
        }

        /* Compute the time remaining to wait.
           tv_usec is certainly positive. */
        result->tv_sec = x->tv_sec - y->tv_sec;
        result->tv_usec = x->tv_usec - y->tv_usec;

        /* Return 1 if result is negative. */
        return x->tv_sec < y->tv_sec;
    }

#endif
};

}  // namespace HPC4WC