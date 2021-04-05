#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>


extern int HdmiInit();
extern int HdmiCapture(int name, int num);
extern int HdmiSave();

extern int SerialInit();
extern void SerialRead(int name);
extern void SerialClose();


//#define PRIORITY 1


int user_alarms, sig_alarms;

void signal_handler(int signum)
{
    switch (signum) {
        case SIGALRM:
            sig_alarms++;
            break;
    }
}


int main(int argc, char *argv[])
{
    if (HdmiInit()) return -1;
    if (SerialInit()) return -1;

    struct sigaction sa;
    struct itimerval tv;

#if PRIORITY
    pid_t pid = getpid();
    if (setpriority(PRIO_PROCESS, pid, -19))
        fprintf(stderr, "Warning: Failed to set priority: %s\n",
                strerror(errno));
#endif

    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    if (sigaction(SIGALRM, &sa, 0)) {
        fprintf(stderr, "Failed to install signal handler!\n");
        return -1;
    }

    tv.it_interval.tv_sec = 0;
    tv.it_interval.tv_usec = 100000;
    tv.it_value.tv_sec = 0;
    tv.it_value.tv_usec = 100000;
    if (setitimer(ITIMER_REAL, &tv, NULL)) {
        fprintf(stderr, "Failed to start timer: %s\n", strerror(errno));
        return 1;
    }

    int count = 0;
    while (1) {
        if (user_alarms != sig_alarms) {
            user_alarms = sig_alarms;

            SerialRead(user_alarms);
            if (HdmiCapture(user_alarms, count)) break;

            count++;
            printf(".");
            fflush(stdout);

        }

        pause();
    }

    printf("\n");
    HdmiSave();
    printf("\n");

    SerialClose();

}
