
import time

def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))


def get_time_for_name_some():
    return time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))

