LOCK_SH = 1
LOCK_EX = 2
LOCK_NB = 4
LOCK_UN = 8

def flock(fd, operation):
    pass

def lockf(fd, operation, length=0, start=0, whence=0):
    pass

def fcntl(fd, cmd, arg=0):
    return 0

def ioctl(fd, request, arg=0, mutate_flag=True):
    return 0
