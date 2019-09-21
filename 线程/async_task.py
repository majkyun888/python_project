# -*- encoding=utf-8 -*-


import uuid
import threading
import psutil, time


class ThreadSafeQueueException(Exception):
    pass


class TaskTypeErrorException(Exception):
    pass


class ThreadSafeQueue(object):

    def __init__(self, max_size = 0):
        self.queue = []
        self.max_size = max_size
        self.lock = threading.Lock()
        self.condition = threading.Condition()

    def size(self):
        self.lock.acquire()
        size = len(self.queue)
        self.lock.release()
        return size

    def put(self, item):
        if self.max_size != 0 and self.size() > self.max_size:
            return ThreadSafeQueueException
        self.lock.acquire()
        self.queue.append(item)
        self.lock.release()
        self.condition.acquire()
        self.condition.notify()
        self.condition.release()

    def put_bach(self, item_list):
        if not isinstance(item_list, list):
            item_list = list(item_list)
        for item in item_list:
            self.put(item)

    def pop(self, block = True, timeout=0):
        if self.size() == 0:
            if block:
                self.condition.acquire()
                self.condition.wait(timeout=timeout)
                self.condition.release()
            else:
                return None
        self.lock.acquire()
        item = None
        if len(self.queue) > 0:
            item = self.queue.pop()
        self.lock.release()
        return item

    def get(self, index):
        self.lock.acquire()
        item = self.queue[index]
        self.lock.release()
        return item


class Task:

    def __init__(self, func, *args, **kwargs):
        self.callable = func
        self.id = uuid.uuid4()
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return 'Task id: ' + str(self.id)


class AsyncTask(Task):

    def __init__(self, func, *args, **kwargs):
        self.result = None
        self.condition = threading.Condition()
        super().__init__(func, *args, **kwargs)

    def set_result(self, result):
        self.condition.acquire()
        self.result = result
        self.condition.notify()
        self.condition.release()

    def get_result(self):
        self.condition.acquire()
        if not self.result:
            self.condition.wait()
        result = self.result
        self.condition.release()
        return result


class SimpleTask(Task):
    def __init__(self, callable):
        super().__init__(callable)


class ProcessThread(threading.Thread):

    def __init__(self, task_queue, *args, **kwargs):
        super(ProcessThread, self).__init__(*args, *kwargs)
        self.dismiss_flag = threading.Event()
        self.task_queue = task_queue
        self.args = args
        self.kwargs = kwargs

    def run(self):
        while True:
            if self.dismiss_flag.is_set():
                break

            task = self.task_queue.pop()
            if not isinstance(task, Task):
                continue

            result = task.callable(*task.args, *task.kwargs)
            if isinstance(task, AsyncTask):
                task.set_result(result)

    def __dismiss(self):
        self.dismiss_flag.set()

    def stop(self):
        self.__dismiss()


class ThreadPool:

    def __init__(self, size = 0):
        if not size:
            size = psutil.cpu_count() * 2
        self.pool = ThreadSafeQueue(size)
        self.task_queue = ThreadSafeQueue()

        for i in range(size):
            self.pool.put(ProcessThread(self.task_queue))

    def start(self):
        for i in range(self.pool.size()):
            thread = self.pool.get(i)
            thread.start()

    def join(self):
        for i in range(self.pool.size()):
            thread = self.pool.get(i)
            thread.stop()
        while self.pool.size():
            thread = self.pool.pop()
            thread.join()

    def put(self, item):
        if not isinstance(item, Task):
            raise TaskTypeErrorException()
        self.task_queue.put(item)

    def bath_put(self, item_list):
        if not isinstance(item_list, list):
            item_list = list(item_list)
        for item in item_list:
            self.put(item)

    def size(self):
        return self.pool.size()


def my_function():
    print("this is a task test.")


def process():
    time.sleep(1)
    print('This is a SimpleTask callable function 1.')
    time.sleep(1)
    print('This is a SimpleTask callable function 2.')


def test():
    test_pool = ThreadPool()
    test_pool.start()
    for i in range(1000):
        simple_task = SimpleTask(process)
        test_pool.put(simple_task)


def test_async_task():

    def async_process():
        num = 0
        for i in range(100):
            num += i
        return num

    test_pool = ThreadPool()
    test_pool.start()

    for i in range(100):
        async_task = AsyncTask(func=async_process)
        test_pool.put(async_task)
        result = async_task.get_result()
        print('Get result: %s' % result)


def test_async_task2():

    def async_process():
        num = 0
        for i in range(100):
            num += i
        time.sleep(5)
        return num

    test_pool = ThreadPool()
    test_pool.start()

    for i in range(1):
        async_task = AsyncTask(func=async_process)
        test_pool.put(async_task)
        print('get result in timestamp: %d' % time.time())
        result = async_task.get_result()
        print('Get result in timestamp: %d: %d' % (time.time(), result))


def test_async_task3():

    def async_process():
        num = 0
        for i in range(100):
            num += i
        return num

    test_pool = ThreadPool()
    test_pool.start()

    for i in range(1):
        async_task = AsyncTask(func=async_process)
        test_pool.put(async_task)
        print('get result in timestamp: %d' % time.time())
        result = async_task.get_result()
        print('Get result in timestamp: %d: %d' % (time.time(), result))


if __name__ == '__main__':
    pass

