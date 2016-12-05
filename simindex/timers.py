import time


class RecordTimer():

    def __init__(self):
        self.run = 0
        self.times = []

    def __enter__(self):
        """
        Initialize new test run
        """
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Stop test run and calculate time taken
        """
        end = time.time()
        self.times.append(end - self.start)
        self.run += 1
