import time


class RecordTimer(object):

    def __init__(self):
        self.run = 0
        self.times = []
        self.marker = []
        self.common_time = [0]

    def __enter__(self):
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_run()

    def start_run(self):
        """
        Initialize new test run
        """
        self.start = time.time()
        self.mark = self.start

    def mark_run(self):
        """
        """
        while self.run >= len(self.marker):
            self.marker.append([])

        interim = time.time()
        self.marker[self.run].append(interim - self.mark)
        self.mark = interim

    def stop_run(self):
        """
        Stop test run and calculate time taken
        """
        end = time.time()
        if self.run < len(self.times):
            self.times[self.run] += end - self.start
        else:
            self.times.append(end - self.start)
        self.run += 1

    def reset_run(self):
        self.run = 0

    def start_common(self):
        self.start_run()

    def stop_common(self):
        end = time.time()
        self.common_time.append(end - self.start)

    def apply_common(self):
        if (len(self.common_time) > 0):
            part = sum(self.common_time) / len(self.times)
            self.times = [x + part for x in self.times]
            # Reset common time
            self.common_time = [0]
