from threading import Thread

from eurelis_llmatoolkit.api.misc.singleton import Singleton


class AbstractWorker(Thread):
    def __init__(self, logger):
        Thread.__init__(self)
        self._logger = logger

    def run(self):
        pass


class WorkerManager(metaclass=Singleton):
    def execute(self, worker: AbstractWorker):
        worker.start()
