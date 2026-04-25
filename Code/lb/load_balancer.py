# Load balancer using Round Robin

class LoadBalancer:
    def __init__(self, workers):
        self.workers = workers
        self.index = 0

    def get_next_worker(self):  # this method returns the next worker in a round-robin fashion
        worker = self.workers[self.index]
        self.index = (self.index + 1) % len(self.workers)
        return worker
    
    def dispatch(self, request): # this method dispatches/assigns the request to the next worker
        worker = self.get_next_worker()
        return worker.process(request)