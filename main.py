from Code.workers.gpu_worker import GPUWorker
from Code.lb.load_balancer import LoadBalancer
from Code.master.scheduler import Scheduler
from Code.client.load_generator import run_load_test

def main():
    workers = [GPUWorker(i) for i in range(4)]  # Simulate 4 GPU workers
    lb = LoadBalancer(workers)
    scheduler = Scheduler(lb)
    run_load_test(scheduler, num_users=1000)

if __name__ == "__main__":
    main()