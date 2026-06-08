import subprocess

import numpy as np

from src.sumo_setup import configure_sumo

configure_sumo()

import traci
from sumolib import checkBinary

from src.trafficsignalcontroller import TrafficSignalController
from src.tsc_factory import tsc_factory
from src.vehiclegen import VehicleGen
from src.helper_funcs import write_to_log

class SumoSim:
    def __init__(self, cfg_fp, sim_len, tsc, nogui, netdata, args, idx):
        self.cfg_fp = cfg_fp
        self.sim_len = sim_len
        self.tsc = tsc
        self.sumo_cmd = 'sumo' if nogui else 'sumo-gui' 
        self.netdata = netdata
        self.args = args
        self.idx = idx

    def gen_sim(self):
        import time as _time
        #create sim stuff and intersections
        #serverless_connect()
        #self.conn, self.sumo_process = self.server_connect()

        port = self.args.port+self.idx
        sumoBinary = checkBinary(self.sumo_cmd)
        self.sumo_process = subprocess.Popen([sumoBinary, "-c",
                                         self.cfg_fp, "--remote-port",
                                         str(port), "--no-warnings",
                                         "--no-step-log", "--random"],
                                         stdout=None, stderr=None)

        # Wait briefly for SUMO to start listening, then connect with retries
        _time.sleep(0.5)
        for _attempt in range(10):
            try:
                self.conn = traci.connect(port)
                break
            except Exception:
                if self.sumo_process.poll() is not None:
                    raise RuntimeError(f"SUMO process died on port {port} (exit code {self.sumo_process.returncode})")
                _time.sleep(1.0)
        else:
            self.sumo_process.terminate()
            raise RuntimeError(f"Could not connect to SUMO on port {port} after 10 retries")

        self.t = 0
        self.v_start_times = {}
        self.v_travel_times = {}
        self.wait_times = []
        self.queue_lengths = []
        self.delays = []
        self.throughputs = []
        self.avg_speeds = []
        self.vehiclegen = None
        if self.args.sim == 'double' or self.args.sim == 'single':
            self.vehiclegen = VehicleGen(self.netdata, 
                                         self.args.sim_len, 
                                         self.args.demand, 
                                         self.args.scale,
                                         self.args.mode, self.conn) 

    def serverless_connect(self):
        traci.start([self.sumo_cmd, 
                     "-c", self.cfg_fp, 
                     "--no-step-log", 
                     "--no-warnings",
                     "--random"])

    def server_connect(self):
        sumoBinary = checkBinary(self.sumo_cmd)
        port = self.args.port+self.idx
        sumo_process = subprocess.Popen([sumoBinary, "-c",
                                         self.cfg_fp, "--remote-port",      
                                         str(port), "--no-warnings",
                                         "--no-step-log", "--random"],
                                         stdout=None, stderr=None)

        return traci.connect(port), sumo_process

    def get_traffic_lights(self):
        #find all the junctions with traffic lights
        trafficlights = self.conn.trafficlight.getIDList()
        junctions = self.conn.junction.getIDList()

        tl_juncs = set(trafficlights).intersection( set(junctions) )
        tls = []
     
        #only keep traffic lights with more than 1 green phase
        for tl in tl_juncs:
            #subscription to get traffic light phases
            self.conn.trafficlight.subscribe(tl, [traci.constants.TL_COMPLETE_DEFINITION_RYG])
            tldata = self.conn.trafficlight.getAllSubscriptionResults()
            logic = tldata[tl][traci.constants.TL_COMPLETE_DEFINITION_RYG][0]

            #for some reason this throws errors for me in SUMO 1.2
            #have to do subscription based above
            '''
            logic = self.conn.trafficlight.getCompleteRedYellowGreenDefinition(tl)[0] 
            '''
            #get only the green phases
            green_phases = [ p.state for p in logic.getPhases()
                             if 'y' not in p.state
                             and ('G' in p.state or 'g' in p.state) ]
            if len(green_phases) > 1:
                tls.append(tl)

        return set(tls) 


    def create_tsc(self, rl_stats, exp_replays, eps, neural_networks = None):
        self.tl_junc = self.get_traffic_lights() 
        if not neural_networks:
            neural_networks = {tl:None for tl in self.tl_junc}
        #create traffic signal controllers for the junctions with lights
        self.tsc = { tl:tsc_factory(self.args.tsc, tl, self.args, self.netdata, rl_stats[tl], exp_replays[tl], neural_networks[tl], eps, self.conn)  
                     for tl in self.tl_junc }

    def update_netdata(self):
        tl_junc = self.get_traffic_lights()
        tsc = { tl:TrafficSignalController(self.conn, tl, self.args.mode, self.netdata, 2, 3)  
                     for tl in tl_junc }

        for t in tsc:
            self.netdata['inter'][t]['incoming_lanes'] = tsc[t].incoming_lanes
            self.netdata['inter'][t]['green_phases'] = tsc[t].green_phases

        all_intersections = set(self.netdata['inter'].keys())
        #only keep intersections that we want to control
        for i in all_intersections - tl_junc:
            del self.netdata['inter'][i]

        return self.netdata

    def sim_step(self):
        self.conn.simulationStep()
        self.t += 1

    def run_offset(self, offset):
        while self.t < offset:
            #create vehicles if vehiclegen class exists
            if self.vehiclegen:
                self.vehiclegen.run()
            self.update_travel_times()
            self.sim_step()

    def run(self):
        import time as _time
        #execute simulation for desired length
        # Max time for entire sim (sim_len steps should take < 10 min normally)
        sim_deadline = _time.time() + max(600, self.sim_len * 0.5)
        while self.t < self.sim_len:

            if _time.time() > sim_deadline:
                raise RuntimeError(f"Simulation timeout: step {self.t}/{self.sim_len} exceeded deadline")

            if self.vehiclegen:
                self.vehiclegen.run()

            self.update_travel_times()

            total_wait = 0
            total_queue = 0
            total_delay = 0
            total_throughput = 0
            total_speed = 0
            speed_count = 0

            # run all traffic signal controllers
            for t in self.tsc:

                self.tsc[t].run()

                metrics = self.tsc[t].trafficmetrics

                # Waiting Time
                if 'waitingtime' in metrics.metrics:
                    total_wait += metrics.get_metric('waitingtime')

                # Queue Length
                if 'queue' in metrics.metrics:
                    total_queue += metrics.get_metric('queue')

                # Delay
                if 'delay' in metrics.metrics:
                    total_delay += metrics.get_metric('delay')

                # Throughput
                if 'throughput' in metrics.metrics:
                    total_throughput += metrics.get_metric('throughput')

                # Speed
                if 'speed' in metrics.metrics:
                    total_speed += metrics.get_metric('speed')
                    speed_count += 1

            # Store history
            self.wait_times.append(total_wait)
            self.queue_lengths.append(total_queue)
            self.delays.append(total_delay)
            self.throughputs.append(total_throughput)

            avg_speed = total_speed / speed_count if speed_count > 0 else 0
            self.avg_speeds.append(avg_speed)

            self.sim_step()

    def update_travel_times(self):
        for v in self.conn.simulation.getDepartedIDList():
            self.v_start_times[v] = self.t

        for v in self.conn.simulation.getArrivedIDList():
            self.v_travel_times[v] = self.t - self.v_start_times[v]
            del self.v_start_times[v]

    def get_intersection_subscription(self):
        tl_data = {}
        lane_vehicles = { l:{} for l in self.lanes}
        for tl in self.tl_junc:
            tl_data[tl] = self.conn.junction.getContextSubscriptionResults(tl)
            if tl_data[tl] is not None:
                for v in tl_data[tl]:
                    lane_vehicles[ tl_data[tl][v][traci.constants.VAR_LANE_ID] ][v] = tl_data[tl][v]
        return lane_vehicles

    def sim_stats(self):

        tt = self.get_travel_times()

        avg_tt = np.mean(tt) if len(tt) > 0 else 0
        std_tt = np.std(tt) if len(tt) > 0 else 0

        avg_wait = np.mean(self.wait_times) if len(self.wait_times) > 0 else 0

        avg_queue = np.mean(self.queue_lengths) if len(self.queue_lengths) > 0 else 0

        avg_delay = np.mean(self.delays) if len(self.delays) > 0 else 0

        throughput = np.mean(self.throughputs) if len(self.throughputs) > 0 else 0
        
        avg_speed = np.mean(self.avg_speeds) if len(self.avg_speeds) > 0 else 0

        return [
            str(round(avg_tt,2)),
            str(round(std_tt,2)),
            str(round(avg_wait,2)),
            str(round(avg_queue,2)),
            str(round(avg_delay,2)),
            str(round(throughput,2)),
            str(round(avg_speed,2))
        ]

    def get_travel_times(self):
        return [self.v_travel_times[v] for v in self.v_travel_times]

    def get_tsc_metrics(self):
        tsc_metrics = {}
        for tsc in self.tsc:
            tsc_metrics[tsc] = self.tsc[tsc].get_traffic_metrics_history()
        return tsc_metrics

    def close(self):
        #self.conn.close()
        self.conn.close()
        self.sumo_process.terminate()
