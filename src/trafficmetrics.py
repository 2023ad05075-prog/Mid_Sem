from src.sumo_setup import configure_sumo

configure_sumo()
from sumolib import checkBinary

import traci

class TrafficMetrics:
    def __init__(self, _id, incoming_lanes, netdata, metric_args, mode, conn):
        self.metrics = {}
        if 'delay' in metric_args:
            lane_lengths = {lane:netdata['lane'][lane]['length'] for lane in incoming_lanes}
            lane_speeds = {lane:netdata['lane'][lane]['speed'] for lane in incoming_lanes}
            self.metrics['delay'] = DelayMetric(_id, incoming_lanes, mode, lane_lengths, lane_speeds )

        if 'queue' in metric_args:
            self.metrics['queue'] = QueueMetric(_id, incoming_lanes, mode)

        if 'emission' in metric_args:
            self.metrics['emission'] = EmissionMetric(_id, incoming_lanes, mode, conn)
            
        if 'waitingtime' in metric_args:
            self.metrics['waitingtime'] = WaitingTimeMetric(_id, incoming_lanes, mode, conn)

        if 'throughput' in metric_args:
            self.metrics['throughput'] = ThroughputMetric(_id, incoming_lanes, mode)

        if 'speed' in metric_args:
            self.metrics['speed'] = SpeedMetric(_id, incoming_lanes, mode)

    def update(self, v_data):
        for m in self.metrics:
            self.metrics[m].update(v_data)

    def get_metric(self, metric):
        return self.metrics[metric].get_metric()

    def get_history(self, metric):
        return self.metrics[metric].get_history()

class TrafficMetric:
    def __init__(self, _id, incoming_lanes, mode):
        self.id = _id
        self.incoming_lanes = incoming_lanes
        self.history = []
        self.mode = mode

    def get_metric(self):
        pass

    def update(self):
        pass

    def get_history(self):
        return self.history

class DelayMetric(TrafficMetric):
    def __init__(self, _id, incoming_lanes, mode, lane_lengths, lane_speeds):
        super().__init__( _id, incoming_lanes, mode)
        self.lane_travel_times = {lane:lane_lengths[lane]/float(lane_speeds[lane]) for lane in incoming_lanes}
        self.old_v = set()
        self.v_info = {}
        self.t = 0

    def get_v_delay(self, v):
        return ( self.t - self.v_info[v]['t'] ) - self.lane_travel_times[self.v_info[v]['lane']]

    def get_metric(self):
        #calculate delay of vehicles on incoming lanes
        delay = 0
        for v in self.old_v:
            #calculate individual vehicle delay
            v_delay = self.get_v_delay(v)
            if v_delay > 0:
                delay += v_delay

        return delay

    def update(self, v_data):
        new_v = set()

        #record start time and lane of new_vehicles
        for lane in self.incoming_lanes:
            for v in v_data[lane]:
                if v not in self.old_v:
                    self.v_info[v] = {}
                    self.v_info[v]['t'] = self.t
                    self.v_info[v]['lane'] = lane
            new_v.update( set(v_data[lane].keys()) )

        self.history.append(self.get_metric())

        #remove vehicles that have left incoming lanes
        remove_vehicles = self.old_v - new_v
        delay = 0
        for v in remove_vehicles:
            del self.v_info[v]
        
        self.old_v = new_v
        self.t += 1

class QueueMetric(TrafficMetric):
    def __init__(self, _id, incoming_lanes, mode):
        super().__init__( _id, incoming_lanes, mode)
        self.stop_speed = 0.3
        self.lane_queues = {lane:0 for lane in self.incoming_lanes}

    def get_metric(self):
        return sum([self.lane_queues[lane] for lane in self.lane_queues])

    def update(self, v_data):
        lane_queues = {}
        for lane in self.incoming_lanes:
            lane_queues[lane] = 0
            for v in v_data[lane]:
                if v_data[lane][v][traci.constants.VAR_SPEED] < self.stop_speed:
                    lane_queues[lane] += 1

        self.lane_queues = lane_queues
        self.history.append(self.get_metric())

class EmissionMetric(TrafficMetric):
    """Tracks CO2 emissions and fuel consumption via SUMO TraCI."""
    def __init__(self, _id, incoming_lanes, mode, conn):
        super().__init__(_id, incoming_lanes, mode)
        self.conn = conn
        self.total_co2 = 0.0
        self.total_fuel = 0.0

    def get_metric(self):
        return {'co2': self.total_co2, 'fuel': self.total_fuel}

    def update(self, v_data):
        co2_step = 0.0
        fuel_step = 0.0
        for lane in self.incoming_lanes:
            for v in v_data[lane]:
                try:
                    co2_step += self.conn.vehicle.getCO2Emission(v)
                    fuel_step += self.conn.vehicle.getFuelConsumption(v)
                except Exception:
                    pass
        self.total_co2 += co2_step
        self.total_fuel += fuel_step
        self.history.append({'co2': co2_step, 'fuel': fuel_step})
            
class WaitingTimeMetric(TrafficMetric):
    def __init__(self, _id, incoming_lanes, mode, conn):
        super().__init__(_id, incoming_lanes, mode)
        self.conn = conn

    def get_metric(self):
        return self.history[-1] if self.history else 0

    def update(self, v_data):
        total_wait = 0

        for lane in self.incoming_lanes:
            try:
                veh_ids = self.conn.lane.getLastStepVehicleIDs(lane)
            except Exception:
                continue

            for veh in veh_ids:
                try:
                    total_wait += self.conn.vehicle.getWaitingTime(veh)
                except Exception:
                    pass

        self.history.append(total_wait)

class ThroughputMetric(TrafficMetric):
    def __init__(self, _id, incoming_lanes, mode):
        super().__init__(_id, incoming_lanes, mode)
        self.prev_vehicles = set()

    def get_metric(self):
        return self.history[-1] if self.history else 0

    def update(self, v_data):

        current_vehicles = set()

        for lane in self.incoming_lanes:
            current_vehicles.update(set(v_data[lane].keys()))

        exited = self.prev_vehicles - current_vehicles

        throughput = len(exited)

        self.prev_vehicles = current_vehicles

        self.history.append(throughput)
    
class SpeedMetric(TrafficMetric):
    def __init__(self, _id, incoming_lanes, mode):
        super().__init__(_id, incoming_lanes, mode)

    def get_metric(self):
        return self.history[-1] if self.history else 0

    def update(self, v_data):
        speeds = []

        for lane in self.incoming_lanes:
            for veh in v_data[lane]:
                speeds.append(
                    v_data[lane][veh][traci.constants.VAR_SPEED]
                )

        avg_speed = sum(speeds)/len(speeds) if len(speeds) > 0 else 0

        self.history.append(avg_speed)