import sys, os, time
from multiprocessing import *

from src.sumosim import SumoSim
from src.picklefuncs import save_data
from src.helper_funcs import check_and_make_dir, get_time_now, write_to_log
from src.controller_types import is_rl_controller

class SimProc(Process):
    def __init__(self, idx, args, barrier, netdata, rl_stats, exp_replays, eps, offset):
        Process.__init__(self)
        self.idx = idx
        self.args = args
        self.barrier = barrier
        self.netdata = netdata
        self.sim = SumoSim(args.cfg_fp, args.sim_len, args.tsc, args.nogui, netdata, args, idx)
        self.rl_stats = rl_stats
        self.exp_replays = exp_replays
        self.eps = eps
        self.offset = offset
        self.initial = True 

    def run(self):
        # DDPG uses TF1 graph mode — must disable eager before any TF ops
        if self.args.tsc == 'ddpg':
            import tensorflow as tf
            tf.compat.v1.disable_eager_execution()

        learner = False
        if self.args.load == True and self.args.mode == 'test':
            load = True
        else:
            load = False

        neural_networks = {}
        if is_rl_controller(self.args.tsc):
            from src.nn_factory import gen_neural_networks

            neural_networks = gen_neural_networks(self.args,
                                                  self.netdata,
                                                  self.args.tsc,
                                                  self.netdata['inter'].keys(),
                                                  learner,
                                                  load,
                                                  self.args.n_hidden)

        print('sim proc '+str(self.idx)+' waiting at barrier ---------')
        write_to_log(' ACTOR #'+str(self.idx)+' WAITING AT SYNC WEIGHTS BARRIER...')
        self.barrier.wait()
        write_to_log(' ACTOR #'+str(self.idx)+'  BROKEN SYNC BARRIER...')
        if self.args.l > 0 and self.args.mode == 'train':
            neural_networks = self.sync_nn_weights(neural_networks)
        #barrier
        #grab weights from learner or load from file
        #barrier

        if self.args.mode == 'train':
            if not is_rl_controller(self.args.tsc):
                self.run_sim(neural_networks)
                if (self.eps == 1.0 or self.eps < 0.02):
                    self.write_to_csv(self.sim.sim_stats())
                self.write_sim_tsc_metrics()
                #self.write_travel_times()
                self.sim.close()
            else:
                consecutive_failures = 0
                max_consecutive_failures = 20
                while not self.finished_updates():
                    try:
                        self.run_sim(neural_networks)
                        consecutive_failures = 0
                        if (self.eps == 1.0 or self.eps < 0.02):
                            self.write_to_csv(self.sim.sim_stats())
                        self.write_sim_tsc_metrics()
                        #self.write_travel_times()
                        self.sim.close()
                    except Exception as e:
                        consecutive_failures += 1
                        write_to_log(f' ACTOR #{self.idx} SIM FAILED ({consecutive_failures}/{max_consecutive_failures}): {e}')
                        print(f'Actor {self.idx} sim failed ({consecutive_failures}): {e}')
                        try:
                            self.sim.close()
                        except Exception:
                            pass
                        if consecutive_failures >= max_consecutive_failures:
                            write_to_log(f' ACTOR #{self.idx} TOO MANY FAILURES, EXITING')
                            raise
                        import time as _time
                        _time.sleep(5.0 * consecutive_failures)

        elif self.args.mode == 'test':
            print(str(self.idx)+' test  waiting at offset ------------- '+str(self.offset))
            print(str(self.idx)+' test broken offset =================== '+str(self.offset))
            self.initial = False
            #just run one sim for stats
            self.run_sim(neural_networks)
            if (self.eps == 1.0 or self.eps < 0.02) and self.args.mode == 'test':
                self.write_to_csv(self.sim.sim_stats())
                path = 'Outputs/results/' + str(self.args.tsc) + '/test/'
                check_and_make_dir(path)
                with open( path + str(self.eps)+'.csv','a+') as f:
                    f.write('-----------------\n')
            self.write_sim_tsc_metrics()
            #self.write_travel_times()
            self.sim.close()
        print('------------------\nFinished on sim process '+str(self.idx)+' Closing\n---------------')

    def run_sim(self, neural_networks):
        start_t = time.time()
        self.sim.gen_sim()

        if self.initial is True:
            #if the initial sim, run until the offset time reached
            self.initial = False
            self.sim.run_offset(self.offset)
            print(str(self.idx)+' train  waiting at offset ------------- '+str(self.offset)+' at '+str(get_time_now()))
            write_to_log(' ACTOR #'+str(self.idx)+' FINISHED RUNNING OFFSET '+str(self.offset)+' to time '+str(self.sim.t)+' , WAITING FOR OTHER OFFSETS...')
            self.barrier.wait()
            print(str(self.idx)+' train  broken offset =================== '+str(self.offset)+' at '+str(get_time_now()))
            write_to_log(' ACTOR #'+str(self.idx)+'  BROKEN OFFSET BARRIER...')

        self.sim.create_tsc(self.rl_stats, self.exp_replays, self.eps, neural_networks)
        write_to_log('ACTOR #'+str(self.idx)+'  START RUN SIM...')
        self.sim.run()
        print('sim finished in '+str(time.time()-start_t)+' on proc '+str(self.idx))
        write_to_log('ACTOR #'+str(self.idx)+'  FINISHED SIM...')

    def write_sim_tsc_metrics(self):
        #get data dict of all tsc in sim
        #where each tsc has dict of all metrics
        tsc_metrics =  self.sim.get_tsc_metrics()
        #create file name and path for writing metrics data
        fname = get_time_now()
        #write all metrics to correct path, separated by mode
        path = 'Outputs/metrics/'+str(self.args.tsc) + '/' + str(self.args.mode)
        for tsc in tsc_metrics:
            for m in tsc_metrics[tsc]:
                mpath = path + '/'+str(m)+'/'+str(tsc)+'/'
                check_and_make_dir(mpath)
                save_data(mpath+fname+'_'+str(self.eps)+'_.p', tsc_metrics[tsc][m])

        travel_times = self.sim.get_travel_times()
        path += '/traveltime/'
        check_and_make_dir(path)
        save_data(path+fname+'.p', travel_times)
        

            
    def write_to_csv(self, data):

        path = 'Outputs/results/' + str(self.args.tsc) + '/' + str(self.args.mode) + '/'
        check_and_make_dir(path)

        fname = path + 'summary.csv'

        # Add header only once
        file_exists = os.path.isfile(fname)

        with open(fname, 'a+') as f:

            if not file_exists:
                f.write('avg_travel_time,std_travel_time,avg_wait,avg_queue,avg_delay,throughput,avg_speed\n')

            f.write(','.join(data) + '\n')

    def finished_updates(self):
        for tsc in self.netdata['inter'].keys():
            print(tsc+'  exp replay size '+str(len(self.exp_replays[tsc])))
            print(tsc+'  updates '+str(self.rl_stats[tsc]['updates']))
            if self.rl_stats[tsc]['updates'] < self.args.updates:
                return False
        return True

    def sync_nn_weights(self, neural_networks):
        for nn in neural_networks:
            weights = self.rl_stats[nn]['online']
            if self.args.tsc in ('ddpg', 'ppo'):
                #sync actor weights
                neural_networks[nn]['actor'].set_weights(weights, 'online')
            elif self.args.tsc == 'dqn':
                neural_networks[nn].set_weights(weights, 'online')
            else:
                #raise not found exceptions
                assert 0, 'Supplied RL traffic signal controller '+str(self.args.tsc)+' does not exist.'
        return neural_networks

