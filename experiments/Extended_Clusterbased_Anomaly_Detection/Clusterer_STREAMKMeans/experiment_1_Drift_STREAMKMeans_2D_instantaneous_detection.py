from mlpro.bf.streams.streams import *
from mlpro.bf.various import Log
from mlpro.oa.streams import *
from mlpro_int_river.wrappers.clusteranalyzers import WrRiverStreamKMeans2MLPro
from mlpro.oa.streams.tasks.anomalydetectors.cb_detectors.drift_detector import ClusterDriftDetector
import csv





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
# 1 Prepare a scenario
class MyScenario(OAScenario):
    C_NAME = 'ClusterDriftScenario'

    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging):

        # 1.1 Get MLPro benchmark Clutser Generator
        stream = StreamMLProClusterGenerator(p_num_dim=2,
                                             p_num_instances=5000,
                                             p_num_clusters=3,
                                             p_radii=[100],
                                             p_change_velocities=True,
                                             p_changed_velocities=[0.5, 0.6],
                                             p_points_of_change_velocities=[1000, 1200],
                                             p_num_clusters_for_change_velocities=2,
                                             p_seed=11,
                                             p_logging=p_logging)


        # 1.2 Set up a stream workflow

        # 1.2.1 Creation of a workflow
        workflow = OAWorkflow( p_name='Anomaly Detection',
                               p_range_max=OAWorkflow.C_RANGE_NONE,
                               p_ada=p_ada,
                               p_visualize=p_visualize,
                               p_logging=p_logging )


        # 1.2.2 Creation of tasks and add them to the workflow

        # Cluster Analyzer
        task_clusterer = WrRiverStreamKMeans2MLPro( p_name='StreamKMeans@River',
                                                   p_chunk_size=50,
                                                   p_n_clusters=3,
                                                   p_halflife=0.5,
                                                   p_sigma=5,
                                                   p_seed=44,
                                                   p_visualize=p_visualize,
                                                   p_logging=p_logging )

        workflow.add_task(p_task = task_clusterer)

        # Anomaly Detector
        task_anomaly_detector = ClusterDriftDetector(p_clusterer=task_clusterer,
                                                     p_with_time_calculation=False,
                                                     p_instantaneous_velocity_change_detection=True,
                                                     p_min_velocity_threshold=0.1,
                                                     p_initial_skip=400,
                                                     p_visualize=p_visualize,
                                                     p_logging=p_logging)

        workflow.add_task(p_task=task_anomaly_detector, p_pred_tasks=[task_clusterer])

        # 1.3 Return stream and workflow
        return stream, workflow



# 2 Prepare for test
cycle_limit = 2000
#logging     = Log.C_LOG_NOTHING
logging     = Log.C_LOG_ALL
visualize   = True
step_rate   = 1

# 3 Instantiate the stream scenario
myscenario = MyScenario( p_mode=Mode.C_MODE_SIM,
                               p_cycle_limit=cycle_limit,
                               p_visualize=visualize,
                               p_logging=logging )

# 4 Reset and run own stream scenario
myscenario.reset()

myscenario.init_plot( p_plot_settings=PlotSettings( p_view = PlotSettings.C_VIEW_2D,
                                                        p_step_rate = step_rate ) )
input('\nPlease arrange all windows and press ENTER to start stream processing...')


tp_before           = datetime.now()
myscenario.run()
tp_after            = datetime.now()
tp_delta            = tp_after - tp_before
duraction_sec       = ( tp_delta.seconds * 1000000 + tp_delta.microseconds + 1 ) / 1000000
myscenario.log(Log.C_LOG_TYPE_W, 'Duration [sec]:', round(duraction_sec,2), ', Cycles/sec:', round(cycle_limit/duraction_sec,2))


# 5 Summary
anomalies         = myscenario.get_workflow()._tasks[1].get_anomalies()
detected_anomalies= len(anomalies)

myscenario.log(Log.C_LOG_TYPE_W, '-------------------------------------------------------')
myscenario.log(Log.C_LOG_TYPE_W, '-------------------------------------------------------')
myscenario.log(Log.C_LOG_TYPE_W, 'Here is the recap of the anomaly detector')
myscenario.log(Log.C_LOG_TYPE_W, 'Number of anomalies: ', detected_anomalies )

for anomaly in anomalies.values():
     anomaly_name = anomaly.C_NAME
     anomaly_id = str(anomaly.id)
     clusters_affected = {}
     clusters = anomaly.get_clusters()
     properties = anomaly.get_properties()
     for x in clusters.keys():
        clusters_affected[x] = {}
        clusters_affected[x]["centroid"] = list(clusters[x].centroid.value)
        clusters_affected[x]["size"] = clusters[x].size.value
        clusters_affected[x]["velocity"] = properties[x]["velocity"]
        clusters_affected[x]["acceleration"] = properties[x]["acceleration"]

     
     inst = anomaly.get_instances()[-1].get_id()
     myscenario.log(Log.C_LOG_TYPE_W, 
                    'Anomaly : ', anomaly_name,
                    '\n Anomaly ID : ', anomaly_id,
                    '\n Instance ID : ', inst,
                    '\n Clusters : ', clusters_affected)

myscenario.log(Log.C_LOG_TYPE_W, '-------------------------------------------------------')
myscenario.log(Log.C_LOG_TYPE_W, '-------------------------------------------------------')
myscenario.log(Log.C_LOG_TYPE_W, 'Duration [sec]:', round(duraction_sec,2), ', Cycles/sec:', round(cycle_limit/duraction_sec,2))

with open('streamkmeans_2d_i.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['Index', 'Anomaly Type', 'Instance', 'Clusters'])

    for anomaly in anomalies.values():
        anomaly_name = anomaly.C_NAME
        anomaly_id = anomaly.id
        inst_id = anomaly.get_instances()[-1].get_id()
        clusters_affected = {}
        clusters = anomaly.get_clusters()
        properties = anomaly.get_properties()
        for x in clusters.keys():
            clusters_affected[x] = {}
            clusters_affected[x]["centroid"] = [round(value, 3) for value in list(clusters[x].centroid.value)]
            clusters_affected[x]["velocity"] = round(properties[x]["velocity"],3)
            clusters_affected[x]["acceleration"] = round(properties[x]["acceleration"],3)

        writer.writerow([anomaly_id, anomaly_name, inst_id, clusters_affected])


input('Press ENTER to exit...')
