from datetime import datetime
from mlpro.bf.streams.streams import *
from mlpro.bf.various import *
from mlpro.bf.data import *
from pathlib import Path
from mlpro.bf.streams.tasks import Rearranger
from mlpro.oa.streams import *
from sparccstream import *
from mlpro.oa.streams.tasks.anomalydetectors.cb_detectors.drift_detector import ClusterDriftDetector
from mlpro_int_river.wrappers.clusteranalyzers.kmeans import WrRiverKMeans2MLPro
import csv





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyScenario(OAScenario):

    C_NAME = 'NetworkTrafficMonitoringExperiment'

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging):

        # 1 Prepare the stream from CSV
        current_file_path = Path(__file__)
        data_directory = current_file_path.parent.parent.parent.parent / "datasets"
        path = str(data_directory)
        
        # 2 Instantiate Stream
        stream = StreamMLProCSV(p_logging=p_logging,
                            p_path_load=path,
                            p_csv_filename="UNSW.csv",
                            p_delimiter=",",
                            p_frame=False,
                            p_header=True,
                            p_list_features=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16', 'PC17', 'PC18', 'PC19', 'PC20', 'PC21', 'PC22', 'PC23'],
                            p_list_labels=['Label'])

        tp_start = datetime.now()
        myiterator = iter(stream)
        print('Features:', stream.get_feature_space().get_num_dim(),
              ', Labels:', stream.get_label_space().get_num_dim(),
              ', Instances:', stream.get_num_instances() )
        for i, curr_instance in enumerate(myiterator):
            curr_data = curr_instance.get_feature_data().get_values()


        # 3.1 Creation of a workflow
        workflow = OAWorkflow(p_name='Input Signal',
                              p_range_max=OAWorkflow.C_RANGE_NONE,
                              p_ada=p_ada,
                              p_visualize=p_visualize, 
                              p_logging=p_logging)
                              

        # 3.2 Rearranger to reduce the number of features
        features = stream.get_feature_space().get_dims()
        features_new = [('F', features[1:3])]      
        #features_new = [('F', [features[3], features[4], features[27], features[29]])]

        task_rearranger = Rearranger(p_name='T1 - Rearranger',
                                     p_range_max=Task.C_RANGE_NONE,
                                     p_visualize=p_visualize,
                                     p_logging=p_logging,
                                     p_features_new=features_new)

        workflow.add_task(p_task=task_rearranger)

        # Cluster Analyzer
        task_clusterer = WrRiverKMeans2MLPro( p_name='#1: KMeans@River',
                                              p_n_clusters=2,
                                              p_halflife=0.05, 
                                              p_sigma=3, 
                                              p_seed=42,
                                              p_visualize=p_visualize,
                                              p_logging=p_logging )


        workflow.add_task(p_task = task_clusterer, p_pred_tasks=[task_rearranger])

        # Anomaly Detector
        task_anomaly_detector = ClusterDriftDetector(p_clusterer=task_clusterer,
                                                     p_with_time_calculation=False,
                                                     p_instantaneous_velocity_change_detection=True,
                                                     p_min_velocity_threshold=0.05,
                                                     p_initial_skip=1,
                                                     p_visualize=p_visualize,
                                                     p_logging=p_logging)


        workflow.add_task(p_task=task_anomaly_detector, p_pred_tasks=[task_clusterer])

        # 4 Return stream and workflow
        return stream, workflow



# 2 Prepare for test
cycle_limit = 10000
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

myscenario.init_plot( p_plot_settings=PlotSettings( p_view = PlotSettings.C_VIEW_3D,
                                                        p_step_rate = step_rate ) )
input('\nPlease arrange all windows and press ENTER to start stream processing...')


tp_before           = datetime.now()
myscenario.run()
tp_after            = datetime.now()
tp_delta            = tp_after - tp_before
duraction_sec       = ( tp_delta.seconds * 1000000 + tp_delta.microseconds + 1 ) / 1000000
myscenario.log(Log.C_LOG_TYPE_W, 'Duration [sec]:', round(duraction_sec,2), ', Cycles/sec:', round(cycle_limit/duraction_sec,2))


# 5 Summary
anomalies         = myscenario.get_workflow()._tasks[2].get_anomalies()
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

with open('ntm_3d_i.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['Instance'])

    for anomaly in anomalies.values():
        anomaly_name = anomaly.C_NAME
        anomaly_id = anomaly.id
        inst_id = anomaly.get_instances()[-1].get_id()
        writer.writerow([inst_id])

input('Press ENTER to exit...')
