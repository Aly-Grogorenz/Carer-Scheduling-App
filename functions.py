# -*- coding: utf-8 -*-

######## Import Libraries ###########
#import folium
#import folium.plugins
import math
import itertools
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import hdbscan
import openrouteservice
from openrouteservice import distance_matrix
from convertbng.util import convert_lonlat

######## Global Statements ###########
seed = 0
#workhours = [7, 9, 12, 15, 18, 20]  # 20 as an end point, not actual shift start time
api_key = '5b3ce3597851110001cf6248e6106ee68a6e4fcbbf86a9948093c1ef'
#coords_falmouth = [50.152544233490545, -5.0788397148905124]   # lat long of center of Falmouth

######### Helper Functions ###############
# from the sub_id (21_1) etc, get the overarching client ID for display purposes
def get_client_id(cl_multi_id):
  ci = cl_multi_id.split("_")
  return ci[0]


def str_time(time):  # returns a string of time in HH:MM format when given a time tuple
  hour = str(time[0])
  if time[1] < 10:
    minute = "0"+str(time[1])
  else:
    minute = str(time[1])

  return hour+":"+minute

# takes a tuple of (hh, mm) and passes the time by a number of minutes:
def pass_time(current_time, passed_min):   # passed_min needs to be positive & a whole number
  current_hour = current_time[0]
  current_min = current_time[1]

  current_min += passed_min

  for i in range(int((passed_min//60)+1)):  # whole multiples of 60 (at least 1)

    if current_min >= 60:
      current_min -= 60
      current_hour += 1

  return(int(current_hour), int(current_min))

# returns a string of the text schedule for a given carer
def print_schedule(ca_id, carers_df, clients_df):
  txt = ""
  which_carer = ca_id
  ca_row = carers_df[carers_df['id'] == which_carer].index[0]

  current_time = (carers_df['shift_times'][ca_row][0], 0)   # start of when the carer leaves
  txt += "Schedule for carer "+str(which_carer)+"\n\n"
  txt += "Start day at "+str_time(current_time)+ " at home ("+ carers_df['start'][ca_row]+").\n"

  for i in range(len(carers_df['time_travelling'][ca_row])):

    # check for break here - is the current time in the client's time windows)
    cu_client = carers_df['route'][ca_row][i+1]  # current client
    cl_row = clients_df[clients_df['id'] == cu_client].index[0]
    need_wait = True   # assumes that carer needs to wait
    for w in clients_df['time_windows'][cl_row]:
      if current_time[0] >= w[0] and current_time[0] <= w[1]:
        need_wait = False

    if need_wait:   # carer waits until the earliest time window for the client
      # find earliest hour:
      h = 25  # latest, impossible
      for w in clients_df['time_windows'][cl_row]:
        if w[0] < h and w[0] >= current_time[0]:
          h = w[0]

      # wait until carer has to leave
      minute = math.floor(60-carers_df['time_travelling'][ca_row][i])
      hour = h - (math.ceil((60-minute)/60))   # reduce hour only if travel time is more than 0
      if minute == 60:
        minute = 0
      current_time = (hour, minute)
      txt += "Wait until "+str_time(current_time) + "\n"

    txt += "Leave current location at "+str_time(current_time)+".\n"
    txt += "Travel for " +str(carers_df['time_travelling'][ca_row][i])+ " minutes.\n"
    current_time = pass_time(current_time, math.ceil(carers_df['time_travelling'][ca_row][i]))  # round up

    txt += "Arrive at client nr. " +str(cu_client)+ " ("+clients_df['location'][cl_row]+")"+ " at "+str_time(current_time)+".\n"
    txt += "Provide care work for one hour.\n\n"
    current_time = pass_time(current_time, 60)  # one hour passes

  txt += "Carer finishes the day at "+str_time(current_time)+".\n"
  # mention workload
  if carers_df['hours'][ca_row] > 0:  # not enough hours
    txt += "They requested "+str(round(carers_df['hours'][ca_row], 2))+ " more workhours."

  elif carers_df['hours'][ca_row] == 0:  # worked exactly right
    txt += "They filled their requested workhours."
  else:
    txt += "They worked overtime for "+ str(abs(round(carers_df['hours'][ca_row], 2)))+ " hours."

  return txt


# this might be the exact same function as above, with slight modifications to allow for multiple visits
def print_schedule_multi(ca_id, carers_df, clients_df):
    txt = ""
    which_carer = ca_id
    ca_row = carers_df[carers_df['id'] == which_carer].index[0]

    current_time = (carers_df['shift_times'][ca_row][0], 0)   # start of when the carer leaves
    txt += "Schedule for carer "+str(which_carer)+"\n\n"
    txt += "Start day at "+str_time(current_time)+ " at home ("+ carers_df['start'][ca_row]+").\n"
    
    for i in range(len(carers_df['time_travelling'][ca_row])):

        # check for break here - is the current time in the client's time windows)
        cu_client = carers_df['route'][ca_row][i+1]  # current client
        cl_row = clients_df[clients_df['id'] == cu_client].index[0]
        need_wait = True   # assumes that carer needs to wait
        for w in clients_df['time_windows'][cl_row]:
            if w[0] <= current_time[0] and current_time[0] <= w[1]:
                need_wait = False

        if need_wait:   # carer waits until the earliest time window for the client
            # find earliest hour:
            h = 25  # latest, impossible
            for w in clients_df['time_windows'][cl_row]:
                if w[0] < h and w[0] >= current_time[0]:
                    h = w[0]

            # wait until carer has to leave
            minute = math.floor(60-carers_df['time_travelling'][ca_row][i])
            hour = h - (math.ceil((60-minute)/60))   # reduce hour only if travel time is more than 0
            if minute == 60:
                minute = 0
            current_time = (hour, minute)
            txt += "Wait until "+str_time(current_time) + "\n"

        txt += "Leave current location at "+str_time(current_time)+".\n"
        txt += "Travel for " +str(carers_df['time_travelling'][ca_row][i])+ " minutes.\n"
        current_time = pass_time(current_time, math.ceil(carers_df['time_travelling'][ca_row][i]))  # round up

        care_work = 1   # future proof?
        txt += "Arrive at " + get_client_id(cu_client)+ " ("+clients_df['location'][cl_row]+")"+ " at "+str_time(current_time)+".\n"
        txt += "Provide care work for "+ str(care_work) +" hour(s).\n\n"
        current_time = pass_time(current_time, care_work*60)  # one hour passes per "client"

    txt += "Carer finishes the day at "+str_time(current_time)+".\n"
    # mention workload
    if carers_df['hours'][ca_row] > 0:  # not enough hours
        txt += "They requested "+str(round(carers_df['hours'][ca_row], 2))+ " more workhours."

    elif carers_df['hours'][ca_row] == 0:  # worked exactly right
        txt += "They filled their requested workhours."
    else:
        txt += "They worked overtime for "+ str(abs(round(carers_df['hours'][ca_row], 2)))+ " hours."
    
    return txt
    


# for which carer
# returns array of start & end times with linked type (care work or travel)
# to be used in plotting the workday of a carer
def table_schedule(ca_id, carers_df, clients_df):
  start_times = []  # in full hours
  end_times = []    # in full hours
  work_types = []   # 'travel' or 'care'
  clients = []      # half as long, only keep clients

  # go through schedule of that carer
  which_carer = ca_id
  ca_row = carers_df[carers_df['id'] == which_carer].index[0]
  current_time = (carers_df['shift_times'][ca_row][0], 0)  # time when carer starts work

  for i in range(len(carers_df['time_travelling'][ca_row])):
    # check for break here - is the current time in the client's time windows)
    cu_client = carers_df['route'][ca_row][i+1]  # current client
    cl_row = clients_df[clients_df['id'] == cu_client].index[0]
    need_wait = True   # assumes that carer needs to wait
    for w in clients_df['time_windows'][cl_row]:
      if current_time[0] >= w[0] and current_time[0] <= w[1]:
        need_wait = False
    if need_wait:   # carer waits until the earliest time window for the client
      # find earliest hour:
      h = 25  # latest, impossible
      for w in clients_df['time_windows'][cl_row]:
        if w[0] < h and w[0] >= current_time[0]:
          h = w[0]
      # wait until carer has to leave
      minute = math.floor(60-carers_df['time_travelling'][ca_row][i])
      hour = h - (math.ceil((60-minute)/60))   # reduce hour only if travel time is more than 0
      if minute == 60:
        minute = 0
      current_time = (hour, minute)   # it is now time to leave
    
    
    # first the carer travels:
    work_types.append('travel')
    start_times.append(current_time[0]+current_time[1]/60)
    current_time = pass_time(current_time, math.ceil(carers_df['time_travelling'][ca_row][i])) # rounds up
    end_times.append(current_time[0]+current_time[1]/60)

    # then they provide care work:
    work_types.append('care')
    start_times.append(current_time[0]+current_time[1]/60)
    current_time = pass_time(current_time, 60)  # one hour passes
    end_times.append(current_time[0]+current_time[1]/60)
    # which client:
    clients.append(cu_client)

  return start_times, end_times, work_types, clients


# receives list of client_ids (string) and the carers_df
def client_visits_info(client_ids, carers_df, clients_df):
    carer_ids = []
    for i in range(len(carers_df)):
        for cl in client_ids:
            if cl in carers_df['route'][i]:
                carer_ids.append(carers_df['id'][i])
    
    visit_times = []   # list of tuples
    visit_duration = [] # list of numbers
    visiters = []   # list of strings
    
    for ca in carer_ids:
        # go through day, write down time as tuple when one of the clients is visited
        ca_row = carers_df[carers_df['id'] == ca].index[0]

        current_time = (carers_df['shift_times'][ca_row][0], 0)   # start of when the carer leaves

        for i in range(len(carers_df['time_travelling'][ca_row])):
            # check for break here - is the current time in the client's time windows)
            cu_client = carers_df['route'][ca_row][i+1]  # current client
            cl_row = clients_df[clients_df['id'] == cu_client].index[0]
            need_wait = True   # assumes that carer needs to wait
            for w in clients_df['time_windows'][cl_row]:
                if current_time[0] >= w[0] and current_time[0] <= w[1]:
                    need_wait = False

            if need_wait:   # carer waits until the earliest time window for the client
                # find earliest hour:
                h = 25  # latest, impossible
                for w in clients_df['time_windows'][cl_row]:
                    if w[0] < h and w[0] >= current_time[0]:
                        h = w[0]

                # wait until carer has to leave
                minute = math.floor(60-carers_df['time_travelling'][ca_row][i])
                hour = h - (math.ceil((60-minute)/60))   # reduce hour only if travel time is more than 0
                if minute == 60:
                    minute = 0
                current_time = (hour, minute)
    
            #carer travels to first client
            current_time = pass_time(current_time, math.ceil(carers_df['time_travelling'][ca_row][i]))  # round up

            # carer arrives at client - check if client is in relevant list
            if cu_client in client_ids:
                # write down start time & how much time passes
                visit_times.append(current_time)
                visiters.append(ca)
                visit_duration.append(1)   # one hour for now

            # either way, one hour (here) passes
            current_time = pass_time(current_time, 60)
      
    return visit_times, visit_duration, visiters


def text_schedule_clients(client_ids, carers_df, clients_df):
    visit_times, visit_duration, visiters = client_visits_info(client_ids, carers_df, clients_df)
    txt = ''
    
    # find max & min of visit time hours:
    earliest_h = 22
    latest_h = 7

    for t in visit_times:
        if t[0] < earliest_h:
            earliest_h = t[0]
        if t[0] > latest_h:
            latest_h = t[0]
            
    
    hours = np.arange(earliest_h, latest_h+1)  # earliest_h, latest_h+1
    for h in hours:
        # find position of that hour in the visit_times array
        for i in range(len(visit_times)):
            if h in visit_times[i]:   # current visit time found
                txt += "Visit at "+ str_time(visit_times[i]) + " by "+ str(visiters[i])+ " for "+ str(visit_duration[i])+ " hour(s).\n"  
                
    return txt


# same function as above, but allows for multiple visits
def clients_visit_info_multi(carer_ids, client_ids, carers_df, clients_df):
    visit_times = []
    visiter = []   # keep track of who visits
    visit_duration = []
    
    for ca in carer_ids:
        # go through day, write down time as tuple when one of the clients is visited
        ca_row = carers_df[carers_df['id'] == ca].index[0]

        current_time = (carers_df['shift_times'][ca_row][0], 0)   # start of when the carer leaves

        for i in range(len(carers_df['time_travelling'][ca_row])):
            # check for break here - is the current time in the client's time windows)
            cu_client = carers_df['route'][ca_row][i+1]  # current client
            cl_row = clients_df[clients_df['id'] == cu_client].index[0]
            need_wait = True   # assumes that carer needs to wait
            for w in clients_df['time_windows'][cl_row]:
                if w[0] <= current_time[0] and current_time[0] <= w[1]:
                    need_wait = False

            if need_wait:   # carer waits until the earliest time window for the client
                # find earliest hour:
                h = 25  # latest, impossible
                for w in clients_df['time_windows'][cl_row]:
                    if w[0] < h and w[0] >= current_time[0]:
                        h = w[0]

                # wait until carer has to leave
                minute = math.floor(60-carers_df['time_travelling'][ca_row][i])
                hour = h - (math.ceil((60-minute)/60))   # reduce hour only if travel time is more than 0
                if minute == 60:
                    minute = 0
                current_time = (hour, minute)
            #carer travels to first client
            current_time = pass_time(current_time, math.ceil(carers_df['time_travelling'][ca_row][i]))  # round up
            # carer arrives at client - check if client is in relevant list
    
            if cu_client in client_ids:
                # write down start time & how much time passes
                visit_times.append(current_time)
                visiter.append(ca)
                visit_duration.append(1)   # one hour for now

            # either way, one hour (here) passes
            current_time = pass_time(current_time, 60)  # one hour passes
    
    
    # new lists without the duplicates
    visiter2 = []
    visit_times2 = []
    visit_duration2 = []

    for v in range(len(visiter)):
        if v == 0:  # first one always gets written down
            #print("First one always counts")
            visiter2.append(visiter[v])
            visit_times2.append(visit_times[v])
            visit_duration2.append(visit_duration[v])
        elif visiter[v-1] == visiter[v]:  # same visiter back to back
            #print(True)
            # check if times are also visit_times apart, if not then can be written down
            if pass_time(visit_times[v-1], visit_duration[v-1]*60) != visit_times[v]:
                #print("But get it anyway")
                visiter2.append(visiter[v])
                visit_times2.append(visit_times[v])
                visit_duration2.append(visit_duration[v])
            else:   # otherwise, increase the value of the exisiting visit_duration2 by the other visit_duration
                visit_duration2[-1] += visit_duration[v]
      
        else:
            #print(False)
            # just take over the values
            visiter2.append(visiter[v])
            visit_times2.append(visit_times[v])
            visit_duration2.append(visit_duration[v])
    
    
    return visit_times2, visiter2, visit_duration2


# text schedule that allows for multiple visits
def text_schedule_clients_multi(carer_ids, client_ids, carers_df, clients_df):
    txt = ""
    
    visit_times2, visiter2, visit_duration2 = clients_visit_info_multi(carer_ids, client_ids, carers_df, clients_df)
    earliest_h2 = 22
    latest_h2 = 7

    for t in visit_times2:
        if t[0] < earliest_h2:
            earliest_h2 = t[0]
        if t[0] > latest_h2:
            latest_h2 = t[0]

    hours = np.arange(earliest_h2, latest_h2+1)  # earliest_h, latest_h+1
    for h in hours:
        #print(h)
        # find position of that hour in the visit_times array
        for i in range(len(visit_times2)):
            #print(visit_times[i])
            if h == visit_times2[i][0]:   # current visit time found
                txt += "Visit at "+ str_time(visit_times2[i]) + " by "+ str(visiter2[i])+ " for "+ str(visit_duration2[i])+ " hour(s).\n"
    
    return txt



###### Coordinate Stuff ########

# converts postcode into long lat format
def convert_to_longlat(postcode, region_postcodes):  # takes post code as string
  pc_index = region_postcodes[region_postcodes['Postcode'] == postcode]['Postcode'].index[0]
  easting = region_postcodes[region_postcodes['Postcode'] == postcode]['Eastings'][pc_index]
  northing = region_postcodes[region_postcodes['Postcode'] == postcode]['Northings'][pc_index]
  converted = convert_lonlat([easting], [northing])
  return [converted[0][0], converted[1][0]]   # return as longlat list

# jitters the coordinates for the visualisation of routings
# by 0.000025 in either direction
def jitter_coords(coord_list):  # take list of coordinates (lists of long lat)
  c_list = []
  for point in coord_list:
    r1 = np.random.randint(2)  # random, either 0 or 1
    r2 = np.random.randint(2)
    offset1 = np.random.randint(25, 51)  # random nr between 25 & 50
    offset2 = np.random.randint(25, 51)

    if r1 == 0:  # first coord in negative
      c1 = point[0]-(offset1/100000)
      0.000025
    elif r1 == 1:
      c1 = point[0]+(offset1/100000)

    if r2 == 0:   # second coord negative
      c2 = point[1]-(offset2/100000)
    elif r2 == 1:
      c2 = point[1]+(offset2/100000)

    c_list.append([c1, c2])

  return c_list


###### Time Window Functions ##########

# given a list of start of workhours and which position in the list (h)
# as well as a df of carers & clients
# returns the points relevant in that time window
def timewindow_points(start, end, ca_df, cl_df):

  # which carers and clients are relevant:
  current_ca = []
  ca_prio = []
  for i in range(len(ca_df)):
    # if the carer can work right now (start time, end time, still hours to work)
    if ca_df['shift_times'][i][0] < end+1 and ca_df['shift_times'][i][1] > start and ca_df['hours'][i] > 0:
      # moved to second row for legibility reasons
      # earliest free hour is before or equal to the current end nr hour
      if ca_df['earliest_free'][i][0] <= end: #or (ca_df['earliest_free'][i][0] == end and carers_df['earliest_free'][i][1] == 0):
        current_ca.append(ca_df['id'][i])    # maybe +1 in previous line? test
        ca_prio.append(ca_df['empty_hours'][i])

  current_cl_all = []
  cl_prio_all = []
  for i in range(len(cl_df)):
    # is there work left to do? (check dwell time):
    if cl_df['dwell'][i] > 0:
      # go through each time window of the client:
      for w in cl_df['time_windows'][i]:
        # if client can be cared for right now (start time, end time, work left to do)
        if w[0] <= start and w[1] >= end:
          current_cl_all.append(cl_df['id'][i])
          cl_prio_all.append(cl_df['flexibility'][i])

  # check of whether there is any point in continuting:
  return current_ca, ca_prio, current_cl_all, cl_prio_all


# returns a list of clients & their priority
# given a list of all clients & either priorities and of carer workloads
def timewindow_clients(start, end, cl_all, cl_prio_all, wl):
  # after the workload of carers has been determined, make subselection of clients:
# highest priority of clients (smaller number = higher prio)
# until workload is filled (can be slightly fuller if equivalent workload)
  current_cl = []  # subselection
  cl_prio = []
  prio_min = min(cl_prio_all)
  prio_max = max(cl_prio_all)
  n = prio_min   # current priority being considered

  cont = True
  while cont:
    for i in range(len(cl_prio_all)):   # go through the list
      if cl_prio_all[i] == n:   # priority at that place matches the current prio being considered
        current_cl.append(cl_all[i])
        cl_prio.append(cl_prio_all[i])
    # after getting all clients with that priority:
    n += 1   # increase priority nr by 1
    if len(current_cl) >= (sum(wl)-len(wl)) or n > prio_max:   # stop of there's enough clients
      cont = False
      break

  return current_cl, cl_prio

# returns the current workload of a given list of carers
# also needs to current start & end times for the time window
def timewindow_wl(start, end, carers, ca_df):
  current_wl = []
  for c in carers:
    c_index = ca_df[ca_df['id'] == c].index[0]

    # max nr of hours in time window x 2/diff to end time and when carer can start work/the nr of full hours the carer can still work (round down)
    if ca_df['earliest_free'][c_index][1] == 0:
      upper_limit = min(ca_df['shift_times'][c_index][1]-start+1, end-start+2, end+2-ca_df['earliest_free'][c_index][0], math.ceil(ca_df['hours'][c_index]+1)) ## +1 for the carer
    else:                                 # change here, either full hour or not; below used to be +1 (conservative)     # used to be math.floor (rounded down)
      upper_limit = min(ca_df['shift_times'][c_index][1]-start+1, end-start+2, end+2-ca_df['earliest_free'][c_index][0], math.ceil(ca_df['hours'][c_index]+1))
    # at least size 2 (self + 1 client)
    current_wl.append(max(2, upper_limit))

  return current_wl

######### Clustering Functions ###########
# making soft_clusters functions differently with precomputed distances...
# use knowledge of the distances and the cluster affiliation of each point
# for any point, take the average distance to all points of a cluster.
# smaller distance = higher probability

def make_soft_clusters(cl_labels, d_matrix):
  soft_clusters = []
  for i in range(len(cl_labels)):
    max_i = max(d_matrix[i])
    probs = []
    for cluster in set(list(cl_labels)):
      ######### not sure about this
      if cluster > -1 or max(cl_labels) == -1:    # no probability for outlier cluster, or only outlier clusters
        # find points in that cluster
        point_indices = np.where(cl_labels == cluster)[0]
        # average distances of those points to i
        avg_dist = np.mean(d_matrix[i][point_indices])

        # translate into probability:
        p = (max_i-avg_dist)/max_i   # if same cluster, avg_dist small, so percentage large
        probs.append(p)   # probabilities for point i to belong to each cluster
    soft_clusters.append(probs)  # probabilities of each point

  return soft_clusters


# make clusters from recalculated distance matrix (np array) & the number of carers
# returns incorrect fullness status
def make_clusters_postcodes(d_matrix, nr_ca):
  clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples = 1,
                            metric='precomputed').fit(d_matrix)

  soft_clusters = make_soft_clusters(clusterer.labels_, d_matrix)

  carer_clusters = clusterer.labels_[0:nr_ca]

  clusters = list(set(list(clusterer.labels_)))

  cl_elements = []   # what the elements (points) of each cluster are
  nr_ca = []   # how many carers are in the cluster
  cl_complete = []   # are the clusters full? ("Not Full" default)
  cl_size = []   # what is size of the cluster = default 0

  for cl in clusters:
    temp = [i for i, e in enumerate(list(clusterer.labels_)) if e == cl]  # list of points in the cluster
    cl_elements.append(temp)
    t = list(carer_clusters).count(cl)   # how many of that cluster are in the list of known carer clusters
    nr_ca.append(t)
    cl_complete.append("Not Full")    # default
    cl_size.append(0)   # default

  cl_df2 = pd.DataFrame({'cluster': clusters,  # index of the cluster
                      'elements': cl_elements,   # points in the cluster
                      'complete': cl_complete,   # statement about how full a cluster is
                      'carers': nr_ca,    # how many carers are in this cluster
                      'size': cl_size})

  # make sure if there's only 1 cluster that it's 0 and not a -1 cluster:
  cl_df2['cluster'][0] = 0

  return cl_df2, soft_clusters



# reassign all points in a -1 cluster to the cluster they most likely belong to (regardless of carer or not)
# also ignores if the cluster is already full or not
##### handle cases: probs 0 or nan -> just move to any cluster?
def reassign_minusone(cluster_df, softs):  # wl = workload
  cl_df2 = cluster_df.copy()
  soft = softs.copy()

  # find all points that are in -1 cluster
  outliers = []
  for l in cl_df2[cl_df2['cluster'] == -1]['elements']:
    outliers += l

  # max probabilities of those points:
  probs = []
  for point in outliers:
    probs.append(max(soft[point]))

  pr = len(outliers)   # copy of length at that point, since the length will change

  if pr > 0:
    for i in range(pr):    # how many points need to be reassigned
      # find the position of the maximum probability
      pos = probs.index(max(probs))
      # which point does that belong to
      current_point = outliers[pos]

      # where that point used to be (outlier cluster, so -1 and max row index)
      old_cluster = -1
      old_row_number = len(cl_df2)-1

      # if the max probability is 0 or nan:
      if max(probs) == 0 or math.isnan(max(probs)):
        # add those points to closest of the free clusters (not yet)  ##########
        # which of the carer clusters are not full yet (row numbers)
        pot_rows = cl_df2[cl_df2['complete'] == 'Not Full'][cl_df2['carers'] > 0].index   # array of rows
        # add point to first row for now, maybe distance matrix could be used but I think this works fine #####
        new_row_number = pot_rows[0]
        new_cluster = cl_df2['cluster'][new_row_number]
      else:   # proceed as before
        # where should that point go
        new_cluster = list(soft[current_point]).index(probs[pos])   ###### this part struggles with NAs
        # what is corresponding row numbers
        new_row_number = cl_df2[cl_df2['cluster'] == new_cluster]['elements'].index[0]   # returns row number

      new_row = cl_df2[cl_df2['cluster'] == new_cluster]['elements'][new_row_number]   # same as below, not [new_cluster at the end]
      new_row.append(current_point)
      cl_df2[cl_df2['cluster'] == new_cluster]['elements'][new_row_number] = new_row

      old_row = cl_df2[cl_df2['cluster'] == old_cluster]['elements'][old_row_number]  # struggles with -1, specifically the last [old_cluster] -> needs actual row number in df
      old_row.remove(current_point)
      cl_df2[cl_df2['cluster'] == old_cluster]['elements'][old_row_number] = old_row

      # finally, remove that point from probs and p
      outliers.remove(current_point)
      probs.remove(probs[pos])

  return cl_df2    # soft cluster affiliations don't change

# after moving points around, update the complete and carers columns (assumes cluster size of 5)
# arguments: cluster df, nr of carers
def update_cl_df_workload(cluster_df, nr_ca, wl):   # add workload (here size of the cluster)
  cl_df2 = cluster_df.copy()

  carers = []
  for i in range(nr_ca):
    carers.append(i)

  # update carer numbers first:
  for row in range(len(cl_df2)):
    n = 0  # count carers in row
    w = 0  # count workload in row  (needs to be done like this in case a carer got moved from the -1 cluster)
    for c in carers:
      if c in cl_df2['elements'][row]:
        n += 1
        w += wl[c]
    cl_df2['carers'][row] = n

    # then update fullness of the cluster
    if n == 0:   # no carers in cluster
      cl_df2['complete'][row] = 'Not Full'
      cl_df2['size'][row] = 0
    else:
      if len(cl_df2['elements'][row]) == w:
        cl_df2['complete'][row] = 'Full'
        cl_df2['size'][row] = w
      elif len(cl_df2['elements'][row]) > w:
        cl_df2['complete'][row] = 'Too Full'
        cl_df2['size'][row] = w
      else:
        cl_df2['complete'][row] = 'Not Full'
        cl_df2['size'][row] = w

  return cl_df2


# given a dataframe with carer rows that are too full, move the points that are least likely to
# belong to that cluster into the next best cluster
# return the new dataframe and the updated soft cluster probabilities (to stop reassignment to old cluster)
# and return a list of points that couldn't be assigned within workload constraints
# also take into account the priority of client allocation (lowest = moved last)

def reassign_carers_too_full_priority(cluster_df, softs, nr_ca, cl_prio):
  cl_df2 = cluster_df.copy()
  soft = softs.copy()     # len(soft) = how many points there are in total
  prios = cl_prio.copy()
  carers = []   # carer indices
  for i in range(nr_ca):
    carers.append(i)
  unassigned_points = []    # to potentially store points that can't go in any cluster (if not enough space)

  # continue until all necessary points have been reassigned
  cont = True
  while cont:

    # indices of clusters that are not yet full (nf) - could include carers or no carers
    nf_index = []
    for l in cl_df2[cl_df2['complete'] == 'Not Full']['cluster']:
      nf_index.append(l)
    # remove -1 cluster for simplicity
    if -1 in nf_index:
      nf_index.remove(-1)

    # get list of points that are in too full carer clusters (any row, not just one)
    p = []
    for l in cl_df2[cl_df2['carers'] > 0][cl_df2['complete'] == 'Too Full']['elements']:
      p += l

    # make sure not to include the carers in this list
    for c in carers:
      if c in p:
        p.remove(c)

    if len(p) == 0:
      cont = False
      break     # end if no client point has to be reassigned

    current_prios = []
    for point in p:
      current_prios.append(prios[point-nr_ca])

    ##### priority considered here: clients with higher priority (lower value) stay in their cluster for longer
    #prio_min = min(prios)
    prio_max = max(current_prios)
    #prio_max = find_max(prios, 100)    # find maximum, ignoring values of 100
    if prio_max == -1:   # there's only -1 priority (already moved points) left
      cont = False
      break

    prioritised_points = []   # only keep points that have the relevant priority
    for point in p:
      if prios[point-nr_ca] == prio_max:
        prioritised_points.append(point)   # keep points with the highest prio value (lowest priority) for reassignment
    p = prioritised_points    # removal caused half the points to be missed, so this is done instead

    # find the highest probabilities of the points belonging to their current (too full) cluster
    h_probs = []
    for point in p:
      if np.isscalar(soft[point]):   # theres only 1 cluster
        h_probs.append(soft[point])
      else:
        h_probs.append(max(soft[point]))

    # find the position of the lowest probability among h_probs
    pos = h_probs.index(min(h_probs))
    # which point does that belong to
    current_point = p[pos]

    # where that point used to be:
    for r in range(len(cl_df2)):
      if current_point in cl_df2.iloc[r]['elements']:
        old_cluster = cl_df2.iloc[r]['cluster']   # this is the "name" of the cluster (shouldn't be zero)
    # what is corresponding row number
    old_row_number = cl_df2[cl_df2['cluster'] == old_cluster]['elements'].index[0]

    # remove that probability from soft (set to 0)
    if np.isscalar(soft[current_point]):   # separate case if there's only 1 prob value
      soft[current_point] = 0
    else:
      soft[current_point][old_cluster] = 0

    if len(nf_index) == 0:    # there's no clusters with capacity
      unassigned_points.append(current_point)   # point to be added to -1 cluster instead of a new one

    else:    # point can be moved from one cluster to the other
      # where should that point go instead (find remaining maximum)
      probs_max = np.max(np.array(soft)[current_point][nf_index])   # maximum probability of soft cluster for point
      ####### changed here to np array

      if len(nf_index) == 1 or probs_max == 0:   # there's only one free cluster or no ideal new cluster
        # just move it to the only/first free cluster
        new_cluster = nf_index[0]

      else:   # there's more than 1 free cluster - 1 free only needs to be handled separately because of the way max() works
        new_cluster = np.where(soft[current_point] == probs_max)[0][0]  # index of new cluster

      new_row_number = cl_df2[cl_df2['cluster'] == new_cluster]['elements'].index[0]

      # move that point into that row of the dataframe
      new_row = cl_df2[cl_df2['cluster'] == new_cluster]['elements'][new_row_number]
      new_row.append(current_point)
      cl_df2[cl_df2['cluster'] == new_cluster]['elements'][new_row_number] = new_row

      # also check the status of the new cluster:
      if cl_df2['carers'][new_row_number] == 0: # not a carer cluster, doesn't matter
        cl_df2['complete'][new_row_number] = 'Not Full'
      elif len(cl_df2[cl_df2['cluster'] == new_cluster]['elements'][new_row_number]) == cl_df2['size'][new_row_number]:
        cl_df2['complete'][new_row_number] = 'Full'
      elif len(cl_df2[cl_df2['cluster'] == new_cluster]['elements'][new_row_number]) > cl_df2['size'][new_row_number]:
        cl_df2['complete'][new_row_number] = 'Too Full'

    # either way, the point gets taken out of the old cluster
    old_row = cl_df2[cl_df2['cluster'] == old_cluster]['elements'][old_row_number]  # struggles with -1, specifically the last [old_cluster] -> needs actual row number in df
    old_row.remove(current_point)
    cl_df2[cl_df2['cluster'] == old_cluster]['elements'][old_row_number] = old_row

    # after moving that point, check if the cluster is still too full now
    if len(cl_df2[cl_df2['cluster'] == old_cluster]['elements'][old_row_number]) == cl_df2['size'][old_row_number]:
      cl_df2['complete'][old_row_number] = 'Full'

    # finally, remove that point from probs and p
    p.remove(current_point)
    h_probs.remove(h_probs[pos])
    # also remove the corresponding priority:
    prios[current_point-nr_ca] = -1  # removed when considering maxima

    # preemptively stop the for loop once there's no more "Too Full"s:
    n_toofull = 0
    for status in cl_df2['complete']:
      if status == 'Too Full':
        n_toofull += 1
    if n_toofull == 0:   # if after going through there's no more "Too Fulls", break the loop and end function
      cont = False
      break

  return cl_df2, soft, unassigned_points


# given a dataframe from clustering as well as soft clusters object & points that are already unassigned
# reassign all points not in carer cluster to a carer cluster
# also add any points that can't be assigned to carer clusters (because of fullness)
# in a separate list & remove from dataframe
# this version also considers the priority of clients into account
 # (smaller number needs to be assigned to a carer cluster first)

def reassign_to_carers_priority(cluster_df, softs, unassigned, nr_ca, cl_prio):
  cl_df2 = cluster_df.copy()
  soft = softs.copy()
  unassigned_points = unassigned.copy()
  prios = cl_prio.copy()

  # a while statement that terminates only when all points have been reassigned
  cont = True # keep going
  while cont:   # gets set to False if the last point has been reclassified

    # get list of points not in carer clusters:
    p = []
    for l in cl_df2[cl_df2['carers'] == 0]['elements']:   # elements where there are no carers in the cluster
      p += l

    # end if there are no more points to reassign
    if len(p) == 0:
      cont = False
      break

    # get associated priorities of these points only:
    current_prios = []
    for point in p:
      current_prios.append(prios[point-nr_ca])

    ### modify p so that only those with specific priority are considered
    # only consider points with the highest priority (lowest value) to ensure those are still serviced
    prio_min = min(current_prios)   # which points are most important
    #if prio_min == 100:   # consists only of already reassigned points
    #  cont = False
    #  break

    prioritised_points = []
    for point in p:
      if prios[point-nr_ca] == prio_min:
        prioritised_points.append(point)    # only look at points with highest priority to reassign to carer clusters
    p = prioritised_points
    #print(p)

    if len(p) == 0:
      cont = False
      break

    # get cluster indices of carers from dataframe
    ca_index = []
    for l in cl_df2[cl_df2['carers'] > 0]['cluster']:  # there's at least one carer in this cluster
      ca_index.append(l)
    #print("Carer clusters: "+ca_index)

    # for each point, find and save the highest probability of affinity to a carer index only
    probs = []
    for point in p:
      probs.append(np.max(np.array(soft)[point][ca_index]))
    pr = len(p)   # copy of length at that point, since the length will change
    #print("Probs to carer clusters: "+probs)
    ###### changed to np array here


    # check if there's still carer clusters that aren't full (otherwise end loop)
    if len(cl_df2[cl_df2['complete'] == 'Not Full'][cl_df2['carers'] > 0]) == 0:
      # go through each point stored in p, add it to unassigned points and remove from df
      for current_point in p:
        unassigned_points.append(current_point)

        # where that point used to be
        for r in range(len(cl_df2)):
          if current_point in cl_df2.iloc[r]['elements']:
            old_cluster = cl_df2.iloc[r]['cluster']   # this is the "name" of the cluster, so can be -1
        # what is corresponding row number
        old_row_number = cl_df2[cl_df2['cluster'] == old_cluster]['elements'].index[0]

        # remove that point
        old_row = cl_df2[cl_df2['cluster'] == old_cluster]['elements'][old_row_number]  # struggles with -1, specifically the last [old_cluster] -> needs actual row number in df
        old_row.remove(current_point)
        cl_df2[cl_df2['cluster'] == old_cluster]['elements'][old_row_number] = old_row

        prios[current_point-nr_ca] = 100

      cont = False
      break   # stop the loop

    else:
      # loop through the length of the probs list
      for i in range(pr):  # the index i doesn't really come into play it's just how many times this repeats at max

        # find the position of the maximum probability
        pos = probs.index(max(probs))
        # which point does that belong to
        current_point = p[pos]

        # where that point used to be:
        for r in range(len(cl_df2)):
          if current_point in cl_df2.iloc[r]['elements']:
            old_cluster = cl_df2.iloc[r]['cluster']   # this is the "name" of the cluster, so can be -1
        # what is corresponding row number
        old_row_number = cl_df2[cl_df2['cluster'] == old_cluster]['elements'].index[0]

        # if the max probability is 0 or nan:
        if max(probs) == 0 or math.isnan(max(probs)):
          # add those points to closest of the free clusters (not yet)
          # which of the carer clusters are not full yet (row numbers)
          pot_rows = cl_df2[cl_df2['complete'] == 'Not Full'][cl_df2['carers'] > 0].index   # array of rows
          # add point to first row for now, maybe distance matrix could be used but I think this works fine #####
          new_row_number = pot_rows[0]
          new_cluster = cl_df2['cluster'][new_row_number]
        else:   # proceed as before
          # where should that point go
          new_cluster = list(soft[current_point]).index(probs[pos])   ###### this part struggles with NAs
          # what is corresponding row numbers
          new_row_number = cl_df2[cl_df2['cluster'] == new_cluster]['elements'].index[0]   # returns row number

        # can that point even go there?
        # check that cluster is not already full
        if cl_df2['complete'][new_row_number] == 'Not Full':
          # point can be moved:
          # move that point into that row of the dataframe
          new_row = cl_df2[cl_df2['cluster'] == new_cluster]['elements'][new_row_number]   # same as below, not [new_cluster at the end]
          new_row.append(current_point)
          cl_df2[cl_df2['cluster'] == new_cluster]['elements'][new_row_number] = new_row

          old_row = cl_df2[cl_df2['cluster'] == old_cluster]['elements'][old_row_number]  # struggles with -1, specifically the last [old_cluster] -> needs actual row number in df
          old_row.remove(current_point)
          cl_df2[cl_df2['cluster'] == old_cluster]['elements'][old_row_number] = old_row

          # after moving that point, check if the cluster is full now (here specifically at least 6 total points)
          if len(cl_df2[cl_df2['cluster'] == new_cluster]['elements'][new_row_number]) == cl_df2['size'][new_row_number]:
            cl_df2['complete'][new_row_number] = 'Full'
          elif len(cl_df2[cl_df2['cluster'] == new_cluster]['elements'][new_row_number]) > cl_df2['size'][new_row_number]:
            cl_df2['complete'][new_row_number] = 'Too Full'

          # finally, remove that point from probs and p
          if current_point in p:
            p.remove(current_point)
          if len(probs) > 0:
            probs.remove(probs[pos])
          # set the priority of the moved point to 100 -> since this algorithm considers minima, it's effectively removed
          prios[current_point-nr_ca] = 100

        # then check: if there's no free carer clusters left, break the for loop and go into other case
        if len(cl_df2[cl_df2['complete'] == 'Not Full'][cl_df2['carers'] > 0]) == 0:
           break

        else:  # that cluster is full or too full
          # otherwise, the point cannot move into that cluster:
          # set the probability of that point for that cluster to 0:
          soft[current_point][new_row_number] = 0
          # break the outer loop because probabilities need to be recalculated
          break

  return cl_df2, soft, unassigned_points


# when given a cluster dataframe with all points in a carer cluster and the result of soft clustering
# also given the coordinates of the points, and the remaining flexible hours (called prio)
# return a new dataframe where each row holding a cluster with 1 carer and a number of clients according to workload

# when given a cluster dataframe with all points in a carer cluster and the result of soft clustering
# also given the distances between points, and the remaining flexible hours (called prio)
# return a new dataframe where each row holding a cluster with 1 carer and a number of clients according to workload

def split_clusters_dmatrix(cluster_df, nr_ca, nr_cl, d_matrix, wl, prio):
  cl_df2 = cluster_df.copy()
  carers = []
  for i in range(nr_ca):
    carers.append(i)
  clients = []
  for i in range(nr_cl):
    clients.append(i+nr_ca)

  carer_df = cl_df2[cl_df2['carers'] > 0]

  cl_index = []   # index of new cluster
  cl_elements = []  # elements of new cluster
  n = 0

  # if all priorities are the same, continue as always:
  for cluster in carer_df['cluster']:
    # if there's only 1 carer, just copy over that row:
    if carer_df['carers'][cluster] == 1:
      cl_index.append(n)
      cl_elements.append(carer_df['elements'][cluster])
      # at the end, increase n (next index)
      n += 1

    else:   # cluster needs to be split

      temp_cluster = carer_df['elements'][cluster]   # the cluster that needs to be split
      joint_carers = []   # list of carers in that row
      cl_clients = []    # get a list of clients in that row

      # split up elements of the cluster into carers and clients
      for element in temp_cluster:
        if element in clients:
          cl_clients.append(element)
        if element in carers:     # carers automatically get added to a new row
          joint_carers.append(element)   # carers that can be considered
          # make a new row for eventual dataframe
          cl_index.append(n)
          n += 1
          cl_elements.append([element])   # already stored as list []

      # get matched priorities of carers in that cluster:
      current_prios = []
      for c in joint_carers:
        current_prios.append(prio[c])

      prios_at_start = current_prios.copy()  # back up because priority list will change

      # go through each level in current_prios (each priority, ascending)
      for pri in list(set(prios_at_start)):
        prio_carers = []     # current joint carers with that priority only
        for i in range(len(joint_carers)):
          if prios_at_start[i] == pri:
            prio_carers.append(joint_carers[i])   # subset of joint_carers

        while len(prio_carers) > 0 and len(cl_clients) > 0:   # if there are carers with that priority
          # go through the points and find the minimum distance
          min_d = 100000   # starting assignment

          # go through each client
          for point in cl_clients:
            # go through each carer
            for carer in prio_carers:
              # get distance to that carer:
              d = d_matrix[carer][point]
              # if that distance is better, store the current best distance, carer, and client
              if d < min_d:
                min_d = d
                best_carer = carer
                best_client = point

          # at the end when the best carer-client combo has been found, move that point into that cluster
          # get index (ind) of which row that carer is currently in
          for j in range(len(cl_elements)):
            if best_carer in cl_elements[j]:   # that's the row
              ind = j
              break
          # add best_client to that row:
          cl_elements[ind].append(best_client)

          # then remove the client from cl_clients
          if best_client in cl_clients:     #### does this fix the problem?
            cl_clients.remove(best_client)

          # assess the workload (size of cluster) for the carer and remove from carer lists if limit is reached
          if len(cl_elements[ind]) >= wl[best_carer]:
            if best_carer in prio_carers:
              prio_carers.remove(best_carer)
            #if best_carer in joint_carers:     ##### test if this fixes the problem, since the creation of later lists relies on this
            #  joint_carers.remove(best_carer)
            # also needs to remove associated priorities

          # break while loop if there's no more available carers or no more clients:
          if len(prio_carers) == 0 or len(cl_clients) == 0:
            break


  # make sure the elements of cl_elements are unique:
  for i in range(len(cl_elements)):
    cl_elements[i] = list(set(cl_elements[i]))

  # make new dataframe: cluster index (new), elements
  # other two rows aren't important anymore, this is the final dataframe
  new_df = pd.DataFrame({'cluster': cl_index,
                         'elements': cl_elements})
  return new_df


######### Route Travelling Functions ###############

# function that, when given a route (without carer) and a distance matrix (in seconds),
# returns the time spent travelling along each leg of the route (incl from carer) in minutes
def route_minutes(route, d_matrix):
  # start at position 0 (carer)
  r = [0]
  r += route

  # list of distances to each leg of the journey in minutes
  dist_list = []
  for i in range(len(r)-1):
    p1 = r[i]
    p2 = r[i+1]
    d = round(d_matrix[p1][p2]/60, 1)
    dist_list.append(d)

  return dist_list

# generate all permutations to clients, taking an np dist matrix which includes the carer
def make_perms_np(d_matrix):
  l = []
  for i in range(len(d_matrix)-1):
    l.append(i+1)   # ignore carer
  perms = list(itertools.permutations(l))

  return perms  # returns a list of tuples

# given an asymmetrical np array, travel all routes and find the shortest ones
def try_all_routes_dmatrix(d_matrix):
  # make permutations of all routes (does not include carer travelling there)
  p = make_perms_np(d_matrix)
  # for each permutation, find the travel time
  travelled = []
  for route in p:
    d = d_matrix[0][route[0]]   # distance from carer to first point in route
    for i in range(len(route)-1):    # between pairs, so -1
      d += d_matrix[route[i]][route[i+1]]   # current index and next, intersection is distance from first to second
    travelled.append(d)

  all_routes = pd.DataFrame({"route": p, "t_distance": travelled})

  # shortest routes only:
  m = all_routes["t_distance"].min(axis=0)
  shortest_routes = all_routes[all_routes['t_distance'] == m]['route']
  return shortest_routes, all_routes    # neither includes the carer node

# given a new_cl dataframe, conversion, and the distances between points
# returns which carers were travelling, their route, and the time spent travelling
def solve_tsp_dmatrix(clusters_df, conv, d_matrix):
  travel_ca = []  # travellaing carers
  travel_route = []  # route they travel
  travel_time = []   # time to reach each node in travel_route

  for elements in clusters_df['elements']:
    elements_sorted = elements.copy()
    elements_sorted.sort()

    # get the respective carer:
    travel_ca.append(conv[elements_sorted[0]])

    # make slice of dist_matrix with only the relevant points:
    # start of with first relevant d-matrix row
    dist_matrix2 = [d_matrix[elements_sorted[0], elements_sorted]]
    if len(elements_sorted) > 1:
      for i in range(len(elements_sorted)-1):  # then add next rows
        next_row = d_matrix[elements_sorted[i+1], elements_sorted]
        dist_matrix2 = np.append(dist_matrix2, [next_row], axis = 0)    ## originally np.stack without array around new_row


    if len(elements) > 2:    # there's more than just the carer and 1 client
      # brute force find the solution (they're not gonna get big enough to cause runtime problems)
      s, a = try_all_routes_dmatrix(dist_matrix2)

      converted_route = []
      for node in s[s.index[0]]:   # take the first best path
        converted_route.append(conv[elements_sorted[node]])   # double conversion: 1 within clustering, 1 within graph theory
      travel_route.append(converted_route)

      # time to travel the route (minutes):
      t = route_minutes(s[s.index[0]], dist_matrix2)
      travel_time.append(t)


    elif len(elements) == 2:  # 1 carer & 1 client
      # just go to that client
      s = elements_sorted[1]
      travel_route.append([conv[s]])
      t = round(dist_matrix2[0][1]/60, 1)    # travel time in seconds, needs to be converted into minutes
      travel_time.append([t])

    else:  # only carer in cluster, leave empty
      s = []
      t = []
      travel_route.append(s)
      travel_time.append(t)

  return travel_ca, travel_route, travel_time


# function that updates carer & clients dataframes,
# given a list of travelling carers & their routes and travel times,
# the global location of each starting point, and the start & end times of the current time window

def travelled_update_df_pc(ca, cl, travel_ca, travel_route, travel_time, start, end):
  ### index in carer df might be off

  ca_df2 = ca.copy()   # carers_df
  cl_df2 = cl.copy()   # clients_df

  # update the carers df:
  for i in range(len(travel_ca)):
    current_carer = travel_ca[i]   # current carer (real index)
    ca_index = ca_df2[ca_df2['id'] == current_carer].index[0]  # row number
    current_route = travel_route[i]
    current_travel = travel_time[i]    # travel times corresponding to route
    hours_worked = 0

    if len(current_route) > 0:  # if travel even happened

      # update position of carer to last place in route:
      cl_index = cl_df2[cl_df2['id'] == current_route[-1]].index[0]
      current_pos = cl_df2['location'][cl_index]

      ca_df2['location'][ca_index] = current_pos


      # update hours worked (reduce by dwell time in cl_df2)
      for cl in current_route:
        # find the client (cl) in the clients table
        cl_row = cl_df2[cl_df2['id'] == cl]['dwell'].index[0]
        # reduce dwell time by 1 (hourly increment)
        cl_df2['dwell'][cl_row] -= 1
        # increase hours worked for the carer (here +1 per client)
        hours_worked += 1

      # update earliest_free for a carer that travelled only
      time_spent_travelling = sum(current_travel)
      time_caring = 60*len(current_route)  # to update time_passed only; assumes 1h visits  ###

      time_passed = time_spent_travelling+time_caring   # how much time passed travelling (in minutes) + dwell of each client (here 1)
      cur_time = ca_df2['earliest_free'][ca_index]
      ca_df2['earliest_free'][ca_index] = pass_time(cur_time, time_passed)

      # also update hours_worked by the fraction taken up by travel time only:
      hours_worked += round(time_spent_travelling/60, 2)  # can make ugly fractions...


    ca_df2['hours'][ca_index] -= hours_worked   # removal of hours worked by being with clients

    # update route travelled
    ca_df2['route'][ca_index] += current_route
    # update time spent travelling that route
    ca_df2['time_travelling'][ca_index] += current_travel

  # afterwards, the maximum remaining working time for carers is based on
  # how many hours of their shift times are still left
  for i in range(len(ca_df2)):
    ca_df2['empty_hours'][i] = min(ca_df2['empty_hours'][i], max((ca_df2['shift_times'][i][1]-end-len(ca_df2['route'][i])+1-math.ceil(ca_df2['hours'][i])), 0)) # deal with negative numbers, -1 because of passage of time


  # update cl_df2 df: if a time windows has passed, the flexibility needs to be updated
  for r in range(len(cl_df2)):   # go through each row
    for w in cl_df2['time_windows'][r]:   # go through each time window
      if start in w:   # if this is the current time window
        # reduce the flexibility number by one:
        cl_df2['flexibility'][r] -= 1

  return ca_df2, cl_df2



######### Hourly Schedule Function #########

# also accepts the region mostcode csv
# function goes through the workhours and returns the updated carer & client dfs
def hourly_schedule(region_pc, workhours, ca_df2, cl_df2):   # replace carers_df, clients_df
  ca_df = ca_df2.copy()
  cl_df = cl_df2.copy()

  for h in range(len(workhours)-1):   # go through start of each shift
    current_start = workhours[h]    # it's the start of the hour
    current_end = workhours[h+1]-1

    # id of current carers, not row number
    current_carers, ca_priority, current_clients_all, cl_priority_all = timewindow_points(current_start, current_end, ca_df, cl_df)
    #print(current_start, current_end)
    #print(current_carers, ca_priority)

    # no carers or clients
    if len(current_carers) == 0 or len(current_clients_all) == 0:
      # change the remaining work hours for carers:
      for i in range(len(ca_df)):
        ca_df['empty_hours'][i] = min(ca_df['empty_hours'][i],
                                      max((ca_df['shift_times'][i][1]-current_end-len(ca_df['route'][i])+1)+1-math.ceil(ca_df['hours'][i]),
                                          0)
                                      )
      # update cl_df: if a time window has passed, the flexibility needs to be updated
      for r in range(len(cl_df)):   # go through each ro
        for w in cl_df['time_windows'][r]:   # go through each time window
          if current_start in w:  # this is the current time window
            cl_df['flexibility'][r] -= 1  # reduce flexibility by one
      continue     # skip this hour if there's no carers or clients

    current_workload = timewindow_wl(current_start, current_end, current_carers, ca_df)
    #print("Workload: " + str(current_workload))
    current_clients, cl_priority = timewindow_clients(current_start, current_end, current_clients_all, cl_priority_all, current_workload)
    #print("Selected clients: "+ str(current_clients))
    #print("Their priority: "+ str(cl_priority))

    # make conversion & current data:
    conversion=current_carers+current_clients    # stores id, not row number

    # current_data needs to be made from the location of the current points, separately for clients & carers:
    current_data = []   # stores the relevant postcodes
    current_coords = []  # store relevant longlat

    for carer in current_carers:
      ca_ind = ca_df[ca_df['id'] == carer]['location'].index[0]
      pc = ca_df['location'][ca_ind]
      current_data.append(pc)
      current_coords.append(convert_to_longlat(pc, region_pc))
    for client in current_clients:
      cl_ind = cl_df[cl_df['id'] == client]['location'].index[0]
      pc = cl_df['location'][cl_ind]
      current_data.append(pc)
      current_coords.append(convert_to_longlat(pc, region_pc))

    # calculate distances
    client = openrouteservice.Client(key=api_key)

    request = {'locations': current_coords,
               'profile': 'driving-car',   # assumes driving only, walking = 'foot-walking'
               'metrics': ['duration']}   # duration in seconds

    dist_matrix_info = client.distance_matrix(**request)   # carry all the info  - might be unnecessary
    dist_matrix = np.array(dist_matrix_info['durations'])  # np array of just distances
    #dist_matrix_df = pd.DataFrame(dist_matrix)   # d-matrix as dataframe


    # clustering
    clu_df, sft = make_clusters_postcodes(dist_matrix, len(current_carers))
    clu_df = reassign_minusone(clu_df, sft)
    clu_df = update_cl_df_workload(clu_df, len(current_carers), current_workload)
    clu_df, sft, unassigned = reassign_carers_too_full_priority(clu_df, sft, len(current_carers), cl_priority)
    clu_df = update_cl_df_workload(clu_df, len(current_carers), current_workload)
    clu_df, sft, unassigned = reassign_to_carers_priority(clu_df, sft, unassigned, len(current_carers), cl_priority)

    new_cl = split_clusters_dmatrix(clu_df, len(current_carers), len(current_clients), dist_matrix, current_workload, ca_priority)

    # solve the tsp:

    travelling_carers, route_travelled, time_travelled = solve_tsp_dmatrix(new_cl, conversion, dist_matrix)
    ca_df, cl_df = travelled_update_df_pc(ca_df, cl_df, travelling_carers, route_travelled, time_travelled, current_start, current_end)
  return ca_df, cl_df
