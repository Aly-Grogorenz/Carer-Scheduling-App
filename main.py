# -*- coding: utf-8 -*-
############# Libraries ####################
import os

# kivy imports
import kivy
kivy.require('2.3.0')  # keep current kivy version

from kivy.properties import ListProperty, BooleanProperty#, ObjectProperty, 
from kivy.lang import Builder
from kivy.graphics import Color, Line
from kivy_garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy_garden.mapview import MapView, MapMarkerPopup

from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.screenmanager import MDScreenManager
from kivymd.uix.filemanager import MDFileManager

# other imports
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import functions
import requests
import re

########## Global Variables ################
seed = 0
workhours = [7, 9, 12, 15, 18, 20]  # 20 as an end point, not actual shift start time
coords_falmouth = [50.152544233490545, -5.0788397148905124]
api_key = '5b3ce3597851110001cf6248e6106ee68a6e4fcbbf86a9948093c1ef'

# for reading in location data from the csv file
colnames = ['Postcode','Positional_quality_indicator','Eastings','Northings','Country_code','NHS_regional_HA_code','NHS_HA_code','Admin_county_code','Admin_district_code','Admin_ward_code']
# postcodes in Falmouth
tr_postcodes = pd.read_csv('tr.csv', names=colnames)
# other postcode lists can go here

########## Internal Functions #############
# sets up the global variables carers_df & clients_df
def setup_dfs():
    global carers_df, clients_df, clients_df_all
    global locations_df
    
    # carers df:
    identity = []
    home = []
    hours = []
    shift = []
    empty = []
    free = []
    route = []
    travel = []

    for r in range(len(carers_info)):
        identity.append(str(carers_info['ID'][r]))  # store as string
        home.append(carers_info['Postcode'][r])

        hour = carers_info['Hours'][r]
        hours.append(hour)

        start = carers_info['Available Start'][r]
        end = carers_info['Available End'][r]
        shift.append((start, end))

        empty.append(end-start-hour)
        free.append((start,0))

        route.append([carers_info['ID'][r]])
        travel.append([])

    carers_df = pd.DataFrame({'id': identity,
                          'start': home,
                          'location': home,
                          'hours': hours,
                          'shift_times': shift,
                          'empty_hours': empty,
                          'earliest_free': free,
                          'route': route,
                          'time_travelling': travel})
    
    # clients_df_all (all sub IDs in one)
    # make clients_df:
    # id, location, dwell (all 1), time_windows (based on when visit), flexibility (len of time_windows)
    # time_windows for visit start = 7-9, 9-12, 12-15, 15-18, 18-20
    # encoded as start of hours (7-8, 9-11, 12-14, 15-17, 18-19)

    # here, assume that clients need to be visited once during *all* time windows that they are available?
    # assume that during each time window, the visit time is 1h (for now) -> would need extra column for visit duration later

    identity = []
    home = []
    time_windows = []
    
    for r in range(len(clients_info)):
        identity.append(clients_info['ID'][r])
        home.append(clients_info['Postcode'][r])
        tw = []   # time windows of current client
        if clients_info['Early Morning'][r] == 'Yes':
            tw.append((workhours[0], workhours[1]-1))
        if clients_info['Morning'][r] == 'Yes':
            tw.append((workhours[1], workhours[2]-1))
        if clients_info['Lunch'][r] == 'Yes':
            tw.append((workhours[2], workhours[3]-1))
        if clients_info['Tea'][r] == 'Yes':
            tw.append((workhours[3], workhours[4]-1))
        if clients_info['Evening'][r] == 'Yes':
            tw.append((workhours[4], workhours[5]-1))
        time_windows.append(tw)



    clients_df_all = pd.DataFrame({'id': identity,
                                   'location': home,
                                   'time_windows': time_windows})
    
    
    # make new df (clients_df, holds one time window per client ID)
    identity = []
    home = []
    time_windows = []
    # keep these to be able to use old functions
    dwell = []
    flex = []

    for i in range(len(clients_df_all)):
        # go through each row
        sub = 1  # sub ID, start at one
        for tw in clients_df_all['time_windows'][i]:
            home.append(clients_df_all['location'][i])   # store location
            identity.append(clients_df_all['id'][i]+"_"+str(sub))
            sub += 1   # for each time window, increase sub ID by one
            time_windows.append([tw])
            # just makes sure the old functions still work...
            dwell.append(1)  # always one (for now)
            flex.append(1)

    clients_df = pd.DataFrame({'id': identity,
                               'location': home,
                               'time_windows': time_windows,
                               'dwell': dwell,
                               'flexibility': flex
                               })
    
    # jittered location for plotting
    np.random.seed(seed)

    # store ids and location in dataframe:
    ids = list(carers_df['id']) + list(clients_df_all['id'])
    ids.sort()   # in ascending order (strings)

    # for each id, find the matched jittered location - needs to be in order of ID to maintain same position
    og_pcs = []  # postcodes
    for i in ids:
        if i in list(carers_df['id']):
            ind = carers_df[carers_df['id'] == i].index[0]
            og_pcs.append(carers_df['start'][ind])
        elif i in list(clients_df_all['id']):
            ind = clients_df_all[clients_df_all['id'] == i].index[0]
            og_pcs.append(clients_df_all['location'][ind])

    og_coords = []  # coordinates
    for pc in og_pcs:
        og_coords.append(functions.convert_to_longlat(pc, tr_postcodes))


    coords = functions.jitter_coords(og_coords)
    locations_df = pd.DataFrame({'id': ids,
                                 'coords': coords})


########## Interface ################
# placeholder plot:
plt.plot([1, 23, 2, 4])


# define screens
class FirstWindow(MDScreen):
    selected_region = ''   # placeholder
    selected_carer = ""
    
    def checkbox_click(self, instance, value, region):
        if value == True:
            FirstWindow.selected_region = region  # global
        else:
            FirstWindow.selected_region = ''
        
    # this function isn't called I think, but removing it breaks the code so it has to stay
    def spinner_clicked(self, value):
        # save selection
        FirstWindow.selected_carer = value
        
    def run_program(self):
        global carers_df, clients_df
        
        # tr_postcodes for now, later allow for variable to change based on selection of region
        carers_df, clients_df = functions.hourly_schedule(tr_postcodes, workhours, carers_df, clients_df)
      
        
class CarerWindow(MDScreen):
    selected_carer = 'Alexandra Taylor'   # default, make sure this is always correct
    selected_display = 'Text'
    route_points = []   # to hold the points in a route
    graph_plotted = False
        
    
    # save selection of carer
    def spinner_clicked(self, value):
        # save selection
        CarerWindow.selected_carer = value  # update
        
    # save selection of display type
    def checkbox_click(self, instance, value, display):
        if value == True:
            CarerWindow.selected_display = display  # global
        else:
            CarerWindow.selected_display = ''


    def update_display(self):
        self.ids.label_reschedule.text = 'Adjust schedule \nfor '+ CarerWindow.selected_carer
        global carers_df
        global clients_df
        
        if CarerWindow.selected_display == 'Table':
            # clear any existing plots
            plt.cla()
            self.table_box.clear_widgets()
            
            # show table box 
            self.ids.table_box.size_hint = 0.9, 0.9
            
            which_carer = CarerWindow.selected_carer
            ca_row = carers_df[carers_df['id'] == which_carer].index[0]

            start_times, end_times, work_types, clients = functions.table_schedule(which_carer, carers_df, clients_df)
            
            # set plotting stuff:
            care_col = 'lightblue'
            travel_col = 'lightgrey'
            # legend
            care_patch = patches.Patch(color=care_col, label = 'Care Work')
            travel_patch = patches.Patch(color=travel_col, label = 'Travelling')
            fig, ax = plt.subplots()
            #canvas = fig.canvas   # testing to see if this works...

            # go through the work patches in work_types:
            cl = 0
            for i in range(len(work_types)):
                if work_types[i] == 'travel':
                    # travel stuff shown here
                    square = patches.Rectangle((0, start_times[i]), 2, (end_times[i]-start_times[i]), edgecolor = 'none', facecolor = travel_col)
                    ax.add_patch(square)    

                elif work_types[i] == 'care':
                    # care work stuff shown here
                    square = patches.Rectangle((0, start_times[i]), 2, (end_times[i]-start_times[i]), edgecolor = 'none', facecolor = care_col)
                    ax.add_patch(square)
                    # add label (which client)
                    if cl == 0: 
                        plt.text(0.05, end_times[i]-0.3, functions.get_client_id(clients[cl]), fontsize = 'small')
                    elif functions.get_client_id(clients[cl]) != functions.get_client_id(clients[cl-1]):
                        plt.text(0.05, end_times[i]-0.3, functions.get_client_id(clients[cl]), fontsize = 'small')
                    cl += 1  # index in client list

            # show shift times
            plt.axhline(y = carers_df['shift_times'][ca_row][0], color = 'red', linewidth = 0.5)
            plt.axhline(y = carers_df['shift_times'][ca_row][1], color = 'red', linewidth = 0.5)

            # set axes
            plt.xlim(0, 2)
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off

            plt.ylim(7, 21)
            plt.gca().invert_yaxis()
            plt.locator_params(axis='y', nbins=16)  # 21-7
            plt.ylabel("Time of Day")
            
            plt.legend(handles=[care_patch, travel_patch])
            plt.title('Schedule for '+str(which_carer))
            
            self.table_box.add_widget(FigureCanvasKivyAgg(plt.gcf()))
            #plt.plot()
            
            
            # hide text instructions
            self.ids.instructions_text.size_hint = None, None
            self.ids.instructions_text.height = '0dp'
            self.ids.instructions_text.width = '0dp'
            self.ids.instructions_text.text = ''
            # hide map instructions
            self.ids.map_box.size_hint = None, None
            self.ids.map_box.height = '0dp'
            self.ids.map_box.width = '0dp'
            self.map_box.clear_widgets()
                    
        
        elif CarerWindow.selected_display == 'Text':
            # change label of Output Box to the corresponding carer
            self.ids.instructions_text.size_hint = 0.9, 0.9
            self.ids.instructions_text.text = functions.print_schedule_multi(CarerWindow.selected_carer, carers_df, clients_df)
            # make table box disappear + clear existing plots
            # clear any existing plots
            self.ids.table_box.size_hint = None, None
            self.ids.table_box.height = '0dp'
            self.ids.table_box.width = '0dp'
            plt.cla()
            self.table_box.clear_widgets()
            
            # make map box disappear
            self.ids.map_box.size_hint = None, None
            self.ids.map_box.height = '0dp'
            self.ids.map_box.width = '0dp'
            self.map_box.clear_widgets()           
        
        
        elif CarerWindow.selected_display == 'Map':
            # show map box (and change label of placeholder button)
            self.ids.map_box.size_hint = 0.9, 0.9
            # put points of carer and final home
            
            # remove any markers that were already there
            self.map_box.clear_widgets()
            # remove points from route_points list
            CarerWindow.route_points = []
            
            # make a new map
            self.map = MapView(zoom = 13, lat=50.152544233490545, lon=-5.0788397148905124)
            
                      
            which_carer = CarerWindow.selected_carer
            ca_row = carers_df[carers_df['id'] == which_carer].index[0]
            node_list2 = carers_df['route'][ca_row]
            # get updated node_list with just overarching client IDs
            node_list = []
            for n in node_list2:
                node_list.append(functions.get_client_id(n))
            pc_coords2 = []  # list of long lat pairs
            for node in node_list:
                # find row in locations_df
                loc_row = locations_df[locations_df['id'] == node].index[0]
                pc_coords2.append(locations_df['coords'][loc_row])
            
            # put markers of these points:
            if len(pc_coords2) == 1:
                # only one point, no route:
                self.pin = MapMarkerPopup(lat = pc_coords2[0][1], lon = pc_coords2[0][0], color = 'blue')
                self.map.add_widget(self.pin)
            else:
                for i in range(len(pc_coords2)-1):
                    current_point = (pc_coords2[i][1], pc_coords2[i][0])  # lat long order
                    next_point = (pc_coords2[i+1][1], pc_coords2[i+1][0])
                    if i == 0:
                        self.pin = MapMarkerPopup(lat = current_point[0], lon =current_point[1], color = 'blue')
                        self.map.add_widget(self.pin)
                    else:
                        self.pin = MapMarkerPopup(lat = current_point[0], lon =current_point[1])
                        self.map.add_widget(self.pin)
                
                    # either way, add second point: 
                    self.pin = MapMarkerPopup(lat = next_point[0], lon = next_point[1])
                    self.map.add_widget(self.pin)
                
                    # add path:
                    self.body = {"coordinates":[pc_coords2[i],pc_coords2[i+1]]}

                    self.headers = {
                        'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
                        'Authorization': api_key,
                        'Content-Type': 'application/json; charset=utf-8'
                        }
                    # request as driving car
                    self.call = requests.post('https://api.openrouteservice.org/v2/directions/driving-car/gpx', json=self.body, headers=self.headers)
                    
                    # string result:
                    string_res = self.call.text

                    tag = 'rtept'
                    reg_str = '</' + tag + '>(.*?)'+ '>'
                    res = re.findall(reg_str, string_res)
                    string1 = str(res)
                    reg_str1 = '"' + '(.*?)' + '"'
                    res1 = re.findall(reg_str1, string1)
                    
                    
                    for p in range(0, len(res1)-1, 2): # step 2
                        
                        # add (invisible) waypoints:
                        self.points_pop = MapMarkerPopup(lat = res1[p], lon = res1[p+1], source = 'me_32.png')
                        CarerWindow.route_points.append(self.points_pop)
                        
                        
                        self.map.add_widget(self.points_pop)
                        
            
            # doesn't draw the line properly, try this: 
            # https://stackoverflow.com/questions/64423950/kivy-garden-mapview-methode-to-draw-itinirary
            
            # youtube guide: https://www.youtube.com/watch?v=5L18F5oDna8&t=55s
            
            self.map_box.add_widget(self.map)
            
            # I tried doing this here but it just broke everything
            
            with self.map.canvas:
                Color(0.5, 0, 0, 1)  # rgb, opacity
                for j in range(len(CarerWindow.route_points)-1):
                    self.map.lines = Line(points=(CarerWindow.route_points[j].pos[0], 
                                              CarerWindow.route_points[j].pos[1],
                                              CarerWindow.route_points[j+1].pos[0],
                                              CarerWindow.route_points[j+1].pos[1]),
                                      width = 4)
            
            
            # hide text instructions
            self.ids.instructions_text.size_hint = None, None
            self.ids.instructions_text.height = '0dp'
            self.ids.instructions_text.width = '0dp'
            self.ids.instructions_text.text = ''
            # hide table output + clear existing plots
            self.ids.table_box.size_hint = None, None
            self.ids.table_box.height = '0dp'
            self.ids.table_box.width = '0dp'
            plt.cla()
            self.table_box.clear_widgets()
        
        else:
            pass
        
        
    # change schedule based on changes entered in the text boxes    
    def reschedule(self):
        global carers_df, clients_df, carers_info, clients_info
        
        # change the availability & nr of workhours of current carer
        # columns: 'Hours', 'Available Start', 'Available End'
        
        # find current carer:
        ca_row = carers_info[carers_info['ID'] == CarerWindow.selected_carer].index[0]
        
        # change info - consider empty fields:
        if self.ids.new_workhours.text:   # is not empty
            carers_info['Hours'][ca_row] = int(self.ids.new_workhours.text)
        if self.ids.new_start.text:
            carers_info['Available Start'][ca_row] = int(self.ids.new_start.text)
        if self.ids.new_end.text:
            carers_info['Available End'][ca_row] = int(self.ids.new_end.text)
        
        # set up the dataframes again
        setup_dfs()
        # do the scheduling again.
        # tr_postcodes for now, later allow for variable to change based on selection of region
        carers_df, clients_df = functions.hourly_schedule(tr_postcodes, workhours, carers_df, clients_df)
        self.ids.label_reschedule.text = 'Schedule has been adjusted'
        


class ClientWindow(MDScreen):
    selected_client = 'Savannah Parker'  # default
    selected_display = 'Text'
    
    # all time slots are set to false
    active_early_morning = BooleanProperty(False)
    active_morning = BooleanProperty(False)
    active_lunch = BooleanProperty(False)
    active_afternoon = BooleanProperty(False)
    active_evening = BooleanProperty(False)
    
    # save which client has been selected
    def spinner_clicked(self, value):
        # save selection
        ClientWindow.selected_client = value  # update
        
    # save which display type has been selected    
    def checkbox_click(self, instance, value, display):
        if value == True:
            ClientWindow.selected_display = display  # global
        else:
            ClientWindow.selected_display = ''        
        
    # saves which time windows are relevant for a client internally and updates the checkboxes accordingly    
    def selectbox_click(self, instance, value, timewindow):
        if timewindow == "Early Morning":
            ClientWindow.active_early_morning = value   # get either true or false
        elif timewindow == "Morning":
            ClientWindow.active_morning = value
        elif timewindow == "Lunch":
            ClientWindow.active_lunch = value
        elif timewindow == "Tea":
            ClientWindow.active_afternoon = value
        elif timewindow == "Evening":
            ClientWindow.active_evening = value
           
        # update select boxes?    
        self.ids.early_morning.active = ClientWindow.active_early_morning
        self.ids.morning.active = ClientWindow.active_morning
        self.ids.lunch.active = ClientWindow.active_lunch
        self.ids.afternoon.active = ClientWindow.active_afternoon
        self.ids.evening.active = ClientWindow.active_evening
            
        
    def update_display(self):
        global carers_df, clients_df
    
        if ClientWindow.selected_display == 'Table':
            # clear any existing plots
            plt.cla()
            self.table_box_cl.clear_widgets()
            
            # show table box 
            self.ids.table_box_cl.size_hint = 0.9, 0.9
            
            
            # get info
            which_client = ClientWindow.selected_client
            
            # function to find client IDS here, for now:
            client_ids = []
            # go through the clients_df & add all ids that start with this?
            for i in range(len(clients_df)):
                splt_id = clients_df['id'][i].split("_")
                if splt_id[0] == which_client:
                    client_ids.append(clients_df['id'][i])
                    
            carer_ids = []
            for i in range(len(carers_df)):  # go through each row
                for cl in client_ids:  # go through each client
                    if cl in carers_df['route'][i]:
                        carer_ids.append(carers_df['id'][i])

            carer_ids = list(set(carer_ids))
                    
            visit_times2, visiter2, visit_duration2 = functions.clients_visit_info_multi(carer_ids, client_ids, carers_df, clients_df)
            

            # plot stuff
            care_col = 'lightblue'
            care_patch = patches.Patch(color=care_col, label = 'Carer Visit')
            unavail_col = 'grey'
            unavail_patch = patches.Patch(color=unavail_col, label = 'Unavailable')
            fig, ax = plt.subplots()
            
            # add patches when client cannot be visited
            for i in range(14):
                hour = i+7   # start of every hour
                busy = True   # assumes that client cannot be visited at that hour
                # go through the time windows of all client_ids:
                for cl in client_ids:
                    cl_row = clients_df[clients_df['id'] == cl].index[0]
                    # go through every time window in clients_df (only one each here, but this is future-proof)
                    for tw in clients_df['time_windows'][cl_row]:
                        # is hour within the time window:
                        if tw[0] <= hour and hour <= tw[1]:
                            busy = False   # client can be visited

                if busy:
                    square = patches.Rectangle((0, hour), 2, (1), edgecolor = 'none', facecolor = unavail_col)
                    ax.add_patch(square)


            for i in range(len(visit_times2)):
                start_time = visit_times2[i][0]+(visit_times2[i][1]/60)

                square = patches.Rectangle((0, start_time), 2, (visit_duration2[i]), edgecolor = 'none', facecolor = care_col)
                ax.add_patch(square)
                # label - only include if first one or if the next one is different
                plt.text(0.05, start_time+0.6, str(visiter2[i]), fontsize = 'small')



            # set axes
            plt.xlim(0, 2)
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off

            plt.ylim(7, 21)
            plt.gca().invert_yaxis()
            plt.locator_params(axis='y', nbins=16)  # 21-7
            plt.ylabel("Time of Day")

            plt.legend(handles=[care_patch, unavail_patch])
            plt.title('Care Plan for '+str(which_client))
            
            self.table_box_cl.add_widget(FigureCanvasKivyAgg(plt.gcf()))
            
            
            # hide text box
            self.ids.instructions_cl_text.text = ''
            self.ids.instructions_cl_text.size_hint = None, None
            self.ids.instructions_cl_text.height = '0dp'
            self.ids.instructions_cl_text.width = '0dp'            
            
            # include availabilities
            # update availability checkboxes here, based on selected carer & clients_info
            # find row in clients_info
            cl_row = clients_info[clients_info['ID'] == ClientWindow.selected_client].index[0]
            # go through the columns & update the properties as needed
            if clients_info['Early Morning'][cl_row] == "Yes":
                ClientWindow.active_early_morning = True
            elif clients_info['Early Morning'][cl_row] == "No":
                ClientWindow.active_early_morning = False
            
            if clients_info['Morning'][cl_row] == "Yes":
                ClientWindow.active_morning = True
            elif clients_info['Morning'][cl_row] == "No":
                ClientWindow.active_morning = False
            
            if clients_info['Lunch'][cl_row] == "Yes":
                ClientWindow.active_lunch = True
            elif clients_info['Lunch'][cl_row] == "No":
                ClientWindow.active_lunch = False
            
            if clients_info['Tea'][cl_row] == "Yes":
                ClientWindow.active_afternoon = True
            elif clients_info['Tea'][cl_row] == "No":
                ClientWindow.active_afternoon = False
            
            if clients_info['Evening'][cl_row] == "Yes":
                ClientWindow.active_evening = True
            elif clients_info['Evening'][cl_row] == "No":
                ClientWindow.active_evening = False
                
            # afterwards, update on the window
            self.ids.early_morning.active = ClientWindow.active_early_morning
            self.ids.morning.active = ClientWindow.active_morning
            self.ids.lunch.active = ClientWindow.active_lunch
            self.ids.afternoon.active = ClientWindow.active_afternoon
            self.ids.evening.active = ClientWindow.active_evening
            
            # update button text
            self.ids.btn_reschedule_cl.text = 'Adjust Availability'
            
    
        elif ClientWindow.selected_display == 'Text':
            # change label of Output Box to the corresponding carer
            self.ids.instructions_cl_text.size_hint = 0.9, 0.9
            # function for finding all instances of client here, for now:
            which_client = ClientWindow.selected_client  # string
            client_ids = []
            # go through the clients_df & add all ids that start with this?
            for i in range(len(clients_df)):
                splt_id = clients_df['id'][i].split("_")
                if splt_id[0] == which_client:
                    client_ids.append(clients_df['id'][i])
            
            # find all carers that visit (unique instances):
            carer_ids = []
            for i in range(len(carers_df)):  # go through each row
                for cl in client_ids:  # go through each client
                    if cl in carers_df['route'][i]:
                        carer_ids.append(carers_df['id'][i])

            carer_ids = list(set(carer_ids))
            
            
            cl_text_sched = 'Visit schedule for '+ str(ClientWindow.selected_client)+'.\n\n'
            cl_text_sched += functions.text_schedule_clients_multi(carer_ids, client_ids, carers_df, clients_df)
            self.ids.instructions_cl_text.text = cl_text_sched
            
            # make table box disappear + clear existing plots
            # clear any existing plots
            self.ids.table_box_cl.size_hint = None, None
            self.ids.table_box_cl.height = '0dp'
            self.ids.table_box_cl.width = '0dp'
            plt.cla()
            self.table_box_cl.clear_widgets()
            
            # include availabilities
            # update availability checkboxes here, based on selected carer & clients_info
            # find row in clients_info
            cl_row = clients_info[clients_info['ID'] == ClientWindow.selected_client].index[0]
            # go through the columns & update the properties as needed
            if clients_info['Early Morning'][cl_row] == "Yes":
                ClientWindow.active_early_morning = True
            elif clients_info['Early Morning'][cl_row] == "No":
                ClientWindow.active_early_morning = False
            
            if clients_info['Morning'][cl_row] == "Yes":
                ClientWindow.active_morning = True
            elif clients_info['Morning'][cl_row] == "No":
                ClientWindow.active_morning = False
            
            if clients_info['Lunch'][cl_row] == "Yes":
                ClientWindow.active_lunch = True
            elif clients_info['Lunch'][cl_row] == "No":
                ClientWindow.active_lunch = False
            
            if clients_info['Tea'][cl_row] == "Yes":
                ClientWindow.active_afternoon = True
            elif clients_info['Tea'][cl_row] == "No":
                ClientWindow.active_afternoon = False
            
            if clients_info['Evening'][cl_row] == "Yes":
                ClientWindow.active_evening = True
            elif clients_info['Evening'][cl_row] == "No":
                ClientWindow.active_evening = False
                
            # afterwards, update on the window
            self.ids.early_morning.active = ClientWindow.active_early_morning
            self.ids.morning.active = ClientWindow.active_morning
            self.ids.lunch.active = ClientWindow.active_lunch
            self.ids.afternoon.active = ClientWindow.active_afternoon
            self.ids.evening.active = ClientWindow.active_evening
            
            # update button text
            self.ids.btn_reschedule_cl.text = 'Adjust Availability'
        
        else:
            pass
    
    
    
    # change the overall schedule based on changes entered on this window
    def reschedule(self):
        # stores the selection of which time windows are available in active_evening etc
        # stores which client in selected_client
        global clients_info, carers_df, clients_df
        
        # find current client in clients_info
        cl_row = clients_info[clients_info['ID'] == ClientWindow.selected_client].index[0]
        
        # change info
        if ClientWindow.active_early_morning:  # True
            clients_info['Early Morning'][cl_row] = "Yes"
        elif not ClientWindow.active_early_morning:  # False
            clients_info['Early Morning'][cl_row] = "No"
        
        if ClientWindow.active_morning:  # True
            clients_info['Morning'][cl_row] = "Yes"
        elif not ClientWindow.active_morning:  # False
            clients_info['Morning'][cl_row] = "No"
        
        if ClientWindow.active_lunch:  # True
            clients_info['Lunch'][cl_row] = "Yes"
        elif not ClientWindow.active_lunch:  # False
            clients_info['Lunch'][cl_row] = "No"
        
        if ClientWindow.active_afternoon:  # True
            clients_info['Tea'][cl_row] = "Yes"
        elif not ClientWindow.active_afternoon:  # False
            clients_info['Tea'][cl_row] = "No"
        
        if ClientWindow.active_evening:  # True
            clients_info['Evening'][cl_row] = "Yes"
        elif not ClientWindow.active_evening:  # False
            clients_info['Evening'][cl_row] = "No"
            
        # set up dataframes
        setup_dfs()
        
        # do the scheduling again.
        # tr_postcodes for now, later allow for variable to change based on selection of region
        carers_df, clients_df = functions.hourly_schedule(tr_postcodes, workhours, carers_df, clients_df)
        self.ids.btn_reschedule_cl.text = 'Availability has been updated'


# this just needs to exist for the cccmd.kv file to have a reference for a root widget
class WindowManager(MDScreenManager):
    pass

########### Main #############

class CCCMDApp(MDApp):
    
    carer_list = ListProperty(['Carer 1', 'Carer 2', 'Carer 3'])  # hold carers (declare like this to update interface)
    client_list = ListProperty(['Client 21', 'Client 22', 'Client 23'])
    
    
    ### file select stuff  ####
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_manager_obj = MDFileManager(
            select_path = self.select_path,
            exit_manager = self.exit_manager
            #preview = True
        )
        self.filepath = ""
        
    def select_path(self, path):
        try:
            self.filepath = path
            self.read_excel_file()
        finally:
            self.exit_manager()
        
    
    def open_file_manager(self):
        self.file_manager_obj.show(os.path.expanduser("~"))  # root directory
        
    def exit_manager(self):
        self.file_manager_obj.close()
        
        
    def read_excel_file(self):
        global clients_info, carers_info
        # read excel file here (global variables)
        clients_info = pd.read_excel(self.filepath, sheet_name=0)
        carers_info = pd.read_excel(self.filepath, sheet_name=1)
        carers_info['ID'] = list(map(str, list(carers_info['ID'])))
        clients_info['ID'] = list(map(str, list(clients_info['ID'])))
        # list of ids (string)
        CCCMDApp.carer_list = list(carers_info['ID'])
        CCCMDApp.client_list = list(clients_info['ID'])
        
        # then set up global dataframes
        setup_dfs()
        
    # colour themes: https://m2.material.io/design/color/the-color-system.html#tools-for-picking-colors
        
    ### app builder ###
    def build(self):
        self.title = 'Carer Scheduling App'
        
        #self.theme_cls.theme_style = "Dark"   # dark mode
        self.theme_cls.primary_palette = "LightBlue"
        # uses new kv file, built up step-by-step
        kv = Builder.load_file('cccmd.kv')
    
        return kv
    

if __name__ == '__main__':
    CCCMDApp().run()