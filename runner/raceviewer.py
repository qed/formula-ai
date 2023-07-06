import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import math
import numpy as np
from scipy.interpolate import make_interp_spline    
import PySimpleGUI as sg

from core.src import jsoner
from core.src.race import RaceData
from core.src.track import TileType, TrackField, TileCell


class CarElement:
    def __init__(self, graph, scale):
        self.graph = graph
        self.scale = scale
        self.radius = scale/2

    def show_step(self, step_data):
        position_x, position_y, angle = Viewer.get_step_fields(step_data)
        self.circle_figure = self.graph.DrawCircle(
            [position_x * self.scale, position_y * self.scale], 
            self.radius, 
            fill_color='blue', 
            line_color='blue'
        )

        angle_x = math.cos(angle) + position_x
        angle_y = math.sin(angle) + position_y
        self.angle_figure = self.graph.DrawLine(
            [position_x * self.scale, position_y * self.scale],
            [angle_x * self.scale, angle_y * self.scale],
            color = 'white', 
            width = 2
        )

    def move_to(self, step_data):
        position_x, position_y, angle = Viewer.get_step_fields(step_data)
        self.graph.delete_figure(self.circle_figure)
        self.circle_figure = self.graph.DrawCircle(
            [position_x * self.scale, position_y * self.scale], 
            self.radius, 
            fill_color='blue', 
            line_color='blue'
        )

        self.graph.delete_figure(self.angle_figure)
        angle_x = math.cos(angle) + position_x
        angle_y = math.sin(angle) + position_y
        self.angle_figure = self.graph.DrawLine(
            [position_x * self.scale, position_y * self.scale],
            [angle_x * self.scale, angle_y * self.scale],
            color = 'white', 
            width = 2
        )


class Viewer:
    def __init__(self,  track_field: TrackField, race_data: RaceData):
        self.track_field = track_field
        self.race_data = race_data
        self.init_config()
    
        self.interpolate_data()
        self.init_components()

    def init_config(self):
        self.window_width = 1600           
        self.window_height = 900	
        self.scale = 20
        self.step_rate = 1
        self.show_cell = True
        
        self.road_color = 'green'
        self.shoulder_color = 'yellow'
        self.wall_color = 'red'
        self.start_color = 'white'
        self.finish_color = 'lightblue'

    def init_components(self):
        track_info = self.track_field.track_info

        scale_x = int(self.window_width / (track_info.column + 2))
        scale_y = int(self.window_height / (track_info.row + 2))
        if scale_x < scale_y:
            self.scale = scale_x
        else:
            self.scale = scale_y
        
        self.text_step = int(20.0/self.scale+.4999)
        if self.text_step < 1:
            self.text_step = 1

        self.window_width = self.scale * (track_info.column + 2)
        self.window_height = self.scale * (track_info.row + 2)

        self.graph = sg.Graph(
            canvas_size=(self.window_width, self.window_height), 
            graph_bottom_left=(-self.scale, self.window_height-self.scale), 
            graph_top_right=(self.window_width-self.scale, -self.scale), 
            background_color='white', 
            key='graph', 
            tooltip='track field')
        
        table = sg.Table(
            values=self.table_data, 
            headings=['sec', 'distance', 'acceleration', 'angular velocity', 'x', 'y', 'angle', 'v_forward', 'v_right'], 
            key='table',
            num_rows=10,
            display_row_numbers=True,
            justification = "right",
            background_color ='gray',
            alternating_row_color='teal',
            )

        self.layout = [
            [sg.Text(race_data.race_info.id, text_color='white', font=('Helvetica', 25))],
            [self.graph],
            [sg.Text('0'), sg.ProgressBar(max_value=self.steps_data.shape[0], orientation='h', size=(20, 20), key='progress_bar'), sg.Text(self.steps_data.shape[0])],
            [sg.Button('Play'), sg.Button('-'), sg.Text('0', key='at_step'), sg.Button('+'), sg.Exit()],
            [table],
            ]
        
        self.window = sg.Window(
            'Race Viewer', 
            self.layout, 
            element_justification='center', 
            grab_anywhere=True, 
            finalize=True
        )

        self.draw_track_field()
        self.car_element = CarElement(self.graph, self.scale)

    def draw_x_coordinate(self, y):

        for x in range(0, self.track_field.track_info.column, self.text_step) :
            self.graph.DrawRectangle(
                (x * self.scale, y * self.scale), 
                (x * self.scale + self.scale, y * self.scale + self.scale), 
                fill_color = 'lightgray', 
                line_color = 'gray', 
                line_width = 1)
            
            self.graph.DrawText(
                x, 
                (x * self.scale, (y + 0.5) * self.scale), 
                color='black', 
                font=('Helvetica', 10))

    def draw_y_coordinate(self, x):
        for y in range(0, self.track_field.track_info.row, self.text_step) :
            self.graph.DrawRectangle(
                (x * self.scale, y * self.scale), 
                (x * self.scale + self.scale, y * self.scale + self.scale), 
                fill_color = 'lightgray', 
                line_color = 'gray', 
                line_width = 1)
            
            self.graph.DrawText(
                y, 
                ((x+0.5) * self.scale, y * self.scale), 
                color='black', 
                font=('Helvetica', 10))

    def draw_track_field(self):
        for y in range(self.track_field.track_info.row) :
            for x in range(self.track_field.track_info.column) :
                if self.track_field.field[y, x]['type'] == TileType.Wall.value:
                    tile_color = self.wall_color
                elif self.track_field.field[y, x]['type'] == TileType.Road.value:
                    tile_color = self.road_color
                elif self.track_field.field[y, x]['type'] == TileType.Shoulder.value:
                    tile_color = self.shoulder_color

                cell = TileCell(y, x)
                if self.track_field.is_start(cell):
                    tile_color = self.start_color
                elif self.track_field.is_finish(cell):
                    tile_color = self.finish_color
                
                self.graph.DrawRectangle(
                    (x * self.scale, y * self.scale), 
                    (x * self.scale + self.scale, y * self.scale + self.scale), 
                    fill_color = tile_color, 
                    line_color = 'gray', 
                    line_width = 1)

        
        if self.show_cell:
            for y in range(0, self.track_field.track_info.row, self.text_step) :
                for x in range(0, self.track_field.track_info.column, self.text_step) :
                    self.graph.DrawText(
                            self.track_field.field[y, x]['distance'], 
                            ((x+.5) * self.scale, (y+.5) * self.scale), 
                            color='black', 
                            font=('Helvetica', 10))
                    
        self.draw_x_coordinate(-1)
        self.draw_x_coordinate(self.track_field.track_info.row)
        self.draw_y_coordinate(-1)
        self.draw_y_coordinate(self.track_field.track_info.column)


    def interpolate_data(self):
        steps = self.race_data.steps
        data = np.empty((len(steps), 9))
        for i, entry in enumerate(steps):
            data[i, 0] = entry.car_state.timestamp/1000.0
            if entry.action is not None:
                data[i,1:3] = entry.action.forward_acceleration, entry.action.angular_velocity
            else:
                data[i,1:3] = 0, 0
            data[i,3:6] = entry.car_state.position.x, entry.car_state.position.y, entry.car_state.wheel_angle
            data[i,6] = entry.car_state.track_state.tile_total_distance
            data[i,7] = entry.car_state.track_state.velocity_forward
            data[i,8] = entry.car_state.track_state.velocity_right

        if self.step_rate == 1:
            self.steps_data = data
        else:
            new_data = np.linspace(0, len(steps) - 1, len(steps) * self.step_rate)
            self.steps_data = make_interp_spline(np.arange(len(steps)), data)(new_data)

        self.table_data = [[j for j in range(self.steps_data.shape[1])] for i in range(self.steps_data.shape[0])]
        for row in range(self.steps_data.shape[0]):
            self.table_data[row][0] = "{:.1f}".format(self.steps_data[row, 0])
            self.table_data[row][1] = "{:.0f}".format(self.steps_data[row, 6])
            self.table_data[row][2] = "{:.3f}".format(self.steps_data[row, 1])
            self.table_data[row][3] = "{:.3f}".format(self.steps_data[row, 2])   
            self.table_data[row][4] = "{:.3f}".format(self.steps_data[row, 3])
            self.table_data[row][5] = "{:.3f}".format(self.steps_data[row, 4])
            self.table_data[row][6] = "{:.3f}".format(self.steps_data[row, 5])
            self.table_data[row][7] = "{:.3f}".format(self.steps_data[row, 7])
            self.table_data[row][8] = "{:.3f}".format(self.steps_data[row, 8])         

    @classmethod
    def get_step_fields(cls, step_data):
        position_x = step_data[3]
        position_y = step_data[4]
        angle = step_data[5]    
        return position_x, position_y, angle
    
    def run(self):
        if (self.steps_data.shape[0] > 0):
            self.car_element.show_step(self.steps_data[0])
        
        at:int = 0
        play:bool = False
        while True:
            event, values = self.window.read(timeout=100/self.step_rate)
            if event == sg.WIN_CLOSED or event == 'Exit':
                break

            if event == '+':
                play = False
                at += 1
                if at < self.steps_data.shape[0]:
                    self.car_element.move_to(self.steps_data[at])
                else:
                    at = 0
            
            if event == '-':
                play = False
                at -= 1
                if at >= 0:
                    self.car_element.move_to(self.steps_data[at])
                else:
                    at = self.steps_data.shape[0] - 1
            
            if event == 'Play':
                play = True
            
            if play:
                at += 1
                if at < self.steps_data.shape[0]:
                    self.car_element.move_to(self.steps_data[at])
                else:
                    at = 0
                    play = False

            self.window['progress_bar'].update(at)
            self.window['at_step'].update(at)

        self.window.close()

if __name__ == "__main__":
    data_folder = sg.popup_get_folder(
        'Select race data folder', 
        default_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),'data')
    )

    race_data, track_field = jsoner.RaceSaver.load_folder(data_folder)
    if race_data is None or track_field is None:
        sg.popup_error('Error loading data', 'Please choose a correct race data folder')
        exit()  

    view = Viewer(track_field, race_data)
    view.run()