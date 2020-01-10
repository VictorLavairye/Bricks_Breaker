
import numpy as np
import numpy.random as rd
import math
import copy

def get_angle(point_1, point_2):
    return np.arctan((point_2[1] - point_1[1])/(point_2[0] - point_1[0]))

class Bricks:

    def __init__(self, height, width, position, initial_durability):
        self.height = height
        self.width = width
        # Spatial system started from the top left corner of the brick
        self.position = position
        self.coordinates = [(self.position[0], self.position[1]), (self.position[0] + self.height, self.position[1] +
                                                                   self.width)]
        self.initial_durability = initial_durability
        self.current_durability = initial_durability
        self.sprite = "sprite" #Faire un dico qui relie la durabilité aux diff"rents sprite

    def get_position(self):
        return self.position

    def set_position(self, new_position):
        self.position = new_position
        self.coordinates = self.coordinates = [(self.position[0], self.position[1]), (self.position[0] + self.height,
                                                                                      self.position[1] + self.width)]

    def get_vertices(self):
        return np.array([(self.position[0], self.position[1]), (self.position[0] + self.height, self.position[1])],
                        [(self.position[0], self.position[1] + self.width), (self.position[0] + self.height, self.position[1] + self.width)])

    def get_distance_to_vertex(self, x, vertex):
        return np.sqrt((x[0] + vertex[0])**2 + (x[1] + vertex[1])**2)

    def get_closest_vertex(self, x):
        vertices = self.get_vertices().reshape((1, 4))
        vertices_distance_to_x = []
        for vertex in vertices:
            vertices_distance_to_x.append(self.get_distance_to_vertex(x, vertex))
        vertices_distance_to_x = np.array(vertices_distance_to_x)
        return vertices_distance_to_x[np.argmin(vertices_distance_to_x)]

    def contiguous_vertices(self, vertex):
        list_of_contiguous_vertices = self.get_vertices().reshape((1, 4))
        list_of_distance_to_vertex = np.zeros((1, 4))
        for i in range(4):
            list_of_distance_to_vertex[i] = self.get_distance_to_vertex(list_of_contiguous_vertices[i], vertex)
        list_of_contiguous_vertices.delete(list_of_contiguous_vertices, np.argmax(list_of_distance_to_vertex), 1)
        list_of_contiguous_vertices.delete(list_of_contiguous_vertices, np.argmin(list_of_distance_to_vertex), 1)
        return list_of_contiguous_vertices

    def get_initial_durability(self):
        return self.initial_durability

    def get_current_durability(self):
        return self.current_durability

    def set_current_durability(self, new_durability):
        self.current_durability = new_durability

    def set_sprite(self):
        self.sprite = "new_sprite"  #Faire le lien entre le dico et la current_durability

    def is_still_durability(self):
        if self.current_durability == 0:
            return False
        else:
            return True


class Ball:

    def __init__(self, radius=10, speed=10):
        self.radius = radius
        self.position = np.array([0, 0])
        self.angle = np.transpose(np.array([0, 0]))
        self.speed = speed
        self.coordinates = np.array([self.position[0] - np.ceil(radius/2), self.position[1] - np.ceil(radius/2)],
                                    [self.position[0] + np.ceil(radius/2), self.position[1] + np.ceil(radius/2)])

    # Position is the coordinates of the center of the ball
    def get_position(self):
        return self.position

    def set_position(self, new_position):
        self.position[0] = new_position[0]
        self.position[1] = new_position[1]

    def compute_coordinates(self):
        self.coordinates = np.array([self.position[0] - np.ceil(self.radius / 2), self.position[1] - np.ceil(self.radius / 2)],
                                    [self.position[0] + np.ceil(self.radius / 2), self.position[1] + np.ceil(self.radius / 2)])

    def get_coordinates(self):
        return self.coordinates

    def get_angle(self):
        return self.angle

    def set_angle(self, new_angle):
        self.angle = new_angle

    def get_speed(self):
        return self.speed

    def set_speed(self, new_speed):
        self.speed = new_speed

    def compute_next_position(self):
        dx = np.cos(self.angle[0]) * self.speed
        dy = np.sin(self.angle[0]) * self.speed
        self.position[0] += dx
        self.position[1] += dy
        self.compute_coordinates()

    def get_radius(self):
        return self.radius

    def set_radius(self, new_radius):
        self.radius = new_radius
        self.compute_coordinates()


class Bar:

    def __init__(self, position=(0, 0), height=50, width=10, bar_speed=10):
        self.height = height
        self.width = width
        self.position = position
        self.coordinates = np.array([self.position[0], self.position[1]],
                                    [self.position[0] + self.height, self.position[1] + self.width])
        self.bar_speed = bar_speed

    def get_position(self):
        return self.position

    def set_position(self, new_position):
        self.position[0] = new_position[0]
        self.position[1] = new_position[1]

    def compute_coordinates(self):
        self.coordinates = np.array([self.position[0], self.position[1]],
                                    [self.position[0] + self.height, self.position[1] + self.width])

    def get_coordinates(self):
        return self.coordinates

    def get_bar_height(self):
        return self.height

    def set_bar_height(self, new_height):
        self.height = new_height
        self.compute_coordinates()

    def get_bar_width(self):
        return self.width

    def set_bar_width(self, new_width):
        self.width = new_width
        self.compute_coordinates()

    def get_bar_speed(self):
        return self.bar_speed

    def set_bar_speed(self, new_speed):
        self.bar_speed = new_speed

    #for instance, the speed of the bar is instantaneous, with no
    # movement = + 1 if the bar goes to the right, - 1 to the left
    def compute_next_position(self, movement):
        self.position = (self.position[0] + movement * self.bar_speed)
        self.compute_coordinates()


class Environment:
    # Bar properties
    standard_bar_height = 20
    standard_bar_width = 2
    standard_bar_speed = 10

    # Bricks properties
    standard_bricks_height = 20
    standard_bricks_width = 20

    def __init__(self, bricks_height=standard_bricks_height, bricks_width=standard_bricks_width,
                 bar_height=standard_bar_height, bar_width=standard_bar_width, display_screen=False):
        #Canvas resolution
        self.canvas_height = 0
        self.canvas_width = 0
        #Bricks
        self.nb_bricks_per_line = 0
        self.nb_bricks_per_column = 0
        self.list_of_bricks = []
        self.bricks_height = bricks_height
        self.bricks_width = bricks_width
        self.boolean_grid = []
        #Balls
        self.list_of_balls = []
        #Bar
        self.list_of_bars = [Bar()] * 3
        self.bar_height = bar_height
        self.bar_width = bar_width
        #Collision
        self.horizontal_collision = False
        self.vertical_collision = False
        #Other
        self.life = 3
        self.level = 1
        self.timer = 0
        if display_screen is True:
            print("Afficher l'écran")

    def load_level(self, level_no):
        self.level = level_no
        self.list_of_bricks = []
        try:
            file = open('level'+str(self.level)+'.txr', "r")
            line = file.readline()
            line_memo = line.replace("\n", "").split(",")
            self.nb_bricks_per_line = int(line_memo[0])
            self.nb_bricks_per_column = int(line_memo[1])
            self.canvas_height = self.nb_bricks_per_line * self.bricks_height
            self.canvas_width = self.nb_bricks_per_column* self.bricks_width + 3 * self.bar_width
            level_content = [[0] * self.nb_bricks_per_line] * self.nb_bricks_per_column
            self.boolean_grid = [[False] * self.nb_bricks_per_line] * self.nb_bricks_per_column
            i = 0
            while line:
                line = file.readline()
                level_content[i] = list(map(int, line.replace("\n", "").split(",")))
                i += 1
            file.close()
            for l in range(0, len(level_content)):
                for c in range(0, len(level_content[l])):
                    if level_content[l][c] > 0:
                        new_brick = Bricks((l * self.bricks_height, c * self.bricks_width), level_content[l][c])
                        self.list_of_bricks.append(new_brick)
                        self.boolean_grid[l][c] = True

            self.reset()
        except IOError:
            #A modifier
            print("The loading of the level is not possible")

    #method that will be use after loading a level and after failure (loss of a life)
    def reset(self):
        standard_bar_x = np.floor((self.canvas_height - self.bar_height) / 2)
        standard_bar_y = self.canvas_width - 2 * self.bar_width
        self.list_of_bars[0] = Bar((standard_bar_x - self.canvas_height, standard_bar_y))
        self.list_of_bars[1] = Bar((standard_bar_x, standard_bar_y))
        self.list_of_bars[2] = Bar((standard_bar_x + self.canvas_height, standard_bar_y))
        self.list_of_balls = [Ball().set_position((standard_bar_x + np.floor(self.bar_height/2), standard_bar_y -
                                                   self.list_of_balls[0].get_radius()))]
        self.list_of_balls[0].set_angle(math.radians(90))

    #It gives the indexes of the ball on the grid
    def get_ball_grid_indexes(self, ball):
        searching_line = True
        searching_column = True
        k_line = 0
        k_column = 0
        while searching_line:
            while searching_column:
                if (ball.get_position()[1] >= k_column * self.bricks_width) and (ball.get_position()[1] <=
                                                                                    (k_column + 1) * self.bricks_width):
                    searching_column = False
                else:
                    k_column += 1
            if (ball.get_position()[0] >= k_line * self.bricks_height) and (ball.get_position()[0] <=
                                                                                (k_line + 1) * self.bricks_height):
                searching_line = False
            else:
                k_line += 1
        return k_line, k_column

    #Is ne the next collision an horizontal one or a vertical one ?
    def next_collision_type(self):
        for ball in self.list_of_balls:
            k_line, k_column = self.get_ball_grid_indexes(ball)
            if (ball.get_angle() > 0) and (ball.get_angle() <= math.radians(90)):
                k_line_boundary = (k_line, self.nb_bricks_per_line)
                k_column_boundary = (k_column, self.nb_bricks_per_column)
            if (ball.get_angle() > math.radians(90)) and (ball.get_angle() < math.radians(180)):
                k_line_boundary = (0, k_line + 1)
                k_column_boundary = (k_column, self.nb_bricks_per_column)
            if (ball.get_angle() >= math.radians(-90)) and (ball.get_angle() < 0):
                k_line_boundary = (k_line, self.nb_bricks_per_line)
                k_column_boundary = (0, k_column + 1)
            if (ball.get_angle() > math.radians(-90)) and (ball.get_angle() < math.radians(-180)):
                k_line_boundary = (0, k_line + 1)
                k_column_boundary = (0, k_column + 1)
            counter_line = k_line_boundary[0]
            counter_column = k_column_boundary[0]
            euclidian_distance_to_grid_line = []
            euclidian_distance_to_grid_column = []
            while counter_line <= k_line_boundary[1]:
                while counter_column <= k_column_boundary[1]:
                    if self.boolean_grid[counter_line][counter_column] is True:




                        brick_test = Bricks(self.bricks_height, self.bricks_width, (counter_line, counter_column), 0)
                        closest_vertex = brick_test.get_closest_vertex(ball.get_position())
                        contiguous_vertices = brick_test.contiguous_vertices(closest_vertex)
                        angles_interval = []
                        closest_vertex_angle = get_angle(ball.get_position(), closest_vertex)
                        for vertex in contiguous_vertices:
                            contiguous_vertex_angle = get_angle(ball.get_position(), vertex)
                            angles_interval.append([min(closest_vertex_angle, contiguous_vertex_angle),
                                                    max(closest_vertex_angle, contiguous_vertex_angle)])
                        if angles_interval[0][0] <= ball.get_angle() <= angles_interval[0][1]:
                            if contiguous_vertices[0][0] - closest_vertex[0] == 0:
                                self.horizontal_collision = True
                                self.vertical_collision = False
                            else:
                                self.horizontal_collision = False
                                self.vertical_collision = True
                        if angles_interval[1][0] < ball.get_angle() <= angles_interval[1][1]:
                            if contiguous_vertices[1][0] - closest_vertex[0] == 0:
                                self.horizontal_collision = True
                                self.vertical_collision = False
                            else:
                                self.horizontal_collision = False
                                self.vertical_collision = True
                        else:
                            raise Exception

    def rebound(self):
        for ball in self.list_of_balls:
            # Bar collision
            for bar in self.list_of_bars:
                if (ball.get_coordinates()[1][1] >= bar.get_coordinates()[0][1]) and (ball.get_coordinates()[1][0] <=
                                                                                      bar.get_coordinates()[0][0]) and (
                        ball.get_coordinates()[0][0] >= bar.get_coordinates()[1][0]):
                    if ball.get_position() > bar.get_position()[0] + np.floor((bar.get_coordinates()[1][0] -
                                                                               bar.get_coordinates()[0][0]) / 2):
                        new_angle = np.arcsin((ball.get_coordinates()[1][0] - np.floor(bar.get_position()[1][0] / 2)) /
                                              (bar.get_coordinates()[1][0] - bar.get_coordinates()[0][1]))
                    else:
                        new_angle = np.arcsin((ball.get_coordinates()[0][0] - np.ceil(bar.get_position()[1][0] / 2)) /
                                              (bar.get_coordinates()[1][0] - bar.get_coordinates()[0][1]))
                    ball.set_angle(new_angle)

            #Brick collision



            #Border collision
            if ball.get_coordinates()[0][1] <= 0:
                new_angle = - ball.get_angle()
            if (ball.get_coordinates()[0][0] <= 0) or (ball.get_coordinates()[1][0] >= self.canvas_height):
                new_angle = - ball.get_angle() + math.radians(180)

            ball.set_angle(new_angle)