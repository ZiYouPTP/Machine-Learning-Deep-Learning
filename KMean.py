import pygame
from random import randint
import math
from sklearn.cluster import KMeans

def distance(p1,p2):
	return math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]))

pygame.init()

screen = pygame.display.set_mode((1200,700))

pygame.display.set_caption("kmeans visualization")

running = True

clock = pygame.time.Clock()

#color
BACKGROUND = (214,214,214)
BACKGROUND_PANEL = (249,255,230)
BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
YELLOW = (230,230,0)
PURPLE = (255,0,255)
SKY = (0,255,255)
ORANGE = (255,125,25)
GRAPE = (100,25,125)
GRASS = (55,155,65)

COLORS = [RED,GREEN,BLUE, YELLOW,PURPLE, SKY, ORANGE,GRAPE,GRASS]

#rectangle
RECTANGLE = (50,50,700,500)
#rectangle background
RECT_BG = (55,55,690,490)

font = pygame.font.SysFont('sans',40)
aria = pygame.font.SysFont('arial',15)

plus_pos = (860,50)
PLUS = font.render('+', True, WHITE)
plus_box = PLUS.get_rect()
plus_box_draw = (plus_pos[0] - 10, plus_pos[1] + 5 , plus_box[2] + 20 , plus_box [3] -10)

sub_pos = (960,50)
SUB = font.render('-', True, WHITE)
sub_box = SUB.get_rect()
sub_box_draw = (sub_pos[0] - 13, sub_pos[1] + 5 , sub_box[2] + 26 , sub_box [3] -10)

run_pos = (900,150)
RUN = font.render('RUN', True, WHITE)
run_box = RUN.get_rect()
run_box_draw = (run_pos[0] - 10, run_pos[1] + 5 , run_box[2] + 20 , run_box [3] -10)

random_pos = (850,250)
RANDOM = font.render('RANDOM', True, WHITE)
random_box = RANDOM.get_rect()
random_box_draw = (random_pos[0] - 10, random_pos[1] + 5 , random_box[2] + 20 , random_box [3] -10)


algorithm_pos = (850,450)
ALGORITHM = font.render('ALGORITHM', True, WHITE)
algorithm_box = ALGORITHM.get_rect()
algorithm_box_draw = (algorithm_pos[0] - 10, algorithm_pos[1] + 5 , algorithm_box[2] + 20 , algorithm_box [3] -10)

reset_pos = (850,550)
RESET = font.render('RESET', True, WHITE)
reset_box = RESET.get_rect()
reset_box_draw = (reset_pos[0] - 10, reset_pos[1] + 5 , reset_box[2] + 20 , reset_box [3] -10)

pos_K = (1050,50)
pos_error = (850,350)

K = 0 
error = 0
points = []
clusters = []
labels = []


while running:
	clock.tick(60)
	screen.fill(BACKGROUND)
	mouse_x, mouse_y = pygame.mouse.get_pos()

	#draw interface
	#draw panel
	pygame.draw.rect(screen,BLACK,RECTANGLE)
	pygame.draw.rect(screen,BACKGROUND_PANEL,RECT_BG)
	
	#plus 
	pygame.draw.rect(screen,BLACK,plus_box_draw)
	screen.blit(PLUS,plus_pos)

	#subtract
	pygame.draw.rect(screen,BLACK,sub_box_draw)
	screen.blit(SUB,sub_pos)

	#run
	pygame.draw.rect(screen,BLACK,run_box_draw)
	screen.blit(RUN,run_pos)

	#random
	pygame.draw.rect(screen,BLACK,random_box_draw)
	screen.blit(RANDOM,random_pos)

	#algorithm
	pygame.draw.rect(screen,BLACK,algorithm_box_draw)
	screen.blit(ALGORITHM,algorithm_pos)

	#reset
	pygame.draw.rect(screen,BLACK,reset_box_draw)
	screen.blit(RESET,reset_pos)

	#K VALUE
	text_K = font.render("K = " + str(K), True, BLACK)
	screen.blit(text_K,pos_K)

	#pos mouse
	pos_mouse = aria.render("(" + str(mouse_x - RECT_BG[0]) + "-" + str(mouse_y - RECT_BG[1]) + ")",True , BLACK ) 
	if RECT_BG[0] < mouse_x < (RECT_BG[0] + RECT_BG[2]) and RECT_BG[1] < mouse_y < (RECT_BG[1] + RECT_BG[3]):
		screen.blit(pos_mouse,(mouse_x + 5,mouse_y -10))

	


	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False
		if event.type == pygame.MOUSEBUTTONDOWN:
			#create point on panel
			if RECT_BG[0] < mouse_x < (RECT_BG[0] + RECT_BG[2]) and RECT_BG[1] < mouse_y < (RECT_BG[1] + RECT_BG[3]):
				point = [mouse_x - RECT_BG[0],mouse_y - RECT_BG[1]]
				points.append(point)
				labels = []

			#change button +
			if (plus_box_draw[0]) < mouse_x < (plus_box_draw[0] + plus_box_draw[2]) and (plus_box_draw[1]) < mouse_y < (plus_box_draw[1] + plus_box_draw[3]):
				if K < len(COLORS):
					K = K + 1
			#change button -
			if (sub_box_draw[0]) < mouse_x < (sub_box_draw[0] + sub_box_draw[2]) and (sub_box_draw[1]) < mouse_y < (sub_box_draw[1] + sub_box_draw[3]):
				if(K != 0):
					K = K - 1
			#run button
			if (run_box_draw[0]) < mouse_x < (run_box_draw[0] + run_box_draw[2]) and (run_box_draw[1]) < mouse_y < (run_box_draw[1] + run_box_draw[3]):
				labels = []

				if clusters == []:
					continue
				#assign point to closet cluster
				for p in points:
					distances_to_cluster = []
					for c in clusters:
						dis = distance(p,c)
						distances_to_cluster.append(dis)
					min_distance = min(distances_to_cluster)
					label = distances_to_cluster.index(min_distance)
					labels.append(label)

				#update cluster
				for i in range(K):
					sum_x = 0 
					sum_y = 0
					count = 0
					for j in range (len(points)):
						if labels[j] == i:
							sum_x += points[j][0]
							sum_y += points[j][1]
							count += 1
					if count != 0:
						new_cluster_x = sum_x/count
						new_cluster_y = sum_y/count
						clusters[i] = [new_cluster_x,new_cluster_y]

			#random button
			if (random_box_draw[0]) < mouse_x < (random_box_draw[0] + random_box_draw[2]) and (random_box_draw[1]) < mouse_y < (random_box_draw[1] + random_box_draw[3]):
				labels = []
				clusters = []
				for i in range(K):
					random_point = [randint(0,RECT_BG[2]), randint(0,RECT_BG[3])]
					clusters.append(random_point)
			#algorithm button
			if (algorithm_box_draw[0]) < mouse_x < (algorithm_box_draw[0] + algorithm_box_draw[2]) and (algorithm_box_draw[1]) < mouse_y < (algorithm_box_draw[1] + algorithm_box_draw[3]):
				try:
					kmeans = KMeans(n_clusters=K).fit(points)
					labels = kmeans.labels_.tolist()
					clusters = kmeans.cluster_centers_.tolist()
				except:
					continue
			#reset button
			if (reset_box_draw[0]) < mouse_x < (reset_box_draw[0] + reset_box_draw[2]) and (reset_box_draw[1]) < mouse_y < (reset_box_draw[1] + reset_box_draw[3]):
				points = []
				labels = []
				clusters = []
				K = 0
	#draw cluster
	for i in range(len(clusters)):
		num = i % len(COLORS)
		pygame.draw.circle(screen,COLORS[num],(int(clusters[i][0]) + RECT_BG[0],int(clusters[i][1]) + RECT_BG[1]),10)

	#draw point
	for i in range(len(points)):
		pygame.draw.circle(screen,BLACK,(points[i][0]+RECT_BG[0],points[i][1]+RECT_BG[1]),6)

		if labels == []:
			pygame.draw.circle(screen,BACKGROUND_PANEL,(points[i][0]+RECT_BG[0],points[i][1]+RECT_BG[1]),5)
		else:
			pygame.draw.circle(screen,COLORS[labels[i]],(points[i][0]+RECT_BG[0],points[i][1]+RECT_BG[1]),5)

	#calculate and draw error
	error = 0 
	if clusters != [] and labels != []:
		for i in range(len(points)):
			error +=distance(points[i],clusters[labels[i]])
	#error
	text_error = font.render("Error = " + str(int(error)),True, BLACK)
	screen.blit(text_error,pos_error)

	pygame.display.flip()

pygame.quit()