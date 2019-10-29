import cv2
import numpy as np


from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class DirectionTracker:

	def __init__(self):
		self.init_color()
		self.list_inside = []
		self.memory = {}
		self.maju = np.zeros((11,), dtype=int)
		self.mundur = np.zeros((11,), dtype=int)
		self.alreadyCounted = []
		self.vehicle = ['articulated truck','bicycle','bus','car','motorcycle','motorized vehicle','non-motorized vehicle','pedestrian','pickup truck','single unit truck','work van']

	def init_color(self):
		np.random.seed(42)
		self.COLORS = np.random.randint(0, 255, size=(200, 3),dtype="uint8")

	def is_in_list(self,points,x):
		i = 0
		for i in range(len(points)):
			if(points[i][0]==x):
				return True,i
		return False,i+1

	def is_inside(self,dot,roi):
		point = Point(dot)
		polygon = Polygon(roi)

		return polygon.contains(point)

	def vehicle_track(self,frame,tracks,roi,classes):
		boxes = []
		indexIDs = []
		previous = self.memory.copy()
		self.memory = {}

		for track in tracks:
			boxes.append([track[0], track[1], track[2], track[3]])
			indexIDs.append(int(track[4]))
			self.memory[indexIDs[-1]] = boxes[-1]


		if len(boxes) > 0:

			#DI MODUL INI BISA LIAT ARAHNYA, BANDINGIN XYWH sama X2Y2W2H2. dibandingin kalau x nya positif berati naik. and vice versa
			i = int(0)
			for box in boxes:
		
				(x, y) = (int(box[0]), int(box[1]))
				(w, h) = (int(box[2]), int(box[3]))

				color = [int(c) for c in self.COLORS[indexIDs[i] % len(self.COLORS)]]
				cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color[::-1], 2)

				if indexIDs[i] in previous:
					previous_box = previous[indexIDs[i]]
					
					(x2, y2) = (int(previous_box[0]), int(previous_box[1]))
					(w2, h2) = (int(previous_box[2]), int(previous_box[3]))
					
					p0 = (int(x + (w-x)/2), int(y + (h-y)/2)) 
					p1 = (int(x2 + (w2-x2)/2),int(y2 + (h2-y2)/2))

					dot_point = [p0,p1]
					print("DOTPOINT",dot_point)
					if(self.is_inside([p0,p1],roi)):
						ret,retId = self.is_in_list(self.list_inside,indexIDs[i])
						if(ret):
							self.list_inside[retId][1].append(p1)
							frame = self.draw_line(frame,retId,self.list_inside) #draw line for direction
						else:
							self.list_inside.append([indexIDs[i],[p0],color])
			
					else:
						ret,retId = self.is_in_list(self.list_inside,indexIDs[i])
						if(ret):
						  self.list_inside[retId][1].append(p0)

					if indexIDs[i] not in self.alreadyCounted:
						self.alreadyCounted.append(indexIDs[i])
						direction = p0[1] - np.mean(p0)

						boo = True
						j=0
						while boo :
							if classes[i] == self.vehicle[j]:
								print(classes[i])
								print(self.vehicle[j])
								if(direction>0):
									self.mundur[j]=self.mundur[j]+1
									boo = False
								else:
									self.maju[j] = self.maju[j]+1
									boo = False
							else:
								j= j+1

				frame = self.drawNumber(frame,indexIDs[i],x,y,color)	  
				i += 1

		# print("MAJUU",self.maju)
		# print("MUNDURRR",self.mundur)	

	def draw_line(self,frame,idx,lists):
		for point in range(len(lists[idx][1])):
		  	try:
		  		frame = cv2.line(frame, lists[idx][1][point], lists[idx][1][point+1], lists[idx][2], 2)
		  	except:
		  		continue

		return frame

	def drawNumber(self,frame,ids,x,y,color):

		text = "{}".format(ids)
		cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

		return frame