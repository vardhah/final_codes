import pygame
import numpy as np
import cv2

class pygameViewer():
    def __init__(self):
        pygame.display.init()
        pygame.font.init()
        self.WIDTH = 800
        self.HEIGHT = 600
        self.display = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("RL Braking System")
    
    def updateViewer(self, image):
        self.clock.tick()
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        image_rescaled = cv2.resize(array, dsize=(self.WIDTH, self.HEIGHT), interpolation=cv2.INTER_CUBIC)
        image_surface = pygame.surfarray.make_surface(image_rescaled.swapaxes(0, 1))
        self.display.blit(image_surface, (0, 0))
        pygame.display.flip()
    
    def stop(self):
        pygame.quit()