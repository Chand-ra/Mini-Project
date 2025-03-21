import pygame
import time
import cv2
import numpy as np

class Animator:
    def __init__(self, graph, start, end, preprocessor=None):
        self.G = graph
        self.start_node, self.end_node = start, end
        self.preprocessor = preprocessor

    def _common_setup(self, algorithm):
        """Shared setup between animation and video recording"""
        G, start_node, end_node = self.G, self.start_node, self.end_node
        
        # Preprocessing phase
        preprocessing_time = 0.0
        if self.preprocessor is not None:
            start_preprocess = time.time()
            self.preprocessor._select_landmarks()
            self.preprocessor._precompute_distances()
            preprocessing_time = (time.time() - start_preprocess)

        # Algorithm execution phase
        algo_start = time.time()
        visited_edges, optimal_path = algorithm(
            G, start_node, end_node, *([self.preprocessor] if self.preprocessor else [])
        )
        algo_time = time.time() - algo_start

        # Calculate node positions
        nodes = list(G.nodes(data=True))
        xs = [data['x'] for _, data in nodes]
        ys = [data['y'] for _, data in nodes]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        screen_width, screen_height = 1280, 720
        node_pos = {}
        for node, data in nodes:
            norm_x = (data['x'] - min_x) / (max_x - min_x)
            norm_y = (data['y'] - min_y) / (max_y - min_y)
            screen_x = int(norm_x * screen_width)
            screen_y = int((1 - norm_y) * screen_height)
            node_pos[node] = (screen_x, screen_y)

        # Precompute edge coordinates
        visited_edges_screen = []
        for u, v in visited_edges:
            if u in node_pos and v in node_pos:
                visited_edges_screen.append((node_pos[u], node_pos[v]))

        optimal_path_edges_screen = []
        for i in range(len(optimal_path) - 1):
            u, v = optimal_path[i], optimal_path[i+1]
            if u in node_pos and v in node_pos:
                optimal_path_edges_screen.append((node_pos[u], node_pos[v]))

        # Create surfaces
        optimal_path_surface = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
        optimal_path_surface.fill((0, 0, 0, 0))
        for start, end in optimal_path_edges_screen:
            pygame.draw.aaline(optimal_path_surface, (50, 205, 50), start, end)

        # Animation timing
        speed_factor = 100
        animation_duration = algo_time * speed_factor

        return {
            'screen_size': (screen_width, screen_height),
            'node_pos': node_pos,
            'visited_edges_screen': visited_edges_screen,
            'optimal_path_surface': optimal_path_surface,
            'animation_duration': animation_duration,
            'preprocessing_time': preprocessing_time,
            'algo_time': algo_time,
            'T': algo_time  # Backward compatibility
        }

    def animate_path(self, algorithm):
        """Run interactive animation"""
        setup = self._common_setup(algorithm)
        screen_width, screen_height = setup['screen_size']
        
        # Pygame initialization
        pygame.init()
        screen = pygame.display.set_mode((screen_width, screen_height), 
                                        pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Pathfinding Animation")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Arial", 24)

        # Run main animation loop
        self._run_animation_loop(
            setup=setup,
            screen=screen,
            clock=clock,
            font=font,
            save_video=False
        )
        pygame.quit()

    def save_animation(self, algorithm, filename, fps=30):
        """Save animation to video file"""
        setup = self._common_setup(algorithm)
        screen_width, screen_height = setup['screen_size']
        
        # Video writer setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(filename, fourcc, fps, 
                                      (screen_width, screen_height))
        
        # Pygame initialization
        pygame.init()
        screen = pygame.display.set_mode((screen_width, screen_height))
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Arial", 24)

        # Run animation and capture frames
        self._run_animation_loop(
            setup=setup,
            screen=screen,
            clock=clock,
            font=font,
            save_video=True,
            video_writer=video_writer,
            fps=fps
        )
        
        video_writer.release()
        pygame.quit()

    def _run_animation_loop(self, setup, screen, clock, font, 
                           save_video=False, video_writer=None, fps=30):
        """Shared animation loop logic"""
        current_frame = 0
        animation_start_time = time.time()
        finish_time = None
        running = True

        # Pre-render static elements
        screen_width, screen_height = setup['screen_size']
        background = pygame.Surface((screen_width, screen_height))
        background.fill((255, 255, 255))
        for u, v, _ in self.G.edges(data=True):
            if u in setup['node_pos'] and v in setup['node_pos']:
                pygame.draw.aaline(background, (200, 200, 200), 
                                 setup['node_pos'][u], setup['node_pos'][v])

        visited_edges_surface = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
        visited_edges_surface.fill((0, 0, 0, 0))

        end_delay = 1.0  # seconds to show final frame

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Animation progress
            current_real_time = time.time() - animation_start_time
            progress = min(current_real_time / setup['animation_duration'], 1.0)
            target_steps = int(progress * len(setup['visited_edges_screen']))
            steps_this_frame = max(target_steps - current_frame, 0)

            # Update visited edges surface
            if steps_this_frame > 0:
                steps_this_frame = min(steps_this_frame, 
                                     len(setup['visited_edges_screen']) - current_frame)
                for i in range(steps_this_frame):
                    start_pt, end_pt = setup['visited_edges_screen'][current_frame + i]
                    pygame.draw.aaline(visited_edges_surface, (0, 0, 255), start_pt, end_pt)
                current_frame += steps_this_frame

            # Render frame
            screen.blit(background, (0, 0))
            if current_frame < len(setup['visited_edges_screen']):
                screen.blit(visited_edges_surface, (0, 0))
            else:
                screen.blit(setup['optimal_path_surface'], (0, 0))
                if finish_time is None:
                    finish_time = time.time() - animation_start_time

            # Draw start/end nodes
            pygame.draw.circle(screen, (255, 0, 0), 
                             setup['node_pos'][self.start_node], 4)
            pygame.draw.circle(screen, (255, 0, 0), 
                             setup['node_pos'][self.end_node], 4)

            # Display timing information
            elapsed_time = finish_time if finish_time is not None else (time.time() - animation_start_time)
            
            # Preprocessing time (static display)
            preprocess_text = font.render(f"Pre-processing Time: {setup['preprocessing_time']:.2f}s", True, (0, 0, 0))
            screen.blit(preprocess_text, (10, 10))
            
            # Algorithm time (dynamic display)
            timer_text = font.render(f"Elapsed Time: {elapsed_time:.2f}s", True, (0, 0, 0))
            screen.blit(timer_text, (10, 40))

            # Video capture
            if save_video:
                frame = pygame.surfarray.array3d(screen)
                frame = np.transpose(frame, (1, 0, 2))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame)

            pygame.display.flip()

            # Exit condition
            if save_video and progress >= 1.0 and (time.time() - animation_start_time) >= (setup['animation_duration'] + end_delay):
                running = False

            clock.tick(90 if not save_video else fps)