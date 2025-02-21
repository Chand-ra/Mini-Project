import pygame
import time
import cv2
import numpy as np

class bidAstar_Animator:
    def __init__(self, graph, start, end, video_filename="animation.mp4"):
        self.G = graph
        self.start_node, self.end_node = start, end
        self.video_filename = video_filename

    def _common_setup(self, algorithm):
        G, start_node, end_node = self.G, self.start_node, self.end_node
        start_time = time.time()
        visited_nodes, visited_edges, optimal_path, meeting_node = algorithm(G, start_node, end_node)
        T = time.time() - start_time

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

        visited_edges_fwd, visited_edges_bwd = visited_edges
        visited_edges_fwd_screen = [(node_pos[u], node_pos[v]) for u, v in visited_edges_fwd]
        visited_edges_bwd_screen = [(node_pos[u], node_pos[v]) for u, v in visited_edges_bwd]

        optimal_path_edges_screen = []
        for i in range(len(optimal_path) - 1):
            u, v = optimal_path[i], optimal_path[i + 1]
            optimal_path_edges_screen.append((node_pos[u], node_pos[v]))

        optimal_path_surface = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
        optimal_path_surface.fill((0, 0, 0, 0))
        for start, end in optimal_path_edges_screen:
            pygame.draw.aaline(optimal_path_surface, (0, 0, 255), start, end)  # Changed to blue

        speed_factor = 100
        animation_duration = T * speed_factor

        return {
            'screen_size': (screen_width, screen_height),
            'node_pos': node_pos,
            'visited_edges_fwd_screen': visited_edges_fwd_screen,
            'visited_edges_bwd_screen': visited_edges_bwd_screen,
            'optimal_path_surface': optimal_path_surface,
            'animation_duration': animation_duration,
            'T': T,
            'meeting_node': node_pos.get(meeting_node, None)
        }

    def animate_path(self, algorithm):
        setup = self._common_setup(algorithm)
        screen_width, screen_height = setup['screen_size']

        pygame.init()
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Bidirectional A* Animation")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Arial", 24)

        video_writer = cv2.VideoWriter(
            self.video_filename,
            cv2.VideoWriter_fourcc(*'mp4v'),
            30,
            (screen_width, screen_height)
        )

        self._run_animation_loop(setup, screen, clock, font, video_writer=video_writer)
        video_writer.release()
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
        pygame.display.set_caption("Bidirectional A-star Animation")
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

    def _run_animation_loop(self, setup, screen, clock, font, save_video=True, video_writer=None, fps=30):
        current_f_frame, current_b_frame = 0, 0
        animation_start_time = time.time()
        running = True

        screen_width, screen_height = setup['screen_size']
        background = pygame.Surface((screen_width, screen_height))
        background.fill((255, 255, 255))
        for u, v, _ in self.G.edges(data=True):
            pygame.draw.aaline(background, (200, 200, 200), setup['node_pos'][u], setup['node_pos'][v])

        visited_edges_surface = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
        meeting_highlighted = False

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            elapsed_time = time.time() - animation_start_time
            progress = min(elapsed_time / setup['animation_duration'], 1.0)

            f_edges = setup['visited_edges_fwd_screen']
            b_edges = setup['visited_edges_bwd_screen']

            if current_f_frame < len(f_edges):
                start_pt, end_pt = f_edges[current_f_frame]
                pygame.draw.aaline(visited_edges_surface, (0, 0, 255), start_pt, end_pt)
                current_f_frame += 1

            if current_b_frame < len(b_edges):
                start_pt, end_pt = b_edges[current_b_frame]
                pygame.draw.aaline(visited_edges_surface, (0, 0, 255), start_pt, end_pt)  # Changed to blue
                current_b_frame += 1

            screen.blit(background, (0, 0))
            screen.blit(visited_edges_surface, (0, 0))

            if progress >= 1.0:
                screen.blit(setup['optimal_path_surface'], (0, 0))

            pygame.draw.circle(screen, (255, 0, 0), setup['node_pos'][self.start_node], 4)
            pygame.draw.circle(screen, (255, 0, 0), setup['node_pos'][self.end_node], 4)

            timer_text = font.render(f"Elapsed Time: {elapsed_time:.2f}s", True, (0, 0, 0))
            screen.blit(timer_text, (10, 10))

            pygame.display.flip()
            clock.tick(fps)

            if save_video and video_writer:
                frame = pygame.surfarray.array3d(screen)
                frame = cv2.transpose(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame)
