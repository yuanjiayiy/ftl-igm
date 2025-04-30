from overcooked_env.visualization.state_visualizer import StateVisualizer
import pygame
import os
import numpy as np
from overcooked_env.static import GRAPHICS_DIR, FONTS_DIR
from overcooked_env.visualization.pygame_utils import scale_surface_by_factor
from moviepy.editor import ImageSequenceClip


class OvercookedSampleRenderer:
    """ Assume Features are (height, width, channels) where channels = 26; Uses Normalized Images;
    Inspired by State Visualizor
    """

    # See lossless_state_encoding in overcooked_mdp.py
    FEATURE_CHANNEL_MAP = [
    "player_0_loc", "player_1_loc", 
    "player_0_orientation_0", "player_0_orientation_1", "player_0_orientation_2", "player_0_orientation_3",
    "player_1_orientation_0", "player_1_orientation_1", "player_1_orientation_2", "player_1_orientation_3",
    "pot_loc", "counter_loc", "onion_disp_loc", "tomato_disp_loc", "dish_disp_loc", "serve_loc",
    "onions_in_pot", "tomatoes_in_pot",
    "onions_in_soup", "tomatoes_in_soup", 
    "soup_cook_time_remaining", "soup_done",
    "dishes", "onions", "tomatoes",
    "urgency"
]
    CHANNEL_FEATURE_MAP = {name: i for i, name in enumerate(FEATURE_CHANNEL_MAP)}

    TERRAIN_IMG = StateVisualizer.TERRAINS_IMG
    OBJECTS_IMG = StateVisualizer.OBJECTS_IMG
    SOUPS_IMG = StateVisualizer.SOUPS_IMG
    CHEFS_IMG = StateVisualizer.CHEFS_IMG
    ARROW_IMG = pygame.image.load(os.path.join(GRAPHICS_DIR, "arrow.png"))
    INTERACT_IMG = pygame.image.load(os.path.join(GRAPHICS_DIR, "interact.png"))

    TILE_TO_FRAME_NAME = StateVisualizer.TILE_TO_FRAME_NAME
    UNSCALED_TILE_SIZE = StateVisualizer.UNSCALED_TILE_SIZE

    MAX_COOK_TIME = 20

  
    def __init__(self, tile_size=15):
        self.tile_size = tile_size
        pygame.init()
        self.cooking_timer_font_size = 10
        self.cooking_timer_font_color = (255, 0, 0)
        roboto_path = os.path.join(FONTS_DIR, "Roboto-Regular.ttf")
        self.cooking_timer_font = pygame.font.Font(roboto_path, self.cooking_timer_font_size)
        
    def get_feature(self, feature_tensor, feature_name, y, x):
        """Gets the current cell value from the feature tensor"""
        if feature_name not in self.CHANNEL_FEATURE_MAP:
            return 0
        channel_idx = self.CHANNEL_FEATURE_MAP[feature_name]
        try:
            return feature_tensor[y,x,channel_idx]
        except IndexError:
            return 0
    def _position_in_unscaled_pixels(self, position):
        """
        get x and y coordinates in tiles, returns x and y coordinates in pixels
        """
        (x, y) = position
        return (self.UNSCALED_TILE_SIZE * x, self.UNSCALED_TILE_SIZE * y)

    def _position_in_scaled_pixels(self, position):
        """
        get x and y coordinates in tiles, returns x and y coordinates in pixels
        """
        (x, y) = position
        return (self.tile_size * x, self.tile_size * y)
    
    def _render_grid(self, surface, grid):
        for y_tile, row in enumerate(grid):
            for x_tile, tile in enumerate(row):
                frame_name = self.TILE_TO_FRAME_NAME.get(tile, "floor")
                self.TERRAIN_IMG.blit_on_surface(
                    surface,
                    self._position_in_unscaled_pixels((x_tile, y_tile)),
                    frame_name
                )
    @staticmethod
    def _soup_frame_name(num_onions, num_tomatoes, status):
        return "soup_%s_tomato_%i_onion_%i" % (
            status,
            num_tomatoes,
            num_onions,
        )
    def _is_player_location(self, feature_tensor, y, x, eps=1e-4):
        """Check if position contains a player"""
        return (self.get_feature(feature_tensor, "player_0_loc", y, x) >= (1 - eps) or 
                self.get_feature(feature_tensor, "player_1_loc", y, x) >= (1 - eps))
    
    def _render_cooking_timers(self, surface, obs, grid, eps=1e-4):
        grid_height, grid_width = len(grid), len(grid[0])
        for y in range(grid_height):
            for x in range(grid_width):
                if grid[y][x] == 'P':  # POT
                    soup_cook_time = self.get_feature(obs, "soup_cook_time_remaining", y, x)
                    soup_done = self.get_feature(obs, "soup_done", y, x) > 0
                    
                    if soup_cook_time > 0 or soup_done:
                        # Create timer text
                        # 20 - cook time
                        if soup_cook_time > 0:
                            timer_text = str(self.MAX_COOK_TIME - int(soup_cook_time))
                        else:
                            # When soup is done but timer is 0, still show "0"
                            timer_text = "20"
                        text_surface = self.cooking_timer_font.render(
                            timer_text, 
                            True, 
                            self.cooking_timer_font_color
                        )
                        
                        # Position the timer text centered on the pot
                        pos = self._position_in_unscaled_pixels((x, y))
                        text_pos = (
                            pos[0] + (self.UNSCALED_TILE_SIZE - text_surface.get_width()) // 2,
                            pos[1] + (self.UNSCALED_TILE_SIZE - text_surface.get_height()) // 2
                        )
                        
                        # Draw the timer text
                        surface.blit(text_surface, text_pos)
    def _render_objects(self, surface, obs, grid, eps=1e-4):
        grid_height, grid_width = len(grid), len(grid[0])
        for y in range(grid_height):
            for x in range(grid_width):
                pos = self._position_in_unscaled_pixels((x,y))

                if self._is_player_location(obs, y, x):
                    continue

                if grid[y][x] == 'P': # POT
                    onions_in_pot = self.get_feature(obs, "onions_in_pot", y, x)
                    tomatoes_in_pot = self.get_feature(obs, "tomatoes_in_pot", y, x)
                    onions_in_soup = self.get_feature(obs, "onions_in_soup", y, x)
                    tomatoes_in_soup = self.get_feature(obs, "tomatoes_in_soup", y, x)
                    soup_cook_time = self.get_feature(obs, "soup_cook_time_remaining", y, x)
                    soup_done = self.get_feature(obs, "soup_done", y, x) > 0

                    if onions_in_pot > 0 or tomatoes_in_pot > 0 or onions_in_soup > 0 or tomatoes_in_soup > 0:
                        status = "idle"
                        if soup_done or soup_cook_time > 0:
                            status = "cooked"
                    
                        num_onions = int(onions_in_pot + onions_in_soup)
                        num_tomatoes = int(tomatoes_in_pot + tomatoes_in_soup)
                        soup_frame = self._soup_frame_name(num_onions,num_tomatoes, status)
                        try:
                            self.SOUPS_IMG.blit_on_surface(surface, pos, soup_frame)
                        except KeyError:
                            print(f"Soup frame '{soup_frame}' not found")
                elif self.get_feature(obs, "dishes", y, x) >= (1 - eps):
                    self.OBJECTS_IMG.blit_on_surface(surface, pos, "dish")
                elif self.get_feature(obs, "onions", y, x) >= (1 - eps):
                    self.OBJECTS_IMG.blit_on_surface(surface, pos, "onion") 
                elif self.get_feature(obs, "tomatoes", y, x) >= (1 - eps):
                    self.OBJECTS_IMG.blit_on_surface(surface, pos, "tomato")
                    

    def _render_players(self, surface, obs, eps=1e-4):
        player_colors = ["blue", "green"]
        directions = ["NORTH", "SOUTH", "EAST", "WEST"]

        for p_idx in range(2): # only two players
            player_loc_ch = self.CHANNEL_FEATURE_MAP[f"player_{p_idx}_loc"]
            player_locs = np.argwhere(obs[:, :, player_loc_ch] >= (1-eps))

            if len(player_locs) > 0:
                player_y, player_x = player_locs[0]
                player_pos = self._position_in_unscaled_pixels((player_x, player_y))

                #Get orientation
                orientation_idx = 0
                for i in range(4):
                    if self.get_feature(obs, f"player_{p_idx}_orientation_{i}", player_y, player_x) >= (1-eps):
                        orientation_idx = i
                        break
                
                direction_name = directions[orientation_idx]
                player_color_name = player_colors[p_idx]

                held_obj_name = ""
                if self.get_feature(obs, "dishes", player_y, player_x) >= (1-eps):
                    held_obj_name = "dish"
                elif self.get_feature(obs, "onions", player_y, player_x) >= (1-eps):
                    held_obj_name = "onion"
                elif self.get_feature(obs, "tomatoes", player_y, player_x) >= (1-eps):
                    held_obj_name = "tomato"
                elif self.get_feature(obs, "soup_done", player_y, player_x) >= (1-eps):
                    onions = self.get_feature(obs, "onions_in_soup", player_y, player_x)
                    tomatoes = self.get_feature(obs, "tomatoes_in_soup", player_y, player_x)
                    if onions > 0 and tomatoes == 0:
                        held_obj_name = "soup-onion"
                    elif tomatoes > 0:
                        held_obj_name = "soup-tomato"
                
                # Render Chef
                chef_frame = direction_name
                if held_obj_name:
                    chef_frame = f"{direction_name}-{held_obj_name}"
                try:
                    self.CHEFS_IMG.blit_on_surface(surface, player_pos, chef_frame)
                except KeyError:
                    print(f"Chef frame '{chef_frame}' not found, using fallback")
                    fallbacks = list(self.CHEFS_IMG.frames_rectangles.keys())
                    if fallbacks:
                        self.CHEFS_IMG.blit_on_surface(surface, player_pos, fallbacks[0])
                
                # Render hat 
                hat_frame = f"{direction_name}-{player_color_name}hat"
                try:
                    self.CHEFS_IMG.blit_on_surface(surface, player_pos, hat_frame)
                except KeyError:
                    pass  # Skip hat if frame not found

    def render_frame(self, obs, grid, un_normalize=True, eps=1e-4):

        height = len(grid)
        width = len(grid[0])

        surface = pygame.Surface((width * self.UNSCALED_TILE_SIZE, height * self.UNSCALED_TILE_SIZE))
        surface.fill((155,101,0))

        if un_normalize:
            obs = self.normalize_obs(obs) 
        
        self._render_grid(surface, grid)
        self._render_objects(surface, obs, grid, eps)
        self._render_cooking_timers(surface, obs, grid, eps)
        self._render_players(surface,obs, eps)

        return surface
    
    def render_trajectory_frames(self, trajectory, grid, output_dir=None):
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "trajectory_viz_output")
        os.makedirs(output_dir, exist_ok=True)
        img_paths = []
        for i, obs in enumerate(trajectory):
            file_path = os.path.join(output_dir, f"viz_frame_{i:04d}.png")
            self.save_obs_image(obs, grid, file_path)
            img_paths.append(file_path)
        return img_paths
    
    def extract_grid_from_obs(self, obs, eps=1e-4):
        grid_height, grid_width = obs.shape[0], obs.shape[1]
        grid = [[' ' for _ in range(grid_width)] for _ in range(grid_height)]
        
        for y in range(grid_height):
            for x in range(grid_width):
                if self.get_feature(obs, "pot_loc", y, x) >= (1 - eps):
                    grid[y][x] = 'P'
                elif self.get_feature(obs, "counter_loc", y, x) >= (1 - eps):
                    grid[y][x] = 'X'
                elif self.get_feature(obs, "onion_disp_loc", y, x) >= (1 - eps):
                    grid[y][x] = 'O'
                elif self.get_feature(obs, "tomato_disp_loc", y, x) >= (1 - eps):
                    grid[y][x] = 'T'
                elif self.get_feature(obs, "dish_disp_loc", y, x) >= (1 - eps):
                    grid[y][x] = 'D'
                elif self.get_feature(obs, "serve_loc", y, x) >= (1 - eps):
                    grid[y][x] = 'S'
        
        # Ensure grid has walls around the edges
        for y in range(grid_height):
            if grid[y][0] == ' ':
                grid[y][0] = 'X'
            if grid[y][grid_width-1] == ' ':
                grid[y][grid_width-1] = 'X'
                
        for x in range(grid_width):
            if grid[0][x] == ' ':
                grid[0][x] = 'X'
            if grid[grid_height-1][x] == ' ':
                grid[grid_height-1][x] = 'X'
        return grid
    
    def render_trajectory_video(self, trajectory, grid, output_dir=None, video_path=None, fps=30, scale=4):
        img_paths = self.render_trajectory_frames(trajectory, grid, output_dir)
        
        if video_path is None:
            video_path = os.path.join(output_dir, "trajectory_viz_video.mp4")
        clip = ImageSequenceClip(img_paths, fps=fps)
        clip.write_videofile(video_path, codec="libx264", fps=fps, verbose=False, logger=None)
        print(f"Video saved to {video_path}")
        return video_path
    
    def convert_flatten_map(self, flat_obs, height=8, width=5):
        """Converts a flattened feature vector to 3D Representation.
        Args:
            flat_obs : Numpy Array of Shape (1041,)
        """
        channels = len(self.FEATURE_CHANNEL_MAP)
        spatial_size = height * width * channels
        if len(flat_obs) < spatial_size:
            print(f"Warning: Flat feature vector length ({len(flat_obs)}) is smaller than " 
                f"expected spatial size ({spatial_size})")
            padded = np.zeros(spatial_size)
            padded[:len(flat_obs)] = flat_obs
            flat_obs = padded
        elif len(flat_obs) > spatial_size:
            # print(f"Note: Using first {spatial_size} elements of {len(flat_obs)}-length vector")
            flat_obs = flat_obs[:spatial_size]
    
        spatial_features = flat_obs.reshape(height, width, channels)
        return spatial_features
    
    def normalize_obs(self, obs, threshold_value=0.1):
        """Normalizes each channel to [0,1] and scales object channels to their max values."""
        normalized = np.zeros_like(obs)
        max_values = {
            "onions_in_pot": 3,
            "tomatoes_in_pot": 3,
            "onions_in_soup": 3, 
            "tomatoes_in_soup": 3,
            "soup_cook_time_remaining": 20
        }
        
        # Map object channels to their indices and max values
        idx_to_max = {}
        for ch_name, max_val in max_values.items():
            if ch_name in self.CHANNEL_FEATURE_MAP:
                ch_idx = self.CHANNEL_FEATURE_MAP[ch_name]
                idx_to_max[ch_idx] = max_val
        
        # Normalize each channel
        for ch_idx in range(obs.shape[-1]):
            channel_data = obs[:, :, ch_idx]
            if ch_idx in idx_to_max:
                # Object channel: normalize from [0, max_val] to [0,1]
                max_val = idx_to_max[ch_idx]
                normalized_channel = channel_data / max_val
            else:
                # Image channel: handle [-1,1] or [0,n] ranges
                input_min, input_max = channel_data.min(), channel_data.max()
                if input_min < 0:
                    # Assume range [-1, 1]
                    normalized_channel = (channel_data + 1.0) / 2.0
                else:
                    # Assume range [0, n]
                    normalized_channel = channel_data / input_max if input_max > 0 else channel_data
            normalized[:, :, ch_idx] = normalized_channel
        
        # Clip all channels to [0,1]
        normalized = np.clip(normalized, 0.0, 1.0)
        
        # Scale object channels back to [0, max_val] using threshold
        for ch_idx, max_val in idx_to_max.items():
            ch_values = normalized[:, :, ch_idx]
            non_zero_mask = ch_values > threshold_value
            if np.any(non_zero_mask):
                scaled_values = ch_values[non_zero_mask] * max_val
                ch_values[non_zero_mask] = np.round(scaled_values)
                # Ensure values do not exceed max_val due to floating point errors
                np.clip(ch_values, 0, max_val, out=ch_values)
                normalized[:, :, ch_idx] = ch_values
        
        return normalized
    

    def save_obs_image(self, obs, grid, file_path, scale=4):
        surface = self.render_frame(obs, grid)
        if scale != 1:
            surface = scale_surface_by_factor(surface, scale)
        pygame.image.save(surface, file_path)
        return file_path
