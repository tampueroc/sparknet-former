import os
import torch
import json
from torchvision.io import read_image

from .transforms import LandscapeNormalize, WeatherNormalize


class FireDataset:
    """
    Custom PyTorch Dataset for fire state sequences, static landscapes, and wind inputs.
    """
    def __init__(self, data_dir, sequence_length=3, transform=None):
        """
        A FireDataset that always uses landscape and weather data.

        Args:
            data_dir (str): Base directory for the dataset.
            sequence_length (int, optional): Number of consecutive frames. Default=3.
            transform (callable, optional): Optional transforms to apply on inputs/targets.
        """
        super().__init__()
        self.data_dir = os.path.expanduser(f"~/{data_dir}")
        self.sequence_length = sequence_length
        self.transform = transform

        # Prepare transforms for normalization
        self.landscape_normalizer = LandscapeNormalize()
        self.weather_normalizer = WeatherNormalize()

        # Internal containers
        self.indices = {}
        self.samples = []

        # 1) Load & normalize landscape data
        self.landscape_data = self.landscape_normalizer(
            os.path.join(self.data_dir, "landscape", "Input_Geotiff.tif")
        )
        self.landscape_max = self.landscape_normalizer.landscape_max
        self.landscape_min = self.landscape_normalizer.landscape_min

        # 2) Load spatial indices
        self._load_spatial_index()

        # 3) Load & normalize weather
        weather_folder = os.path.join(self.data_dir, 'landscape', 'Weathers')
        weather_history_path = os.path.join(self.data_dir, 'landscape', 'WeatherHistory.csv')
        self.weathers, self.weather_history = self.weather_normalizer.fit_transform(
            weather_folder, weather_history_path
        )
        # 4) Prepare sub-sequences
        self._prepare_samples()

    def _load_spatial_index(self):
        path = os.path.join(self.data_dir, 'landscape', 'indices.json')
        with open(path, 'r') as f:
            self.indices = json.load(f)

    def _prepare_samples(self):
        fire_root = os.path.join(self.data_dir, 'fire_frames')
        iso_root = os.path.join(self.data_dir, 'isochrones')
        seq_dirs = sorted(os.listdir(fire_root))

        for seq_dir in seq_dirs:
            seq_id = seq_dir.replace('sequence_', '')
            fseq_path = os.path.join(fire_root, seq_dir)
            iseq_path = os.path.join(iso_root, seq_dir)

            fire_files = sorted(f for f in os.listdir(fseq_path) if f.endswith('.png'))
            iso_files = sorted(f for f in os.listdir(iseq_path) if f.endswith('.png'))

            num_frames = len(fire_files)
            assert num_frames == len(iso_files), f"{seq_id} mismatch in frames"

            # Crop the big landscape to just the chunk for this sequence
            # Indices file structure: "str(int(seq_id))": [y, y_, x, x_]
            seq_id = str(int(seq_id))
            if seq_id not in self.indices:
                raise ValueError(f"No indices found for seq {seq_id}")
            y, y_, x, x_ = self.indices[seq_id]

            # Extract from xarray => shape [C, height, width]
            cropped = self.landscape_data[:, y:y_, x:x_].values
            cropped_tensor = torch.from_numpy(cropped).float()

            # Build sub-sequences of length up to `self.sequence_length`
            T = min(num_frames, self.sequence_length)
            if T < 2:
                continue
            # For each possible start
            for start in range(num_frames - T + 1):
                # Example sub-sequence: frames [start, start+1, ..., start+T-1]
                sub_ids = list(range(start, start + T))
                sample = {
                    'sequence_id': seq_id,
                    'fire_path': fseq_path,
                    'iso_path': iseq_path,
                    'fire_files': fire_files,
                    'iso_files': iso_files,
                    'fire_frame_indices': sub_ids[:-1], # Past frames
                    'iso_target_index': sub_ids[-1],    # Final
                    'landscape': cropped_tensor
                }
                self.samples.append(sample)

    def __getitem__(self, index):
        item = self.samples[index]
        seq_id = item['sequence_id']

        # 1) Build Fire Sequence (past frames)
        frame_masks = []
        for frame_idx in item['fire_frame_indices']:
            fpath = os.path.join(item['fire_path'], item['fire_files'][frame_idx])
            img = read_image(fpath)  # [3, H, W]
            mask = torch.where(img[1] == 231, 1.0, 0.0)  # [H, W]
            frame_masks.append(mask)
        # Stack => [T, H, W]
        fire_sequence = torch.stack(frame_masks, dim=0)

        # 2) Static Data (landscape)
        static_data = item['landscape']  # [C, H, W]

        # 3) Wind Inputs
        # We'll only gather wind data for the final frame in that sub-sequence
        last_frame_idx = item['fire_frame_indices'][-1]
        weather_file_name = self.weather_history.iloc[int(seq_id) - 1].values[0].split("Weathers/")[1]
        weather_df = self.weathers[weather_file_name]
        row = weather_df.iloc[last_frame_idx]
        ws, wd = row['WS'], row['WD']

        # Use the normalization transform
        wind_inputs = self.weather_normalizer(ws, wd).unsqueeze(0)  # => [1, 2]

        # 4) Isochrone Mask (target)
        itarget = item['iso_target_index']
        ipath = os.path.join(item['iso_path'], item['iso_files'][itarget])
        iso_img = read_image(ipath)  # [3, H, W]
        isochrone_mask = torch.where(iso_img[1] == 231, 1.0, 0.0).unsqueeze(0)  # => [1, H, W]

        # Optional transform (e.g. cropping, scaling, etc.)
        if self.transform:
            # Here you can adapt how your transform is applied
            # For example, apply transform to each time slice in fire_sequence
            transformed_frames = []
            for t in range(fire_sequence.shape[0]):
                # If transform expects shape [C, H, W], we add a dummy channel
                f_t = fire_sequence[t].unsqueeze(0)  # => [1, H, W]
                f_t = self.transform(f_t)
                transformed_frames.append(f_t.squeeze(0))
            fire_sequence = torch.stack(transformed_frames, dim=0)

            # static_data is [C, H, W] => pass directly
            static_data = self.transform(static_data)

            # isochrone_mask is [1, H, W] => pass directly
            isochrone_mask = self.transform(isochrone_mask)

        return fire_sequence, static_data, wind_inputs, isochrone_mask

    def __len__(self):
        return len(self.samples)


