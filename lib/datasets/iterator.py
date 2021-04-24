import os
import cv2
import numpy as np
import torch.utils.data as data
import pickle

class Video(object):
    """basic Video class"""

    def __init__(self, vid_path):
        self.open(vid_path)

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()

    def reset(self):
        self.close()
        self.vid_path = None
        self.frame_count = -1
        self.faulty_frame = None
        return self

    def open(self, vid_path):
        assert os.path.exists(vid_path), "VideoIter:: cannot locate: `{}'".format(vid_path)

        # close previous video & reset variables
        self.reset()

        # try to open video
        cap = cv2.VideoCapture(vid_path)
        if cap.isOpened():
            self.cap = cap
            self.vid_path = vid_path
        else:
            raise IOError("VideoIter:: failed to open video: `{}'".format(vid_path))

        return self

    def count_frames(self, check_validity=False):
        offset = 0
        if self.vid_path.endswith('.flv'):
            offset = -1
        unverified_frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) + offset
        
        if check_validity:
            frame_index = []
            verified_frame_count = 0
            for i in range(unverified_frame_count):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                if self.cap.grab():
                    verified_frame_count += 1
                    frame_index.append(i)
            self.frame_count = verified_frame_count
        else:
            self.frame_count = unverified_frame_count
            frame_index = [i for i in range(unverified_frame_count)]
        frame_index = np.array(frame_index)
        assert self.frame_count > 0, "VideoIter:: Video: `{}' has no frames".format(self.vid_path)
        return self.frame_count, frame_index

    def extract_frames(self, idxs, force_color=True):
        frames = self.extract_frames_fast(idxs, force_color)
        if frames is None:
            # try slow method:
            frames = self.extract_frames_slow(idxs, force_color)
        return frames

    def extract_frames_fast(self, idxs, force_color=True):
        assert self.cap is not None, "No opened video."
        if len(idxs) < 1:
            return []

        frames = []
        pre_idx = max(idxs)
        for idx in idxs:
            assert (self.frame_count < 0) or (idx < self.frame_count), \
                "idxs: {} > total valid frames({})".format(idxs, self.frame_count)
            if pre_idx != (idx - 1):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            res, frame = self.cap.read() # in BGR/GRAY format
        
            pre_idx = idx
            if not res:
                self.faulty_frame = idx
                return None
            if len(frame.shape) < 3:
                if force_color:
                    # Convert Gray to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        return frames

    def extract_frames_slow(self, idxs, force_color=True):
        assert self.cap is not None, "No opened video."
        if len(idxs) < 1:
            return []

        frames = [None] * len(idxs)
        idx = min(idxs)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        while idx <= max(idxs):
            res, frame = self.cap.read() # in BGR/GRAY format
            
            if (not res) and (idx in idxs):
                # end of the video
                self.faulty_frame = idx
                return None
            if idx in idxs:
                # fond a frame
                if len(frame.shape) < 3:
                    if force_color:
                        # Convert Gray to RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                else:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pos = [k for k, i in enumerate(idxs) if i == idx]
                for k in pos:
                    frames[k] = frame
            idx += 1
        return frames

    def close(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
        return self

class VideoIter(data.Dataset):
    def __init__(self,
                 args,
                 video_prefix,
                 txt_list,
                 sampler,
                 video_transform=None,
                 force_color=True,
                 shuffle_list_seed=None):
        super(VideoIter, self).__init__()
        # load params
        self.args = args
        self.sampler = sampler
        self.force_color = force_color
        self.video_prefix = video_prefix
        self.video_transform = video_transform
        self.rng = np.random.RandomState(shuffle_list_seed if shuffle_list_seed else 0)
        # load video list
        self.video_list = self._get_video_list(video_prefix=video_prefix,
                                               txt_list=txt_list)
        if shuffle_list_seed is not None:
            self.rng.shuffle(self.video_list)
        self.video_frame_info = {}

    def __getitem__(self, index):
        clip_input, label, vid_subpath = self.getitem_from_raw_video(index)
        return clip_input, label

    def __len__(self):
        return len(self.video_list)
    
    def _get_video_list(self, video_prefix, txt_list):
        video_list = []
        new_video_info = {}
        with open(txt_list) as f:
            lines = f.read().splitlines()
            for i, line in enumerate(lines):
                v_id, label, video_subpath = line.split()
                if "," in label:
                    label = label.split(",")
                    label = list(map(int, label))
                else:
                    label = [int(label)]
 
                video_path = os.path.join(video_prefix, video_subpath.split("/")[-1])
                if not os.path.exists(video_path):
                    video_path = os.path.join(video_prefix, video_subpath)
                    if not os.path.exists(video_path):
                        raise ValueError("No video in {}!!".format(video_path))
                info = [int(v_id), label, video_path]
                video_list.append(info)
        return video_list

    def getitem_from_raw_video(self, index):

        v_id, label, vid_subpath = self.video_list[index]
        video_path = os.path.join(self.video_prefix, vid_subpath)

        faulty_frames = []
        successfule_trial = False
        
  
        vid_name = vid_subpath.split("/")[-1]
        with Video(vid_path=video_path) as video:
            
            frame_count, frame_index = video.count_frames(check_validity=False)
            for i_trial in range(20):
                if i_trial>=1:
                    if vid_name not in self.video_frame_info.keys():
                        frame_count, frame_index = video.count_frames(check_validity=True)
                        subdict = {"count" : frame_count, "index" : frame_index}
                        self.video_frame_info.update({vid_name : subdict})
                    else:
                        frame_count = self.frame_count = self.video_frame_info[vid_name]["count"]
                        frame_index = self.video_frame_info[vid_name]["index"]
                # dynamic sampling
                sampled_idxs = self.sampler.sampling(range_max=frame_count, index=frame_index, v_id=v_id)
                if set(list(sampled_idxs)).intersection(faulty_frames):
                    continue
                prev_sampled_idxs = list(sampled_idxs)
                # extracting frames
                sampled_frames = video.extract_frames(idxs=sampled_idxs, force_color=self.force_color)
                if sampled_frames is None:
                    faulty_frames.append(video.faulty_frame)
                else:
                    successfule_trial = True
                    break

        

        clip_input = np.asarray(sampled_frames)

        # apply video augmentation
        if self.video_transform is not None:
            clip_input = self.video_transform(clip_input)
        return clip_input, label, vid_subpath