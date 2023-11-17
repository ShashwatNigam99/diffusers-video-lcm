
# # 

# class Text2VideoDataset:
#     def __init__(
#         self,
#         train_shards_path_or_url: Union[str, List[str]],
#         num_train_examples: int,
#         per_gpu_batch_size: int,
#         global_batch_size: int,
#         num_workers: int,
#         per_video_frames: int = 16,
#         video_fps: int = 10,
#         resolution: int = 512,
#         shuffle_buffer_size: int = 1000,
#         pin_memory: bool = False,
#         persistent_workers: bool = False,
#     ):
#         if not isinstance(train_shards_path_or_url, str):
#             train_shards_path_or_url = [list(braceexpand(urls)) for urls in train_shards_path_or_url]
#             # flatten list using itertools
#             train_shards_path_or_url = list(itertools.chain.from_iterable(train_shards_path_or_url))

#         # def transform(example):
#         #     # resize image
#         #     image = example["image"]
#         #     image = TF.resize(image, resolution, interpolation=transforms.InterpolationMode.BILINEAR)

#         #     # get crop coordinates and crop image
#         #     c_top, c_left, _, _ = transforms.RandomCrop.get_params(image, output_size=(resolution, resolution))
#         #     image = TF.crop(image, c_top, c_left, resolution, resolution)
#         #     image = TF.to_tensor(image)
#         #     image = TF.normalize(image, [0.5], [0.5])

#         #     example["image"] = image
#         #     return example

#         decoder_kwargs = {
#             "n_frames": per_video_frames,  # get `per_video_frames` frames from each video
#             "fps": video_fps,  # downsample to `video_fps` FPS
#             "num_threads": 12,  # use 12 threads to decode the video
#         }
#         resize_size = crop_size = resolution
#         batch_size = per_gpu_batch_size

#         self._train_dataset  = get_video_dataset(
#             urls=train_shards_path_or_url,
#             batch_size=batch_size,
#             decoder_kwargs=decoder_kwargs,
#             resize_size=resize_size,
#             crop_size=crop_size,
#         )
        
#         # processing_pipeline = [
#         #     wds.decode("pil", handler=wds.ignore_and_continue),
#         #     wds.rename(image="jpg;png;jpeg;webp", text="text;txt;caption", handler=wds.warn_and_continue),
#         #     wds.map(filter_keys({"image", "text"})),
#         #     wds.map(transform),
#         #     wds.to_tuple("image", "text"),
#         # ]

#         # # Create train dataset and loader
#         # pipeline = [
#         #     wds.ResampledShards(train_shards_path_or_url),
#         #     tarfile_to_samples_nothrow,
#         #     wds.shuffle(shuffle_buffer_size),
#         #     *processing_pipeline,
#         #     wds.batched(per_gpu_batch_size, partial=False, collation_fn=default_collate),
#         # ]

#         num_worker_batches = math.ceil(num_train_examples / (global_batch_size * num_workers))  # per dataloader worker
#         num_batches = num_worker_batches * num_workers
#         num_samples = num_batches * global_batch_size

#         # each worker is iterating over this
#         # self._train_dataset = wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)
#         self._train_dataloader = wds.WebLoader(
#             self._train_dataset,
#             batch_size=None,
#             shuffle=False,
#             num_workers=num_workers,
#             pin_memory=pin_memory,
#             persistent_workers=persistent_workers,
#         )
#         # add meta-data to dataloader instance for convenience
#         self._train_dataloader.num_batches = num_batches
#         self._train_dataloader.num_samples = num_samples

#     @property
#     def train_dataset(self):
#         return self._train_dataset

#     @property
#     def train_dataloader(self):
#         return self._train_dataloader
