from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file("configs/mask3d_config.yaml")
print(cfg.dump())  # Shows full config settings for debugging