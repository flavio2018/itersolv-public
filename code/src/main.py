#!/usr/bin/env python

import hydra
import logging
import omegaconf
from tasks.utils import build_task


@hydra.main(config_path="../conf", version_base='1.2')
def main(cfg):
	logging.info(omegaconf.OmegaConf.to_yaml(cfg))
	task = build_task(cfg)
	task.run()


if __name__ == '__main__':
	main()
