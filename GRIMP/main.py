import toml


import argparse
from types import SimpleNamespace
import GRIMP.utils as utils
import GRIMP.pipeline as pipeline
from GRIMP.logging import GrimpLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", action="store")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    config_path = 'config.toml'
    base_config = toml.load(config_path)
    run_configs = utils.prepare_config_dict(base_config)

    for d in run_configs:
        logger = GrimpLogger()
        d = utils.complete_config(d)
        args = SimpleNamespace(**d)
        logger.add_dict("parameters", vars(args))
        logger.add_run_name()
        logger.add_value("parameters", "num_estimators", 0)
        logger.add_time("start_training")
        graph_dataset, best_state, init_params = pipeline.run_training(args, logger)
        logger.add_time("end_training")
        logger.add_duration("start_training", "end_training", "duration_training")
        logger.print_summary()
        logger.save_json()
        pipeline.run_testing(
            args, graph_dataset, best_state, init_params, logger=logger
        )

        print(f"Completed run {logger.run_id}")
