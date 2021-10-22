import argparse
import experiments
import inspect


parser = argparse.ArgumentParser(description='Parse arguments for the META-framework.')
parser.add_argument('experiment', help="experiment to be run")

parser.add_argument('-c', '--collectionId', type=str,
                    help="collection to be used (i.e., RQV04).",
                    required=True, choices=['RQV04'])

parser.add_argument('-m', '--mLabel', type=str,
                    help="measure label (i.e., map).",
                    required=True, choices=['map'])

parser.add_argument('-d', '--distrMeasure', type=str,
                    help="distributional measure considered",
                    choices=['sARE', 'sRE', 'sSRE', 'sRSRE', 'sRSRE2', 'sSRE2'])


parser.add_argument('-rt', '--rankType', type=str,
                    help="type of rank",
                    choices=['average', 'min', 'max', 'first', 'dense'])

parser.add_argument('-pr', '--processors', type=int,
                    help="number of processors",
                    default=1)

parser.add_argument('-qp', '--queryPartitioning', type=str,
                    help="set of considered queries (name of the file in query_paritioning, without extension).")

expsNames = [en for en, _ in inspect.getmembers(experiments, inspect.isclass)]


params = vars(parser.parse_args())

params['expLabel'] = f"{params['experiment']}_{params['collectionId']}_{params['mLabel']}_{params['distrMeasure']}_{params['rankType']}"
if "queryPartitioning" in params and params["queryPartitioning"] is not None:
    params['expLabel'] = f"{params['expLabel']}_{params['queryPartitioning']}"
else:
    del params["queryPartitioning"]

if params['experiment'] in expsNames:
    experiment = eval(f"experiments.{params['experiment']}(**params)")
    experiment.run_experiment()
else:
    raise NotImplementedError(f"required experiment {params['experiment']} not recognized")