import os
import glob
import sys
import json
import click
import numpy as np
from simindex import SimEngine, MDySimII, MDySimIII, MDyLSH, RecordTimer, BlockingKey
import json
import logging.config

try:
    profile
except NameError as e:
    def profile(func):
        def inner(*args, **kwargs):
            return func(*args, **kwargs)
        return inner


def setup_logging(default_path='logging.json',
                  default_level=logging.INFO,
                  env_key='LOG_CFG'):
    """Setup logging configuration """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


@click.command(context_settings=dict(help_option_names=[u'-h', u'--help']))
@click.argument('index_file', type=click.Path(exists=True))
@click.argument('query_file', type=click.Path(exists=True))
@click.argument('train_file', type=click.Path(exists=True))
@click.option(
    u'-i', u'--index-attributes', multiple=True,
    help=u'Attribute to extract from index file.\
           Use multiple times if needed!'
)
@click.option(
    u'-q', u'--query-attributes', multiple=True,
    help=u'List of attributes to extract from query file.\
           Use multiple times if needed!'
)
@click.option(
    u'-t', u'--train-attributes', multiple=True,
    help=u'List of attributes to extract from query file.\
           Use multiple times if needed!'
)
@click.option(
    u'-s', u'--gold-standard', type=click.Path(exists=True),
    help=u'CSV with gold standard data'
)
@click.option(
    u'-g', u'--gold-attributes', multiple=True,
    help=u'List of attributes to extract from gold standard.\
           Use multiple times if needed!'
)
@click.option(
    u'-b', u'--baseline', multiple=True,
    help=u'Baseline to fit. Use multiple times if needed!'
)
@click.option(
    u'-m', u'--indexer', help=u'Which indexer to use!'
)
@click.option(
    u'--clf', help=u'Which clf to use!', type=click.STRING
)
@click.option(
    u'--similarity', help=u'Which similarity to use!', type=click.STRING
)
@click.option(
    u'--sim-avp/--no-sim-avp', help=u'Use average precision score?',
    default=True
)
@click.option(
    u'--classifier/--no-classifier', help=u'Usage of a classifier?',
    default=True
)
@click.option(
    u'--classifier-scoring', help=u'', default="f1", type=click.STRING
)
@click.option(
    u'--full-simvector/--no-full-simvector', help=u'Calalucate full simvector?',
    default=False
)
@click.option(
    u'--parfull-simvector/--no-parfull-simvector', help=u'Calalucate full simvector?',
    default=False
)
@click.option(
    u'--gt-labels/--no-gt-labels', help=u'Provide Label Generate with ground truth matches?',
    default=True
)
@click.option(
    u'--gt-thresholds', help=u'',
    type=(float, float, int, float, float), default=(0.3, 0.3, 2, 0.1, 0.25)
)
@click.option(
    u'--dnf-depths', help=u'',
    type=(int, int), default=(2, 3)
)
@click.option(
    u'--dnf-filters', help=u'',
    type=(int, float), default=(100, 0.85)
)
@click.option(
    u'-rt', u'--run-type', help=u'What are you benchmarking?\
                                 evaluation - calculate all metrics\
                                 plot - draw results'
)
@click.option(
    u'-o', u'--output', help=u'File name to print the result to or \
                               the read them from.'
)
@click.option(
    u'-r', u'--run_name', help=u'Common prefix of output file from one\
                                 run to draw measures collectively.'
)
@profile
def main(index_file, index_attributes,
         query_file, query_attributes,
         train_file, train_attributes,
         gold_standard, gold_attributes,
         run_type, classifier, classifier_scoring,
         full_simvector, parfull_simvector,
         gt_labels, gt_thresholds,
         dnf_depths, dnf_filters,
         clf, similarity, sim_avp,
         output, run_name, indexer,
         baseline):
    """
    Analyze simindex engine!
    """
    setup_logging()
    # Sanity check
    if (len(index_attributes) != len(query_attributes)):
        print("Query attribute count must equal index attribute count!")
        sys.exit(0)

    # Get a name for this analyze run
    datasetname = os.path.basename(train_file).split('_')[0]
    engine_datadir = '.engine'  # Where to save/load engine state to/from

    if run_type == "fit":
        # Clean engine state
        for index_state in glob.glob("%s/.%s*.cls" % (engine_datadir, datasetname)):
            os.remove(index_state)

        model = {}
        timer = RecordTimer()

        print()
        print("##############################################################")
        print("  Fitting training dataset.")
        print("##############################################################")
        clf_name = None
        clf_params = None
        if clf:
            if clf == "svmlinear":
                clf_name = "SVM"
                clf_params = [{'kernel': ['linear'],  'C': [0.1, 1, 10, 100, 1000]}]
            elif clf == "svmrbf":
                clf_name = "SVM"
                clf_params = [{'kernel': ['rbf'],     'C': [0.1, 1, 10, 100, 1000]}]
            elif clf == "decisiontree":
                clf_name = "DT"
                clf_params = [{'max_features': ['auto', 'sqrt', 'log2']}]

        if indexer == "MDySimII":
            engine = SimEngine(datasetname, indexer=MDySimII,
                               label_thresholds=gt_thresholds,
                               max_bk_conjunction=dnf_depths[0], max_bk_disjunction=dnf_depths[1],
                               max_blocksize=dnf_filters[0], min_goodratio=dnf_filters[1],
                               clf_cfg=clf_name, clf_cfg_params=clf_params,
                               datadir=engine_datadir, verbose=True,
                               use_average_precision_score=sim_avp,
                               use_classifier=classifier, clf_scoring=classifier_scoring,
                               use_full_simvector=full_simvector,
                               use_parfull_simvector=parfull_simvector)
        elif indexer == "MDySimIII":
            engine = SimEngine(datasetname, indexer=MDySimIII,
                               label_thresholds=gt_thresholds,
                               max_bk_conjunction=dnf_depths[0], max_bk_disjunction=dnf_depths[1],
                               max_blocksize=dnf_filters[0], min_goodratio=dnf_filters[1],
                               clf_cfg=clf_name, clf_cfg_params=clf_params,
                               datadir=engine_datadir, verbose=True,
                               use_average_precision_score=sim_avp,
                               use_classifier=classifier, clf_scoring=classifier_scoring,
                               use_full_simvector=full_simvector,
                               use_parfull_simvector=parfull_simvector)
        elif indexer == "MDyLSH":
            engine = SimEngine(datasetname, indexer=MDyLSH,
                               label_thresholds=gt_thresholds,
                               max_bk_conjunction=dnf_depths[0], max_bk_disjunction=dnf_depths[1],
                               max_blocksize=dnf_filters[0], min_goodratio=dnf_filters[1],
                               clf_cfg=clf_name, clf_cfg_params=clf_params,
                               datadir=engine_datadir, verbose=True,
                               use_average_precision_score=sim_avp,
                               use_classifier=classifier, clf_scoring=classifier_scoring,
                               use_full_simvector=full_simvector,
                               use_parfull_simvector=parfull_simvector)

        if len(baseline) > 0:
            baseline_scheme = []
            for line in baseline:
                field = 0
                lineattrs = line.split(' ')
                for index, train_attribute in enumerate(train_attributes):
                    if train_attribute == lineattrs[1]:
                        field = index - 1  # -1 for record id

                baseline_scheme.append([int(lineattrs[0]), int(field), lineattrs[2]])

            similarities = []
            for train_attribute in train_attributes[1:]:
                similarities.append('SimDamerau')

            engine.set_baseline({'scheme': baseline_scheme,
                                 'similarities': similarities})

        if similarity:
            similarities = []
            for _ in train_attributes[1:]:
                if similarity == "bag":
                    similarities.append('SimBag')
                if similarity == "compression":
                    similarities.append('SimCompression')
                if similarity == "hamming":
                    similarities.append('SimHamming')
                if similarity == "levenshtein":
                    similarities.append('SimLevenshtein')
                if similarity == "jaccard":
                    similarities.append('SimJaccard')
                if similarity == "jaro":
                    similarities.append('SimJaro')
                if similarity == "jarowinkler":
                    similarities.append('SimJaroWinkler')
                if similarity == "ratio":
                    similarities.append('SimRatio')
                if similarity == "wdegree":
                    similarities.append('SimWdegree')

            engine.similarities = similarities
            engine.save_similarities()

        if gt_labels:
            engine.read_ground_truth(gold_standard, gold_attributes)

        with timer:
            engine.fit_csv(train_file, train_attributes)
        model["fit_time"] = timer.times.pop()

        blocking_scheme = engine.blocking_scheme_to_strings()
        similarities = engine.similarities
        # Try to replace field number with names
        if len(train_attributes) > 0:
            for blocking_key in blocking_scheme:
                blocking_key[1] = train_attributes[blocking_key[1] + 1]

            for index, sim in enumerate(similarities):
                similarities[index] = (train_attributes[index + 1], sim)

        model["blocking_scheme"] = blocking_scheme
        model["blocking_scheme_max_c"] = engine.max_bk_conjunction
        model["similarity"] = similarity
        model["similarities"] = engine.similarities
        model["best_classifier"] = type(engine.clf).__name__
        model["best_params"] = engine.clf_best_params
        model["best_score"] = engine.clf_best_score
        model["conjunctions"] = engine.max_bk_conjunction
        model["disjunctions"] = engine.max_bk_disjunction
        model["max_blocksize"] = engine.max_blocksize
        model["min_goodratio"] = engine.min_goodratio
        model["clf_result_grid"] = engine.clf_result_grid
        model["clf"] = clf
        model["use_classifier"] = engine.use_classifier
        model["use_fullvector"] = engine.use_full_simvector
        model["use_parfullvector"] = engine.use_parfull_simvector
        model["gt_lbl_thresholds"] = engine.label_thresholds
        if len(baseline) == 0:
            model["positive_labels"] = engine.nP
            model["negative_labels"] = engine.nN
            model["positive_filtered_labels"] = engine.nfP
            model["negative_filtered_labels"] = engine.nfN
            model["use_gt_matches"] = engine.gold_pairs != None

        # Save metrics
        fp = open(output, mode='w')
        json.dump(model, fp, sort_keys=True, indent=4)

    elif run_type == "build":
        engine = None
        if indexer == "MDySimII":
            engine = SimEngine(datasetname, indexer=MDySimII,
                               datadir=engine_datadir, verbose=False,
                               use_classifier=classifier,
                               use_full_simvector=full_simvector,
                               use_parfull_simvector=parfull_simvector)
        elif indexer == "MDySimIII":
            engine = SimEngine(datasetname, indexer=MDySimIII,
                               datadir=engine_datadir, verbose=False,
                               use_classifier=classifier,
                               use_full_simvector=full_simvector,
                               use_parfull_simvector=parfull_simvector)
        elif indexer == "MDyLSH":
            engine = SimEngine(datasetname, indexer=MDyLSH,
                               datadir=engine_datadir, verbose=False,
                               use_classifier=classifier,
                               use_full_simvector=full_simvector,
                               use_parfull_simvector=parfull_simvector)

        print()
        print("##############################################################")
        print("  Build and Query (%s) on %s without calculating metrics" % (indexer, datasetname))
        print("##############################################################")

        print()
        print("------------------------------ 1 -----------------------------")
        print("Loading fitted model.")
        engine.fit_csv(train_file, train_attributes)

        print()
        print("------------------------------ 2 -----------------------------")
        print("Building the index.")
        engine.build_csv(index_file, index_attributes)

        print()
        print("------------------------------ 3 -----------------------------")
        print("Run Queries.")
        engine.query_csv(query_file, query_attributes)

    elif run_type == "evaluation":
        measurements = {}

        # Prepare record timers
        insert_timer = RecordTimer()
        query_timer = RecordTimer()
        timer = RecordTimer()

        print()
        print("##############################################################")
        print("  Analyzing engine (%s) on %s dataset:" % (indexer, datasetname))
        print("##############################################################")

        engine = None
        if indexer == "MDySimII":
            engine = SimEngine(datasetname, indexer=MDySimII, verbose=True,
                               datadir=engine_datadir,
                               insert_timer=insert_timer,
                               query_timer=query_timer,
                               use_classifier=classifier,
                               use_full_simvector=full_simvector,
                               use_parfull_simvector=parfull_simvector)
        elif indexer == "MDySimIII":
            engine = SimEngine(datasetname, indexer=MDySimIII, verbose=True,
                               datadir=engine_datadir,
                               insert_timer=insert_timer,
                               query_timer=query_timer,
                               use_classifier=classifier,
                               use_full_simvector=full_simvector,
                               use_parfull_simvector=parfull_simvector)
        elif indexer == "MDyLSH":
            engine = SimEngine(datasetname, indexer=MDyLSH, verbose=True,
                               datadir=engine_datadir,
                               insert_timer=insert_timer,
                               query_timer=query_timer,
                               use_classifier=classifier,
                               use_full_simvector=full_simvector,
                               use_parfull_simvector=parfull_simvector)
        print()
        print("------------------------------ 1 -----------------------------")
        print("Loading fitted training dataset.")
        with timer:
            engine.fit_csv(train_file, train_attributes)
        measurements["fit_time"] = timer.times.pop()
        print("\tFitting complete in: %fs" % measurements["fit_time"])

        engine.read_ground_truth(gold_standard, gold_attributes)

        print()
        print("------------------------------ 2 -----------------------------")
        print("Building the index.")
        with timer:
            engine.build_csv(index_file, index_attributes)
        measurements["build_time"] = timer.times.pop()
        measurements["build_time_sum"] = sum(insert_timer.times) + \
                                         sum(insert_timer.common_time)
        measurements["inserts_mean"] = np.mean(insert_timer.times)
        measurements["inserts_sec"] = 1 / measurements["inserts_mean"]
        print("\tBuild time: %fs" % measurements["build_time"])
        print("\tBuild time (sum): %fs" % measurements["build_time_sum"])

        insert_timer.apply_common()
        measurements["insert_times"] = insert_timer.times
        print("\tIndex mean insert time: %fs" % np.mean(insert_timer.times))

        print()
        print("------------------------------ 3 -----------------------------")
        print("Run Queries.")
        with timer:
            engine.query_csv(query_file, query_attributes)
        measurements["query_time"] = timer.times.pop()
        measurements["query_time_sum"] = sum(query_timer.times) + \
                                         sum(query_timer.common_time)
        print("\tQuery time: %fs" % measurements["query_time"])
        print("\tQuery time (sum): %fs" % measurements["query_time_sum"])
        query_timer.apply_common()
        measurements["query_times"] = query_timer.times
        measurements["queries_mean"] = np.mean(query_timer.times)
        measurements["queries_sec"] = 1 / measurements["queries_mean"]
        print("\tQuery mean time: %fs" % measurements["queries_mean"])
        print("\tQueries (s):", measurements["queries_sec"])

        print()
        print("------------------------------ 4 -----------------------------")
        print("Metrics.")
        measurements["pair_completeness"] = engine.pairs_completeness()
        measurements["pairs_quality"] = engine.pairs_quality()
        measurements["reduction_ratio"] = engine.reduction_ratio()
        measurements["recall"] = engine.recall()
        measurements["precision"] = engine.precision()
        measurements["f1_score"] = engine.f1_score()
        if engine.use_classifier:
            measurements["average_precision"] = engine.average_precision()
            print("\tAverage Precision:", measurements["average_precision"])
            precisions, recalls, thresholds = engine.precision_recall_curve()
            measurements["prc_precisions"] = precisions.tolist()
            measurements["prc_recalls"] = recalls.tolist()
            measurements["prc_thresholds"] = thresholds.tolist()

        print("\tPairs completeness:", measurements["pair_completeness"])
        print("\tPairs quality:", measurements["pairs_quality"])
        print("\tReduction ratio:", measurements["reduction_ratio"])
        print("\tRecall:", measurements["recall"])
        print("\tPrecision:", measurements["precision"])
        print("\tF1-Score:", measurements["f1_score"])

        # Save metrics
        fp = open(output, mode='w')
        json.dump(measurements, fp, sort_keys=True, indent=4)

    # Cleanup index state - next run might use a different
    for index_state in glob.glob("%s/.%s*.idx" % (engine_datadir, datasetname)):
        os.remove(index_state)

    sys.exit(0)


if __name__ == "__main__":  # pragma: no cover
        main(prog_name="benchmark")
