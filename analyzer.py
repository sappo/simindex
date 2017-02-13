import os
import glob
import sys
import json
import click
import numpy as np
from simindex import SimEngine, MDySimII, MDySimIII, MDyLSH, RecordTimer, BlockingKey
import json

try:
    profile
except NameError as e:
    def profile(func):
        def inner(*args, **kwargs):
            return func(*args, **kwargs)
        return inner

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
         run_type, output, run_name, indexer,
         baseline):
    """
    Analyze simindex engine!
    """
    # Sanity check
    if (len(index_attributes) != len(query_attributes)):
        print("Query attribute count must equal index attribute count!")
        sys.exit(0)

    # Get a name for this analyze run
    datasetname = os.path.basename(train_file).split('_')[0]
    engine_datadir = '.engine'  # Where to save/load engine state to/from

    if run_type == "fit":
        # Clean engine state
        for index_state in glob.glob("%s/.%s*" % (engine_datadir, datasetname)):
            os.remove(index_state)

        model = {}
        timer = RecordTimer()

        print()
        print("##############################################################")
        print("  Fitting training dataset.")
        print("##############################################################")
        engine = SimEngine(datasetname, datadir=engine_datadir, verbose=True)

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

            for index, similarity in enumerate(similarities):
                similarities[index] = (train_attributes[index + 1], similarity)

        model["blocking_scheme"] = blocking_scheme
        model["similarities"] = engine.similarities
        if len(baseline) == 0:
            model["positive_labels"] = engine.nP
            model["negative_labels"] = engine.nN
            model["positive_filtered_labels"] = engine.nfP
            model["negative_filtered_labels"] = engine.nfN

        # Save metrics
        fp = open(output, mode='w')
        json.dump(model, fp, sort_keys=True, indent=4)

    elif run_type == "build":
        engine = None
        if indexer == "MDySimII":
            engine = SimEngine(datasetname, indexer=MDySimII,
                               datadir=engine_datadir, verbose=False)
        elif indexer == "MDySimIII":
            engine = SimEngine(datasetname, indexer=MDySimIII,
                               datadir=engine_datadir, verbose=False)
        elif indexer == "MDyLSH":
            engine = SimEngine(datasetname, indexer=MDyLSH,
                               datadir=engine_datadir, verbose=False)

        print()
        print("##############################################################")
        print("  Build and Query without calculating metrics")
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
        print("Analyzing engine (%s) on %s dataset:" % (indexer, datasetname))

        engine = None
        if indexer == "MDySimII":
            engine = SimEngine(datasetname, indexer=MDySimII, verbose=True,
                               datadir=engine_datadir,
                               insert_timer=insert_timer,
                               query_timer=query_timer)
        elif indexer == "MDySimIII":
            engine = SimEngine(datasetname, indexer=MDySimIII, verbose=True,
                               datadir=engine_datadir,
                               insert_timer=insert_timer,
                               query_timer=query_timer)
        elif indexer == "MDyLSH":
            engine = SimEngine(datasetname, indexer=MDyLSH, verbose=True,
                               datadir=engine_datadir,
                               insert_timer=insert_timer,
                               query_timer=query_timer)

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
        measurements["pair_completeness"] = engine.pair_completeness()
        measurements["reduction_ratio"] = engine.reduction_ratio()
        measurements["recall"] = engine.recall()
        measurements["precision"] = engine.precision()
        measurements["f1_score"] = engine.f1_score()
        precisions, recalls, thresholds = engine.precision_recall_curve()
        measurements["prc_precisions"] = precisions.tolist()
        measurements["prc_recalls"] = recalls.tolist()
        measurements["prc_thresholds"] = thresholds.tolist()

        print("\tPair completeness:", measurements["pair_completeness"])
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
