import os
import glob
import sys
import time
import json
import click
import numpy as np
from pprint import pprint
from simindex import SimEngine, MDySimII, MDySimIII, MDyLSH, RecordTimer
from simindex import draw_precision_recall_curve, \
                     draw_record_time_curve, \
                     draw_frequency_distribution, \
                     draw_bar_chart, \
                     show

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
    u'-rt', u'--run-type', help=u'What are you benchmarking?\
                                 evaluation - calculate all metrics\
                                 index - measure indexing time\
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
@click.option(
    u'-m', u'--indexer', help=u'Which method to use for indexing?\
                                     MDySimII\
                                     MDySimIII\
                                     MDyLSH'
)
def main(index_file, index_attributes,
         query_file, query_attributes,
         train_file, train_attributes,
         gold_standard, gold_attributes,
         run_type, output, run_name, indexer):
    """Run a basic matching task on input file."""
    # Sanity check
    if (len(index_attributes) != len(query_attributes)):
        print("Query attribute count must equal index attribute count!")
        sys.exit(0)

    name = os.path.basename(train_file).split('_')[0]
    for engine_state in glob.glob("./.%s*" % name):
        os.remove(engine_state)

    if run_type == "evaluation":
        measurements = {}
        # Prepare record timers
        insert_timer = RecordTimer()
        query_timer = RecordTimer()
        timer = RecordTimer()

        print()
        print("##############################################################")
        print("Analyzing engine (%s) on %s dataset:" % (indexer, name))

        engine = None
        if indexer == "MDySimII":
            engine = SimEngine(name, indexer=MDySimII, verbose=True,
                               insert_timer=insert_timer,
                               query_timer=query_timer)
        elif indexer == "MDySimIII":
            engine = SimEngine(name, indexer=MDySimIII, verbose=True,
                               insert_timer=insert_timer,
                               query_timer=query_timer)
        elif indexer == "MDyLSH":
            engine = SimEngine(name, indexer=MDyLSH, verbose=True,
                               insert_timer=insert_timer,
                               query_timer=query_timer)

        print()
        print("------------------------------ 1 -----------------------------")
        print("Fitting training dataset.")
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
        print("\tBuild time: %fs" % measurements["build_time"])
        print("\tBuild time (sum): %fs" % (sum(insert_timer.times) +
                                                sum(insert_timer.common_time)))
        insert_timer.apply_common()
        measurements["insert_times"] = insert_timer.times
        print("\tIndex mean insert time: %fs" % np.mean(insert_timer.times))

        print()
        print("------------------------------ 3 -----------------------------")
        print("Run Queries.")
        with timer:
            engine.query_csv(query_file, query_attributes)
        measurements["query_time"] = timer.times.pop()
        print("\tQuery time: %fs" % measurements["query_time"])
        print("\tQuery time (sum): %fs" % (sum(query_timer.times) +
                                          sum(query_timer.common_time)))
        query_timer.apply_common()
        measurements["query_times"] = query_timer.times
        mean = np.mean(query_timer.times)
        print("\tQuery mean time: %fs" % mean)
        print("\tQueries (s):", 1 / mean)

        print()
        print("------------------------------ 4 -----------------------------")
        print("Metrics.")
        measurements["pair_completeness"] = engine.pair_completeness()
        measurements["reduction_ratio"] = engine.reduction_ratio()
        measurements["recall"] = engine.recall()
        measurements["precision"] = engine.precision()
        measurements["f1_score"] = engine.f1_score()
        measurements["y_true"] = engine.y_true
        measurements["y_pred"] = engine.y_pred
        measurements["y_scores"] = engine.y_scores

        print("\tPair completeness:", measurements["pair_completeness"])
        print("\tReduction ratio:", measurements["reduction_ratio"])
        print("\tRecall:", measurements["recall"])
        print("\tPrecision:", measurements["precision"])
        print("\tF1-Score:", measurements["f1_score"])

        # Save results
        fp = open(output, mode='w')
        json.dump(measurements, fp, sort_keys=True, indent=4)

    elif run_type == "index":
        fp = open(output)
        measurements = json.load(fp)
        fp.close()

        engine = None
        if indexer == "MDySimII":
            engine = SimEngine(name, indexer=MDySimII)
        elif indexer == "MDySimIII":
            engine = SimEngine(name, indexer=MDySimIII)
        elif indexer == "MDyLSH":
            engine = SimEngine(name, indexer=MDyLSH)

        # TODO: after fitting
        engine.read_ground_truth(gold_standard, gold_attributes)

        fp = open(output, mode='w')
        json.dump(measurements, fp, sort_keys=True, indent=4)

    elif run_type == "plot":
        memory_usage = {}
        index_build_time = {}
        insert_times = {}
        query_times = {}
        for resultfile in glob.glob("./%s*" % run_name):
            fp = open(resultfile)
            measurements = json.load(fp)
            fp.close()
            indexer_dataset = resultfile[resultfile.index('_') + 1:]
            indexer = indexer_dataset[:indexer_dataset.index('_')]
            dataset = indexer_dataset[indexer_dataset.index('_') + 1:]

            if dataset not in insert_times:
                insert_times[dataset] = {}
                query_times[dataset] = {}

            insert_times[dataset][indexer] = measurements["insert_times"]
            query_times[dataset][indexer] = measurements["query_times"]

            draw_precision_recall_curve2(measurements["prc_curve"],
                                        indexer_dataset)

            # Sort data by indexer and then by dataset
            # if indexer not in memory_usage:
                # memory_usage[indexer] = {}
                # index_build_time[indexer] = {}

            # memory_usage[indexer][dataset] = measurements["memory_usage"] / 1024
            index_build_time[indexer][dataset] = measurements["build_time"]

        for dataset in insert_times.keys():
            draw_record_time_curve(insert_times[dataset], dataset, "insertion")

        for dataset in query_times.keys():
            draw_record_time_curve(query_times[dataset], dataset, "query")

        # draw_bar_chart(memory_usage, "Memory usage", "MiB")
        draw_bar_chart(index_build_time, "Index build time", "Seconds (s)")

        # Show plots
        show()

    sys.exit(0)


if __name__ == "__main__":  # pragma: no cover
        main(prog_name="benchmark")

