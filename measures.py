import glob
import sys
import time
import json
import click
import jellyfish
from sklearn.metrics import recall_score, precision_score
from difflib import SequenceMatcher
from simindex import DySimII, DyLSH, DyKMeans, RecordTimer
from simindex import draw_precision_recall_curve, \
                     draw_record_time_curve, \
                     draw_frequency_distribution, \
                     draw_bar_chart, \
                     show


def _compare(a, b):
    return SequenceMatcher(None, a, b).ratio()


def _encode_soundex(a):
    return jellyfish.soundex(a)


def _encode_metaphone(a):
    return jellyfish.metaphone(a)


def _encode_first3(a):
    return a.strip()[:3]


@click.command(context_settings=dict(help_option_names=[u'-h', u'--help']))
@click.argument('index_file', type=click.Path(exists=True))
@click.argument('query_file', type=click.Path(exists=True))
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
    u'-s', u'--gold-standard', type=click.Path(exists=True),
    help=u'CSV with gold standard data'
)
@click.option(
    u'-g', u'--gold-attributes', multiple=True,
    help=u'List of attributes to extract from gold standard.\
           Use multiple times if needed!'
)
@click.option(
    u'-e', u'--encoding_list', multiple=True,
    help=u'List of encodings to apply in order to attributes\
           Use multiple times if needed!'
)
@click.option(
    u'-c', u'--similarity_list', multiple=True,
    help=u'List of similarities to apply in order to attributes\
           Use multiple times if needed!'
)
@click.option(
    u'-t', u'--run_type', help=u'What are you benchmarking?\
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
    u'-m', u'--index_method', help=u'Which method to use for indexing?\
                                     DySimII\
                                     DyLSH'
)
def main(index_file, index_attributes,
         query_file, query_attributes,
         gold_standard, gold_attributes,
         encoding_list, similarity_list,
         run_type, output, run_name, index_method):
    """Run a basic matching task on input file."""
    if (len(index_attributes) != len(query_attributes)):
        print("Query attribute count must equal index attribute count!")
        sys.exit(0)

    # Parse encoding functions
    encoding_fns = []
    for encoding in encoding_list:
        if encoding == "soundex":
            encoding_fns.append(_encode_soundex)
        elif encoding == "metaphone":
            encoding_fns.append(_encode_metaphone)
        elif encoding == "first3":
            encoding_fns.append(_encode_first3)

    # Parse similarity functions
    similarity_fns = []
    for similarity in similarity_list:
        if similarity == "default":
            similarity_fns.append(_compare)

    if run_type == "evaluation":
        measurements = {}
        # Prepare record timers
        insert_timer = RecordTimer()
        query_timer = RecordTimer()

        indexer = None
        if index_method == "DySimII":
            indexer = DySimII(len(index_attributes) - 1, top_n=1,
                              simmetric_fn=similarity_fns,
                              encode_fn=encoding_fns,
                              gold_standard=gold_standard,
                              gold_attributes=gold_attributes,
                              insert_timer=insert_timer, query_timer=query_timer)
        elif index_method == "DyLSH":
            indexer = DyLSH(top_n=1, lsh_threshold=0.09, lsh_num_perm=40,
                            similarity_fn=similarity_fns,
                            encode_fn=encoding_fns,
                            gold_standard=gold_standard,
                            gold_attributes=gold_attributes,
                            insert_timer=insert_timer, query_timer=query_timer)
        elif index_method == "DyKMeans":
            indexer = DyKMeans(top_n=1,
                               similarity_fn=similarity_fns,
                               encode_fn=encoding_fns,
                               gold_standard=gold_standard,
                               gold_attributes=gold_attributes,
                               insert_timer=insert_timer,
                               query_timer=query_timer)

        # Build index
        start = time.time()
        indexer.fit_csv(index_file, index_attributes)
        end = time.time()
        measurements["build_time"] = end - start
        print("Index build time: %f" % measurements["build_time"])
        print("Index build time (sum): %f" % (sum(insert_timer.times) + sum(insert_timer.common_time)))

        insert_timer.apply_common()
        measurements["insert_times"] = insert_timer.times

        if index_method == "DySimII":
            measurements["block_frequency"] = {}
            for index, block_freq in enumerate(indexer.frequency_distribution()):
                measurements["block_frequency"][index_attributes[index + 1]] = block_freq
        elif index_method == "DyLSH" or index_method == "DyKMeans":
            measurements["block_frequency"] = indexer.frequency_distribution()

        # Run Queries
        start = time.time()
        result, \
            measurements["y_true1"], \
            measurements["y_scores"], \
            measurements["y_true2"], \
            measurements["y_pred"] \
            = indexer.query_from_csv(query_file, query_attributes)
        end = time.time()
        measurements["query_time"] = end - start
        print("Index query time: %f" % measurements["query_time"])

        query_timer.apply_common()
        measurements["query_times"] = query_timer.times

        # Calculate Precision/Recall
        measurements["query_records"] = len(result)
        measurements["recall_blocking"] = indexer.recall()
        measurements["precision"] = precision_score(measurements["y_true2"],
                                                    measurements["y_pred"])
        measurements["recall"] = recall_score(measurements["y_true2"],
                                              measurements["y_pred"])
        print("Query records:", measurements["query_records"])
        print("Recall blocking:", measurements["recall_blocking"])
        print("P1:", measurements["precision"])
        print("R1:", measurements["recall"])

        fp = open(output, mode='w')
        json.dump(measurements, fp, sort_keys=True, indent=4)

    elif run_type == "index":
        fp = open(output)
        measurements = json.load(fp)
        fp.close()

        indexer = None
        if index_method == "DySimII":
            indexer = DySimII(len(index_attributes) - 1,
                              simmetric_fn=similarity_fns,
                              encode_fn=encoding_fns)
        elif index_method == "DyLSH":
            indexer = DyLSH(top_n=1, lsh_threshold=0.09, lsh_num_perm=40,
                            similarity_fn=similarity_fns,
                            encode_fn=encoding_fns)
        elif index_method == "DyKMeans":
            indexer = DyKMeans(top_n=1,
                               similarity_fn=similarity_fns,
                               encode_fn=encoding_fns)

        start = time.time()
        indexer.fit_csv(index_file, index_attributes)
        end = time.time()
        measurements["build_time"] = end - start

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

            draw_frequency_distribution(measurements["block_frequency"],
                                        indexer_dataset,
                                        "Block")

            draw_precision_recall_curve(measurements["y_true1"],
                                        measurements["y_scores"],
                                        indexer_dataset)

            # Sort data by indexer and then by dataset
            if indexer not in memory_usage:
                memory_usage[indexer] = {}
                index_build_time[indexer] = {}

            memory_usage[indexer][dataset] = measurements["memory_usage"] / 1024
            index_build_time[indexer][dataset] = measurements["build_time"]

        for dataset in insert_times.keys():
            draw_record_time_curve(insert_times[dataset], dataset, "insertion")

        for dataset in query_times.keys():
            draw_record_time_curve(query_times[dataset], dataset, "query")

        draw_bar_chart(memory_usage, "Memory usage", "MiB")
        draw_bar_chart(index_build_time, "Index build time", "Seconds (s)")


        # Show plots
        show()

    sys.exit(0)


if __name__ == "__main__":  # pragma: no cover
        main(prog_name="benchmark")
