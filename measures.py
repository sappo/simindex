import os
import glob
import sys
import time
import json
import click
from sklearn.metrics import recall_score, precision_score
import jellyfish
from difflib import SequenceMatcher
from simindex import DySimII, RecordTimer
from simindex import draw_precision_recall_curve, \
                     draw_record_time_curve, \
                     draw_frequency_distribution, \
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
    help=u'List of encodings to apply in order to attributes'
)
@click.option(
    u'-c', u'--similarity_list', multiple=True,
    help=u'List of similarities to apply in order to attributes'
)
@click.option(
    u'-t', u'--run_type', help=u'CSV with gold standard data'
)
@click.option(
    u'-o', u'--output', help=u'File name to print the result to or \
                               the read them from.'
)
@click.option(
    u'-r', u'--run_name', help=u''
)
def main(index_file, index_attributes,
         query_file, query_attributes,
         gold_standard, gold_attributes,
         encoding_list, similarity_list,
         run_type, output, run_name):
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

        dy_sim = DySimII(len(index_attributes) - 1, top_n=1,
                         simmetric_fn=similarity_fns, encode_fn=encoding_fns,
                         gold_standard=gold_standard,
                         gold_attributes=gold_attributes,
                         insert_timer=insert_timer, query_timer=query_timer)

        # Build index
        start = time.time()
        dy_sim.insert_from_csv(index_file, index_attributes)
        end = time.time()
        measurements["build_time"] = end - start
        print("Index build time:", measurements["build_time"])

        measurements["insert_times"] = insert_timer.times
        measurements["query_times"] = query_timer.times

        measurements["block_frequency"] = {}
        for index, block_freq in enumerate(dy_sim.frequency_distribution()):
            measurements["block_frequency"][index_attributes[index + 1]] = block_freq

        # Run Queries
        result, \
            measurements["y_true1"], \
            measurements["y_scores"], \
            measurements["y_true2"], \
            measurements["y_pred"] \
            = dy_sim.query_from_csv(query_file, query_attributes)

        # Calculate Precision/Recall
        measurements["query_records"] = len(result)
        measurements["recall_blocking"] = dy_sim.recall()
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

        dy_sim = DySimII(len(index_attributes) - 1,
                         simmetric_fn=similarity_fns, encode_fn=encoding_fns)

        start = time.time()
        dy_sim.insert_from_csv(index_file, index_attributes)
        end = time.time()
        measurements["build_time"] = end - start

        fp = open(output, mode='w')
        json.dump(measurements, fp, sort_keys=True, indent=4)

    elif run_type == "plot":
        for resultfile in glob.glob("./%s*" % run_name):
            fp = open(resultfile)
            measurements = json.load(fp)
            fp.close()
            resultset = resultfile[resultfile.index('_') + 1:]

            draw_frequency_distribution(measurements["block_frequency"],
                                        resultset,
                                        "Block")

            draw_record_time_curve(measurements["insert_times"],
                                   resultset,
                                   "insertion")
            draw_record_time_curve(measurements["query_times"],
                                   resultset,
                                   "query")

            draw_precision_recall_curve(measurements["y_true1"],
                                        measurements["y_scores"],
                                        resultset)

            # Show plots
            show()

    sys.exit(0)


if __name__ == "__main__":  # pragma: no cover
        main(prog_name="benchmark")
