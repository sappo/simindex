#!/usr/bin/env python3
import os
import glob
import re
import json
import click
import urwid
import subprocess
import itertools as it
import numpy as np
from collections import defaultdict
from simindex.plot import draw_prc, \
                     draw_record_time_curve, \
                     draw_bar_chart, \
                     show
import warnings
warnings.filterwarnings("ignore")


def exit_program(button):
    raise urwid.ExitMainLoop()


class JarvisMenu(urwid.WidgetPlaceholder):

    def __init__(self):
        super(JarvisMenu, self).__init__(urwid.SolidFill(u'#'))
        self.box_level = 0
        self.max_box_levels = 4

        self.refresh_menu()

    def refresh_menu(self):
        reports = defaultdict(list)
        for report in sorted(glob.glob("./reports/*"), reverse=True):
            if os.path.basename(report).startswith("mprofile") \
                    or report.count('fit'):
                continue

            report = os.path.basename(report)
            reports[(report.split('_')[0],
                     report.split('_')[2])].append(report.split('_')[1])

        saved_reports = defaultdict(list)
        for report in sorted(glob.glob("./evaluation/*"), reverse=True):
            if os.path.basename(report).startswith("mprofile") \
                    or report.count('fit'):
                continue

            report = os.path.basename(report)
            saved_reports[(report.split('_')[0],
                           report.split('_')[2])].append(report.split('_')[1])

        while self.box_level > 0:
            self.original_widget = self.original_widget[0]
            self.box_level -= 1

        menu_elements = [
            self.result_menu(key[0], key[1], value, './reports',
                             reports, saved_reports)
            for key, value in reports.items()
        ]
        menu_elements.extend([
            self.sub_menu(u'Saved reports ...', [
                self.result_menu(key[0], key[1], value, './evaluation',
                                 reports, saved_reports)
                for key, value in saved_reports.items()
            ])
        ])
        menu_elements.append(self.menu_button(u'Quit', exit_program))

        self.top_menu = self.menu(u'Choose a report!', menu_elements)
        self.open_box(self.top_menu)

    def open_box(self, box):
        self.original_widget = \
                urwid.Overlay(urwid.LineBox(box),
                              self.original_widget,
                              align='center', width=('relative', 90),
                              valign='middle', height=('relative', 94),
                              min_width=24, min_height=8,
                              left=self.box_level * 3,
                              right=(self.max_box_levels - self.box_level - 1) * 3,
                              top=self.box_level * 2,
                              bottom=(self.max_box_levels - self.box_level - 1) * 2)
        self.box_level += 1

    def keypress(self, size, key):
        if (key == 'esc' or key == 'q') and self.box_level > 1:
            self.original_widget = self.original_widget[0]
            self.box_level -= 1
        elif (key == 'esc' or key == 'q') and self.box_level == 1:
            exit_program(None)
        elif key == 'j':
            return super(JarvisMenu, self).keypress(size, "down")
        elif key == 'k':
            return super(JarvisMenu, self).keypress(size, "up")
        else:
            return super(JarvisMenu, self).keypress(size, key)

    def menu(self, title, choices):
        body = [urwid.Text(title), urwid.Divider()]
        body.extend(choices)
        return urwid.ListBox(urwid.SimpleFocusListWalker(body))

    def result_menu(self, run, dataset, indexer, prefix,
                    reports, saved_reports):
        btn_caption = "%s (%s) - %s" % (run, dataset, ', '.join(indexer))
        title = "Choose an option for %s!" % btn_caption
        compare_elements = [
            self.compare_menu(key[0], key[1], value, './reports', 1)
            for key, value in reports.items()
        ]
        compare_elements.extend([
            self.sub_menu(u'Saved reports ...', [
                self.compare_menu(key[0], key[1], value, './evaluation', 2)
                for key, value in saved_reports.items()
            ])
        ])
        compare_contents = self.menu("Choose a report to compare!", compare_elements)
        def open_comparemenu(button):
            return self.open_box(compare_contents)

        contents = self.menu(title, [
                        self.menu_button(u'Model', self.model_info),
                        self.menu_button(u'Metrics', self.metrics_info),
                        self.menu_button(u'Plots', self.show_plots),
                        self.sub_menu(u'Memory Usage', [
                            self.menu_button(idx, self.show_mprof) for idx in
                            sorted(it.chain(indexer, ['fit']))
                            ]),
                        self.menu_button(u'Compare', open_comparemenu),
                        self.save_menu(),
                        self.menu_button(u'Delete', self.delete_report),
                   ])
        def open_menu(button):
            self.prefix = prefix
            self.run = run
            self.dataset = dataset
            return self.open_box(contents)
        return self.menu_button([btn_caption], open_menu)

    def compare_menu(self, run, dataset, indexer, prefix, level):
        btn_caption = "%s (%s) - %s" % (run, dataset, ', '.join(indexer))
        def open_menu(button):
            self.compareprefix = prefix
            self.comparerun = run
            self.comparedataset = dataset
            return self.comparison()
        return self.menu_button([btn_caption], open_menu)

    def save_menu(self):
        edit = urwid.Edit(u'Enter name for report: ')
        contents = self.menu("Save report", [
            edit,
            urwid.Divider(),
            self.menu_button("Save", self.save_report)
        ])
        def open_menu(button):
            self.edit = edit
            return self.open_box(contents)
        return self.menu_button(['Save'], open_menu)

    def sub_menu(self, caption, choices):
        contents = self.menu(caption, choices)
        def open_menu(button):
            return self.open_box(contents)
        return self.menu_button([caption], open_menu)

    def menu_button(self, caption, callback):
        button = urwid.Button(caption)
        urwid.connect_signal(button, 'click', callback)
        return urwid.AttrMap(button, None, focus_map='reversed')

    def model_info(self, button):
        model_report = "%s/%s_fit_%s" % (self.prefix, self.run, self.dataset)
        with open(model_report) as fp:
            model = json.load(fp)

        simlearner_P = '?'
        simlearner_N = '?'
        if "positive_filtered_labels" in model:
            simlearner_P = model["positive_filtered_labels"]
        if "negative_filtered_labels" in model:
            simlearner_N = model["negative_filtered_labels"]

        simrow = [urwid.Text(u'Similarities - P (%s), N (%s)' % (
                        simlearner_P, simlearner_N))]
        for index, similarity in enumerate(model["similarities"]):
            simrow.append(urwid.Text(u'Field %d: %s' % (index, similarity)))

        # Format blocking scheme
        blocking = defaultdict(list)
        for blocking_key in model["blocking_scheme"]:
            blocking[blocking_key[0]].append(blocking_key[1:])

        maxconjunction = max(len(blocking[x]) for x in blocking.keys())
        columns = [[] for x in range(maxconjunction)]
        for keyid in blocking.keys():
            curcol = 0
            while curcol < maxconjunction:
                if curcol < len(blocking[keyid]):
                    blocking_key = blocking[keyid][curcol]
                    text  = "Field     %s\n" % blocking_key[0]
                    text += "Encoder   %s\n" % blocking_key[1]
                    text += "Score     %s" % blocking_key[2]
                    columns[curcol].append(urwid.Divider('-'))
                    columns[curcol].append(urwid.Text([text]))
                else:
                    columns[curcol].append(urwid.Divider('-'))
                    columns[curcol].append(urwid.Divider())
                    columns[curcol].append(urwid.Divider())
                    columns[curcol].append(urwid.Divider())

                curcol += 1

        for index, col in enumerate(columns):
            columns[index] = urwid.Pile(col)

        blocking_P = '?'
        blocking_N = '?'
        if "positive_labels" in model:
            blocking_P = model["positive_labels"]
        if "negative_labels" in model:
            blocking_N = model["negative_labels"]

        self.open_box(
            urwid.ListBox([
                urwid.Text("Model."), urwid.Divider(),
                urwid.Columns(simrow), urwid.Divider(),
                urwid.Text("Blocking Scheme - P (%s), N (%s)" % (
                    blocking_P, blocking_N)),
                urwid.Divider(),
                urwid.Columns(columns),
                urwid.Divider(),
            ])
         )

    def comparison(self):
        metrics = self.read_metrics()
        othermetrics = self.read_metrics(True)

        mcolumns = self.metrics_columns(metrics)
        ocolumns = self.metrics_columns(othermetrics)

        columns = []
        for indexer in sorted(mcolumns.keys()):
            if indexer in ocolumns:
                columns.append(urwid.Pile(mcolumns[indexer]))
                columns.append(urwid.Pile(ocolumns[indexer]))

        self.open_box(
            urwid.ListBox([
                urwid.Text("Comparison."), urwid.Divider(),
                urwid.Columns(columns),
                urwid.Divider(),
            ])
        )

    def metrics_columns(self, metrics):
        columns_data = defaultdict(list)
        for indexer in metrics.keys():
            columns_data[indexer].append(urwid.Text([indexer]))
            columns_data[indexer].append(urwid.Divider())
            columns_data[indexer].append(urwid.Divider())

        for indexer, measurements in metrics.items():
            text =  "Index\n"
            text += "Pair completeness: %f\n" % measurements["pair_completeness"]
            text += "Reduction ratio:   %f\n" % measurements["reduction_ratio"]
            columns_data[indexer].append(urwid.Text([text]))
            text =  "Query\n"
            text += "Recall:            %f\n" % measurements["recall"]
            text += "Precision:         %f\n" % measurements["precision"]
            text += "F1-Score:          %f\n" % measurements["f1_score"]
            columns_data[indexer].append(urwid.Text([text]))
            text =  "Times\n"
            text += "Build time:        %f\n" % measurements["build_time"]
            text += "Build time sum:    %f\n" % measurements["build_time_sum"]
            text += "Query time:        %f\n" % measurements["query_time"]
            text += "Query time sum:    %f\n" % measurements["query_time_sum"]
            text += "Inserts mean:      %f\n" % measurements["inserts_mean"]
            text += "Queries mean:      %f\n" % measurements["queries_mean"]
            text += "Inserts (s):       %f\n" % measurements["inserts_sec"]
            text += "Queries (s):       %f\n" % measurements["queries_sec"]
            columns_data[indexer].append(urwid.Text([text]))
            text =  "Other\n"
            text += "Memory peak        %f\n" % measurements["build_memory_peak"]
            columns_data[indexer].append(urwid.Text([text]))

        return columns_data

    def metrics_info(self, button):
        metrics = self.read_metrics()
        columns_data = self.metrics_columns(metrics)
        columns = []
        for indexer in sorted(columns_data.keys()):
            columns.append(urwid.Pile(columns_data[indexer]))

        self.open_box(
            urwid.ListBox([
                urwid.Text("Metrics."), urwid.Divider(),
                urwid.Columns(columns),
                urwid.Divider(),
            ])
        )

    def save_report(self, button):
        name = self.edit.edit_text
        for report in glob.glob("%s/*%s*%s*" % (self.prefix, self.run, self.dataset)):
            reportname = os.path.basename(report)
            parts = reportname.split('_')
            if reportname.startswith('mprofile'):
                filename = 'mprofile_%s_%s' % (name, '_'.join(parts[2:]))
            else:
                filename = '%s_%s' % (name, '_'.join(parts[1:]))

            os.renames(report, "./evaluation/%s" % filename)

        if not os.path.exists(self.prefix):
            os.mkdir(self.prefix)

        self.refresh_menu()

    def delete_report(self, button):
        for report in glob.glob("%s/*%s*%s*" % (self.prefix, self.run, self.dataset)):
            os.remove(report)

        self.refresh_menu()

    def read_metrics(self, compare=False):
        metrics = {}
        if compare:
            prefix = self.compareprefix
            run = self.comparerun
            dataset = self.comparedataset
        else:
            prefix = self.prefix
            run = self.run
            dataset = self.dataset

        for resultfile in glob.glob("%s/%s*%s" % (prefix, run, dataset)):
            if resultfile.count("fit"):
                continue

            with open(resultfile) as fp:
                measurements = json.load(fp)

            indexer_dataset = resultfile[resultfile.index('_') + 1:]
            indexer = indexer_dataset[:indexer_dataset.index('_')]
            metrics[indexer] = measurements

        return metrics

    def show_mprof(self, button):
        indexer = button.label
        subprocess.Popen([
            '/bin/sh', '-c',
             'mprof plot %s/mprofile_%s_%s_%s.dat' % (self.prefix,
                                                      self.run,
                                                      indexer,
                                                      self.dataset)
        ])

    def show_plots(self, button):
        memory_usage = defaultdict(dict)
        index_build_time = defaultdict(dict)
        insert_times = defaultdict(dict)
        query_times = defaultdict(dict)
        for resultfile in glob.glob("%s/%s*%s" % (self.prefix, self.run, self.dataset)):
            if resultfile.count("fit"):
                continue

            fp = open(resultfile)
            measurements = json.load(fp)
            fp.close()
            indexer_dataset = resultfile[resultfile.index('_') + 1:]
            indexer = indexer_dataset[:indexer_dataset.index('_')]
            dataset = indexer_dataset[indexer_dataset.index('_') + 1:]

            insert_times[dataset][indexer] = measurements["insert_times"]
            query_times[dataset][indexer] = measurements["query_times"]

            draw_prc(np.array(measurements["prc_precisions"]),
                     np.array(measurements["prc_recalls"]),
                     np.array(measurements["prc_thresholds"]),
                     indexer_dataset)

            # Sort data by indexer and then by dataset
            memory_usage[indexer][dataset] = measurements["build_memory_peak"]
            index_build_time[indexer][dataset] = measurements["build_time"]

        for dataset in insert_times.keys():
            draw_record_time_curve(insert_times[dataset], dataset, "insertion")

        for dataset in query_times.keys():
            draw_record_time_curve(query_times[dataset], dataset, "query")

        draw_bar_chart(memory_usage, "Memory usage", "MiB")
        draw_bar_chart(index_build_time, "Index build time", "Seconds (s)")

        # Show plots
        show()


def parse_mprofile(filename):
    # Make sure mprofile hasn't been processed
    if filename.count('_') > 1:
        return

    python_overhead = 0
    peak_memory = 0
    prefix = None
    indexer = None
    run_type = None
    dataset = None
    with open(filename) as handle:
        for line in handle.readlines():
            line_parts = line.split(' ')
            if line_parts[0] == 'FUNC':
                function = line_parts[1]
                if function == '__main__.main':
                    python_overhead = float(line_parts[2])
                else:
                    function_peak_memory = float(line_parts[4])
                    if function_peak_memory > peak_memory:
                        peak_memory = function_peak_memory

            elif line_parts[0] == 'CMDLINE':
                dataset_matches = re.finditer(r'([^/_]+)\w+.csv', line)
                dataset_match = next(dataset_matches)
                dataset = dataset_match.group(1)

                prefix_matches = re.finditer(r'-r ([^ ]+)', line)
                prefix_match = next(prefix_matches, None)
                prefix = prefix_match.group(1)

                type_matches = re.finditer(r'--run-type ([^ ]+)', line)
                type_match = next(type_matches, None)
                if type_match:
                    run_type = type_match.group(1)

                idx_matches = re.finditer(r'-m ([^ ]+)', line)
                idx_match = next(idx_matches, None)
                if idx_match:
                    indexer = idx_match.group(1)

    # Save memory consumption to resultfile
    if indexer:
        resultfile = "./reports/%s_%s_%s" % (prefix, indexer, dataset)
        try:
            with open(resultfile, mode='r') as fp:
                measurements = json.load(fp)
                measurements["build_memory_peak"] = peak_memory - python_overhead

            with open(resultfile, mode='w') as fp:
                json.dump(measurements, fp, sort_keys=True, indent=4)
        except FileNotFoundError:
            os.remove(filename)
            return

    # Modify memory usage report to only contain useful information
    if indexer:
        parsed_filename = "./reports/mprofile_%s_%s_%s.dat" % (prefix, indexer, dataset)
    else:
        parsed_filename = "./reports/mprofile_%s_%s_%s.dat" % (prefix, run_type, dataset)

    with open(filename) as handle:
        with open(parsed_filename, mode='w') as whandle:
            for line in handle.readlines():
                line_parts = line.split(' ')
                if line_parts[0] == 'FUNC':
                    function = line_parts[1]
                    if function == '__main__.main':
                        continue
                if line_parts[0] == 'MEM':
                    memory = float(line_parts[1])
                    if memory < python_overhead or memory > peak_memory:
                        continue

                whandle.write(line)

    os.remove(filename)


@click.command(context_settings=dict(help_option_names=[u'-h', u'--help']))
def main():

    # Parse memory usage output
    for mprofile in glob.glob("./reports/mprofile*"):
        parse_mprofile(mprofile)


    # Open menu with available reports
    top = JarvisMenu()
    urwid.MainLoop(top, palette=[('reversed', 'standout', '')]).run()

if __name__ == "__main__":  # pragma: no cover
        main(prog_name="jarvis")
