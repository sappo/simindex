#!/usr/bin/env python3
import os
import glob
import re
import json
import click
import urwid
import subprocess
import itertools as it
from collections import defaultdict
from simindex import draw_precision_recall_curve, \
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
        for report in glob.glob("./reports/14*"):
            report = os.path.basename(report)
            reports[(report.split('_')[0],
                     report.split('_')[2])].append(report.split('_')[1])

        saved_reports = defaultdict(list)
        for report in glob.glob("./evaluation/*"):
            if os.path.basename(report).startswith("mprofile"):
                continue

            report = os.path.basename(report)
            saved_reports[(report.split('_')[0],
                           report.split('_')[2])].append(report.split('_')[1])

        while self.box_level > 0:
            self.original_widget = self.original_widget[0]
            self.box_level -= 1

        menu_elements = [
            self.result_menu(key[0], key[1],
                             value, './reports') for key, value in reports.items()
        ]
        menu_elements.extend([
            self.sub_menu(u'Saved reports ...', [
                self.result_menu(key[0], key[1],
                                 value, './evaluation') for key, value in saved_reports.items()
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
                              valign='middle', height=('relative', 90),
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

    def result_menu(self, run, dataset, indexer, prefix):
        btn_caption = "%s (%s) - %s" % (run, dataset, ', '.join(indexer))
        title = "Choose an option for %s!" % btn_caption
        contents = self.menu(title, [
                        self.menu_button(u'Metrics', self.item_chosen),
                        self.menu_button(u'Plots', self.show_plots),
                        self.sub_menu(u'Memory Usage', [
                            self.menu_button(idx, self.show_mprof) for idx in
                            sorted(it.chain(indexer, ['fit']))
                            ]),
                        self.save_menu(),
                        self.menu_button(u'Delete', self.delete_report),
                   ])
        def open_menu(button):
            self.prefix = prefix
            self.run = run
            self.dataset = dataset
            return self.open_box(contents)
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

    def item_chosen(self, button):
        metrics = self.read_metrics()

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
        for report in glob.glob("%s/*%s*" % (self.prefix, self.run)):
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
        for report in glob.glob("%s/*%s*" % (self.prefix, self.run)):
            os.remove(report)

        self.refresh_menu()

    def read_metrics(self):
        metrics = {}
        for resultfile in glob.glob("%s/%s*" % (self.prefix, self.run)):
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
        for resultfile in glob.glob("%s/%s*" % (self.prefix, self.run)):
            fp = open(resultfile)
            measurements = json.load(fp)
            fp.close()
            indexer_dataset = resultfile[resultfile.index('_') + 1:]
            indexer = indexer_dataset[:indexer_dataset.index('_')]
            dataset = indexer_dataset[indexer_dataset.index('_') + 1:]

            insert_times[dataset][indexer] = measurements["insert_times"]
            query_times[dataset][indexer] = measurements["query_times"]

            draw_precision_recall_curve(measurements["y_true"],
                                        measurements["y_scores"],
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
                dataset_matches = re.finditer(r'([a-zA-Z0-9-]+)[\w]+.csv', line)
                dataset_match = next(dataset_matches)
                dataset = dataset_match.group(1)

                prefix_matches = re.finditer(r'-r (\w+)', line)
                prefix_match = next(prefix_matches, None)
                prefix = prefix_match.group(1)

                type_matches = re.finditer(r'--run-type (\w+)', line)
                type_match = next(type_matches, None)
                if type_match:
                    run_type = type_match.group(1)

                idx_matches = re.finditer(r'-m (\w+)', line)
                idx_match = next(idx_matches, None)
                if idx_match:
                    indexer = idx_match.group(1)

    # Save memory consumption to resultfile
    if indexer:
        resultfile = "./reports/%s_%s_%s" % (prefix, indexer, dataset)
        with open(resultfile, mode='r') as fp:
            measurements = json.load(fp)
            measurements["build_memory_peak"] = peak_memory - python_overhead

        with open(resultfile, mode='w') as fp:
            json.dump(measurements, fp, sort_keys=True, indent=4)

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
    for mprofile in glob.glob("./reports/mprofile_2*"):
        parse_mprofile(mprofile)


    # Open menu with available reports
    top = JarvisMenu()
    urwid.MainLoop(top, palette=[('reversed', 'standout', '')]).run()

if __name__ == "__main__":  # pragma: no cover
        main(prog_name="jarvis")
