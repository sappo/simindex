#!/usr/bin/env python3
import os
import sys
import glob
import re
import json
import click
import concurrent.futures
from datetime import datetime, timedelta
from pathlib import Path
import urwid
import subprocess
import itertools as it
import numpy as np
from collections import defaultdict
import simindex.plot as splt

import warnings
warnings.filterwarnings("ignore")


def exit_program(button):
    raise urwid.ExitMainLoop()


class JarvisMenu(urwid.WidgetPlaceholder):

    def __init__(self):
        super(JarvisMenu, self).__init__(urwid.SolidFill(u'#'))
        self.box_level = 0
        self.max_box_levels = 4
        self.edit_mode = False
        self.selected_reports = []
        self.close_on_level = []
        self.quick_compare_mode = False

        self.refresh_menu()

    def set_mainloop(self, mainloop):
        self.mainloop = mainloop
        self.mainloop.set_alarm_in(0, self.process_mprofiles)

    def process_mprofiles(self, mainloop, _):
        self.open_box(self.menu('Processing memory profiles...', []), small=True)
        self.mainloop.draw_screen()

        # Parse memory usage output
        for mprofile in glob.glob("./reports/mprofile*"):
            parse_mprofile(mprofile)

        self.close_box()

    def refresh_menu(self):
        self.selected_reports.clear()
        self.selected_reports.clear()
        reports = defaultdict(list)
        for report in sorted(glob.glob("./reports/*"), reverse=True):
            if os.path.basename(report).startswith("mprofile") \
                    or report.count('fit'):
                continue

            report = os.path.basename(report)
            reports[(report.split('_')[0],
                     report.split('_')[2])].append(report.split('_')[1])

        saved_reports = defaultdict(list)
        for report in sorted(filter(lambda f: os.path.isfile(f), glob.glob("./evaluation/*")), reverse=True):
            if os.path.basename(report).startswith("mprofile") \
                    or report.count('fit'):
                continue

            report = os.path.basename(report)
            saved_reports[(report.split('_')[0],
                           report.split('_')[2])].append(report.split('_')[1])

        evaldirs = [x[0] for x in os.walk("./evaluation") if os.path.isdir(x[0]) and x[0] != "./evaluation"]

        its_reports = {}
        for dir in sorted(evaldirs):
            maschine = os.path.basename(dir)
            its_reports[maschine] = defaultdict(list)
            its_report = its_reports[maschine]
            for report in sorted(glob.glob("%s/*" % dir), reverse=True):
                if os.path.basename(report).startswith("mprofile") \
                        or report.count('fit'):
                    continue

                report = os.path.basename(report)
                its_report[(report.split('_')[0],
                            report.split('_')[2])].append(report.split('_')[1])

        while self.box_level > 0:
            self.close_box()

        menu_elements = [
            self.result_menu(key[0], key[1], value, './reports',
                             reports, saved_reports, its_reports)
            for key, value in sorted(reports.items(), reverse=True)
        ]
        menu_elements.extend([
            self.sub_menu(u'Saved reports (%d) ...' % len(saved_reports), [
                self.result_menu(key[0], key[1], value, './evaluation',
                                 reports, saved_reports, its_reports)
                for key, value in sorted(saved_reports.items(), key=lambda i: i[0])
            ])
        ])

        evaldirs = [x[0] for x in os.walk("./evaluation") if os.path.isdir(x[0]) and x[0] != "./evaluation"]
        for dir in sorted(evaldirs):
            machine = os.path.basename(dir)
            its_report = its_reports[machine]
            menu_elements.extend([
                self.sub_menu(u'%s reports (%d) ...' % (machine, len(its_report)), [
                    self.result_menu(key[0], key[1], value, './evaluation/%s' % machine,
                                     reports, saved_reports, its_reports)
                    for key, value in sorted(its_report.items(), reverse=True)
                ])
            ])

        menu_elements.extend([
            self.sub_menu(u'Run remote evaluation ...', [
                self.machines_menu(u'Baseline', "eval_baseline_ncvoter.sh"),
                self.machines_menu(u'Classifier', "eval_classifier_ncvoter.sh"),
                self.machines_menu(u'GT vs no GT', "eval_gtvsnogt_ncvoter.sh"),
                self.machines_menu(u'Similarities', "eval_simeffect_ncvoter.sh"),
                self.machines_menu(u'Similarities (linear)',
                                   "eval_simeffect_ncvoter.sh --clf=svmlinear"),
                self.machines_menu(u'Similarities (rbf)',
                                   "eval_simeffect_ncvoter.sh --clf=svmrbf"),
                self.machines_menu(u'Similarities (dt)',
                                   "eval_simeffect_ncvoter.sh --clf=decisiontree"),
                self.machines_menu(u'FusionLearner (linear)',
                                   "eval_fusionlearn_ncvoter.sh --clf=svmlinear"),
                self.machines_menu(u'FusionLearner (rbf)',
                                   "eval_fusionlearn_ncvoter.sh --clf=svmrbf"),
                self.machines_menu(u'FusionLearner (dt)',
                                   "eval_fusionlearn_ncvoter.sh --clf=decisiontree"),
                self.menu_button(u'Grep Data', self.machines_grep_data),
            ])
        ])
        menu_elements.append(self.menu_button(u'Quit', exit_program))

        self.top_menu = self.menu(u'Choose a report!', menu_elements)
        self.open_box(self.top_menu)

    def machines_grep_data(self, button):
        self.open_box(self.menu('Checking for data (0/19)', []), small=True)
        self.mainloop.draw_screen()

        def do_rsync(machine):
            cmd = "ssh -o ProxyCommand=\"ssh -W %%h:%%p ksapp002@login1.cs.hs-rm.de\" ksapp002@%s ps ax | grep [a]nalyzer.py" % machine
            with subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE) as proc:
                stdout = proc.stdout.read()

            if not stdout:
                cmd2 = "rsync -avz --remove-source-files -e 'ssh -o ProxyCommand=\"ssh -W %%h:%%p ksapp002@login1.cs.hs-rm.de\"' ksapp002@%s:/its/ksapp002/reports/ evaluation/%s"
                retcode = subprocess.call([cmd2 % (machine, machine)], shell=True, stdout=subprocess.DEVNULL)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for no in np.arange(19):
                machine = "its%s" % str(no).zfill(2)
                futures.append(executor.submit(do_rsync, machine))

            counter = 0
            for future in concurrent.futures.as_completed(futures):
                self.close_box()
                self.open_box(self.menu('Checking for data (%d/19)' % counter, []), small=True)
                self.mainloop.draw_screen()
                counter += 1

        self.close_box()


    def machines_menu(self, label, script):
        def open_menu(button):
            self.script = script
            self.machines = set()
            def check_machine(checkbox, state):
                if state == True:
                    self.machines.add(checkbox.label)
                else:
                    self.machines.remove(checkbox.label)

            self.open_box(self.menu('Checking for WIP (0/19)', []), small=True)
            self.mainloop.draw_screen()

            def is_wip(machine):
                cmd = "ssh -o ProxyCommand=\"ssh -W %%h:%%p ksapp002@login1.cs.hs-rm.de\" ksapp002@%s ps ax | grep [a]nalyzer.py" % machine
                with subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE) as proc:
                    return proc.stdout.read()

            checkboxes = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_stdout = {}
                for no in np.arange(19):
                    machine = "its%s" % str(no).zfill(2)
                    future_to_stdout[executor.submit(is_wip, machine)] = machine

                counter = 0
                for future in concurrent.futures.as_completed(future_to_stdout):
                    machine = future_to_stdout[future]
                    stdout = future.result()
                    if not stdout:
                        checkboxes[machine] = urwid.CheckBox(machine, on_state_change=check_machine)

                    self.close_box()
                    self.open_box(self.menu('Checking for WIP (%d/19)' % counter, []), small=True)
                    self.mainloop.draw_screen()
                    counter += 1

                self.close_box()

            contents = []
            contents.extend([checkboxes[machine] for machine in sorted(checkboxes.keys())])
            contents.append(self.menu_button("Run", self.run_evaluation))
            menu = self.menu("Choose the machines to run the evaluation on!", contents)
            return self.open_box(menu)

        return self.menu_button([label], open_menu)

    def run_evaluation(self, button):
        for machine in self.machines:
            machine_report_dir = './evaluation/%s' % machine
            if not os.path.exists(machine_report_dir):
                os.mkdir(machine_report_dir)

            cmd = "ssh -o \"ProxyCommand=ssh -W %%h:%%p ksapp002@login1.cs.hs-rm.de\" ksapp002@%s -t \"nohup bash -ic '( ( cd ~/workspace/simindex > /dev/null 2>&1 && ./%s > /dev/null 2>&1 ) & )'\""
            retcode = subprocess.call([cmd % (machine, self.script)], shell=True)
            if retcode == 0:
                pass #OK
            else:
                pass #Wrong

        self.close_box()

    def open_box(self, box, small=False):
        width = 90
        height = 94
        if small:
            width = 30
            height = 34
        footer = urwid.Text("(%s) -  [t] Toggle Mode, [c] Compare Menu"
                    % ("Single-Mode" if not self.quick_compare_mode else "Multi-Mode"))
        frame = urwid.Frame(box, footer=footer)
        self.original_widget = \
            urwid.Overlay(urwid.AttrMap(urwid.LineBox(frame), '', 'line'),
                          self.original_widget,
                          align='left', width=('relative', width),
                          valign='middle', height=('relative', height),
                          min_width=24, min_height=8,
                          left=(self.box_level + 1) * 3,
                          right=(self.max_box_levels - self.box_level - 1) * 3,
                          top=self.box_level * 2,
                          bottom=(self.max_box_levels - self.box_level - 1) * 2)
        self.box_level += 1

    def close_box(self):
        self.original_widget = self.original_widget[0]
        self.box_level -= 1
        if self.box_level in self.close_on_level:
            if len(self.close_on_level) > 0:
                del self.close_on_level[-1]
            if len(self.selected_reports) > 0:
                del self.selected_reports[-1]

    def keypress(self, size, key):
        if self.edit_mode:
            if key == 'esc':
                self.edit_mode = False
                self.mainloop.screen.register_palette_entry('line', '', '')
                self.mainloop.screen.clear()
            elif key == '_':
                # Forbidden in filenames
                return
            else:
                return super(JarvisMenu, self).keypress(size, key)
        else:
            if (key == 'esc' or key == 'q') and self.box_level > 1:
                self.close_box()
            elif (key == 'esc' or key == 'q') and self.box_level == 1:
                exit_program(None)
            elif key == 'c':
                return self.open_quick_compare_menu()
            elif key == 'i':
                self.edit_mode = True
                self.mainloop.screen.register_palette_entry('line', 'dark red', '')
                self.mainloop.screen.clear()
            elif key == 'j':
                return super(JarvisMenu, self).keypress(size, "down")
            elif key == 'k':
                return super(JarvisMenu, self).keypress(size, "up")
            elif key == 't':
                self.quick_compare_mode = not self.quick_compare_mode
                return self.refresh_menu()
            else:
                return super(JarvisMenu, self).keypress(size, key)

    def menu(self, title, choices):
        body = [urwid.Text(title), urwid.Divider()]
        body.extend(choices)
        return urwid.ListBox(urwid.SimpleFocusListWalker(body))

    def result_menu(self, run, dataset, indexer, prefix,
                    reports, saved_reports, its_reports):
        short_info = self.model_info_short(prefix, run, dataset)
        btn_caption = "%s    %s (%s) - %s" % (run.ljust(31), short_info, dataset, ', '.join(indexer))
        def open_menu(button):
            title = "Choose an option for %s!" % btn_caption
            def open_comparemenu(button):
                compare_contents = self.compare_menu(reports, saved_reports, its_reports, 1)
                return self.open_box(compare_contents)

            contents = self.menu(title, [
                            self.menu_button(u'Model', self.model_info),
                            self.menu_button(u'Metrics', self.metrics_info),
                            self.plot_menu(u'Show Plots', self.show_plots),
                            self.plot_menu(u'Save Plots', self.save_plots),
                            self.sub_menu(u'Memory Usage', [
                                self.menu_button(idx, self.show_mprof) for idx in
                                sorted(it.chain(indexer, ['fit']))
                                ]),
                            self.menu_button(u'Compare', open_comparemenu),
                            self.save_menu(),
                            self.menu_button(u'Delete', self.delete_report),
                       ])
            self.selected_reports.append((prefix, run, dataset))
            self.close_on_level.append(self.box_level)
            return self.open_box(contents)

        def select_report(checkbox, state, user_data):
            if state:
                self.selected_reports.append(user_data)
            else:
                self.selected_reports.remove(user_data)

        if self.quick_compare_mode:
            return  urwid.CheckBox([btn_caption], on_state_change=select_report,
                                   user_data=(prefix, run, dataset))
        else:
            return self.menu_button([btn_caption], open_menu)

    def open_quick_compare_menu(self):
        title = "Compare %d choosen reports" % len(self.selected_reports)
        if len(self.selected_reports):
            contents = [
                            self.menu_button(u'Metrics', self.metrics_info),
                            self.plot_menu(u'Show Plots', self.show_plots),
                            self.plot_menu(u'Save Plots', self.save_plots),
                       ]
        else:
            contents = []

        return self.open_box(self.menu(title, contents))

    def compare_menu(self, reports, saved_reports, its_reports, level):
        compare_elements = [
            self.compare_menu_button(key[0], key[1], value, './reports', level,
                reports, saved_reports, its_reports)
            for key, value in sorted(reports.items(), reverse=True)
        ]
        compare_elements.extend([
            self.sub_menu(u'Saved reports ...', [
                self.compare_menu_button(key[0], key[1], value, './evaluation', level,
                    reports, saved_reports, its_reports)
                for key, value in sorted(saved_reports.items(), key=lambda i: i[0])
            ])
        ])
        for machine in sorted(its_reports.keys()):
            its_report = its_reports[machine]
            compare_elements.extend([
                self.sub_menu(u'%s reports (%d) ...' % (machine, len(its_report)), [
                    self.result_menu(key[0], key[1], value, './evaluation/%s' % machine,
                                     reports, saved_reports, its_reports)
                    for key, value in sorted(its_report.items(), reverse=True)
                ])
            ])
        return self.menu("Choose a report to compare!", compare_elements)


    def compare_menu_button(self, run, dataset, indexer, prefix, level,
                            reports, saved_reports, its_reports):
        def open_menu(button):
            def open_comparemenu(button):
                compare_contents = self.compare_menu(reports, saved_reports, its_reports, level + 1)
                return self.open_box(compare_contents)

            contents = self.menu("Compare with - %s" % btn_caption, [
                            self.menu_button(u'Model', self.model_info),
                            self.menu_button(u'Metrics', self.metrics_info),
                            self.plot_menu(u'Show Plots', self.show_plots),
                            self.plot_menu(u'Save Plots', self.save_plots),
                            self.menu_button(u'Compare', open_comparemenu),
                       ])

            self.selected_reports.append((prefix, run, dataset))
            self.close_on_level.append(self.box_level)
            return self.open_box(contents)

        short_info = self.model_info_short(prefix, run, dataset)
        btn_caption = "%s   %s (%s) - %s" % (run.ljust(31), short_info, dataset, ', '.join(indexer))
        return self.menu_button([btn_caption], open_menu)

    def save_menu(self):
        def open_menu(button):
            edit = urwid.Edit(u'Enter name for report: ')
            contents = self.menu("Save report", [
                edit,
                urwid.Divider(),
                self.menu_button("Save", self.save_report)
            ])
            self.edit = edit
            return self.open_box(contents)

        return self.menu_button(['Save'], open_menu)

    def plot_menu(self, label, callback):
        def open_menu(button):
            self.plot_combine = urwid.Edit(u'[PRC] Combine reports into one PRC: ')
            self.plot_combine.set_edit_text("1")
            self.prc_repeat_colors = urwid.CheckBox("[PRC] Repeat Colors")
            self.bar_landscape = urwid.CheckBox("[Bar] Landscape Mode")

            self.plots = set()
            def check_plots(checkbox, state):
                if state == True:
                    self.plots.add(checkbox.label)
                else:
                    self.plots.remove(checkbox.label)

            checkboxes = []
            checkboxes.append(urwid.CheckBox("PRC", on_state_change=check_plots))
            checkboxes.append(urwid.CheckBox("Inserts", on_state_change=check_plots))
            checkboxes.append(urwid.CheckBox("Queries", on_state_change=check_plots))
            checkboxes.append(urwid.CheckBox("Insert Time", on_state_change=check_plots))
            checkboxes.append(urwid.CheckBox("Inserts per second", on_state_change=check_plots))
            checkboxes.append(urwid.CheckBox("Query Time", on_state_change=check_plots))
            checkboxes.append(urwid.CheckBox("Queries per second", on_state_change=check_plots))
            checkboxes.append(urwid.CheckBox("Memory Usage", on_state_change=check_plots))
            for box in checkboxes:
                box.toggle_state()

            elements = [self.plot_combine, self.prc_repeat_colors,
                        self.bar_landscape, urwid.Divider()]
            elements.extend(checkboxes)
            elements.extend([
                urwid.Divider(),
                self.menu_button("Okay", callback)
            ])
            contents = self.menu("Plot settings", elements)
            return self.open_box(contents)

        return self.menu_button([label], open_menu)

    def sub_menu(self, caption, choices):
        def open_menu(button):
            contents = self.menu(caption, choices)
            return self.open_box(contents)

        return self.menu_button([caption], open_menu)

    def menu_button(self, caption, callback):
        button = urwid.Button(caption)
        urwid.connect_signal(button, 'click', callback)
        return urwid.AttrMap(button, None, focus_map='reversed')

    def model_info_short(self, prefix, run, dataset):
        model_report = "%s/%s_fit_%s" % (prefix, run, dataset)
        with open(model_report) as fp:
            model = json.load(fp)

        if model.get("use_classifier", True):
            if model.get("clf", None):
                clf_info = "clf" + model.get("clf", "")
            else:
                clf_info = "clflearn"
        else:
            clf_info = "noclf"

        if model.get("use_fullvector", False):
            vec_info = "full"
        elif model.get("use_parfullvector", False):
            vec_info = "par"
        else:
            vec_info = "var "

        maxbs = model.get("max_blocksize", 100)
        minrt = model.get("min_goodratio", 0.9)
        bs_info = "bs(%d, %.2f)" % (maxbs, minrt)

        if model.get("use_gt_matches", True):
            gt_info = "gt              "
        else:
            thresholds = model.get("gt_lbl_thresholds", (0.1, 0.6))
            gt_info = "nogt(%.2f, %.2f)" % (thresholds[0], thresholds[1])

        if model.get("similarity", None):
            sim_info = "sim" + model.get("similarity", "")
        else:
            sim_info = "simlearn"

        return clf_info.ljust(19) + vec_info.ljust(8) + gt_info.ljust(20) + bs_info.ljust(15) + sim_info.ljust(20)


    def model_info(self, button):
        prefix, run, dataset = self.selected_reports[-1]
        model_report = "%s/%s_fit_%s" % (prefix, run, dataset)
        with open(model_report) as fp:
            model = json.load(fp)

        simlearner_P = '?'
        simlearner_N = '?'
        if "positive_filtered_labels" in model:
            simlearner_P = model["positive_filtered_labels"]
        if "negative_filtered_labels" in model:
            simlearner_N = model["negative_filtered_labels"]

        simrow = [urwid.Text(u'Similarities\nP (%s), N (%s)' % (
                        simlearner_P, simlearner_N))]
        for index, similarity in enumerate(model["similarities"]):
            if type(similarity) == str:
                simrow.append(urwid.Text(u'Field %d:\n%s' % (index, similarity)))
            else:
                simrow.append(urwid.Text(u'%s\n%s' % (similarity[0], similarity[1])))

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

        text = ""
        clf_result_grid = model.get("clf_result_grid", None)
        if clf_result_grid:
            for classifier in clf_result_grid.keys():
                text += "  %s.\n" % classifier
                for mean, std, params in clf_result_grid[classifier]:
                    text += "  %0.3f (+/-%0.03f) for %r\n" % (mean, std * 2, params)

        result_grid = urwid.Text([text])

        gt_text = "%r" % model.get("use_gt_matches", True)
        if not model.get("use_gt_matches", True):
            thresholds = model.get("gt_lbl_thresholds", (0.1, 0.6))
            gt_text += " (%.2f, %.2f)" % (thresholds[0], thresholds[1])

        self.open_box(
            urwid.ListBox([
                urwid.Text("Model."),
                urwid.Divider(),
                urwid.Text(self.model_info_short(prefix, run, dataset)),
                urwid.Divider(),
                urwid.Text("Fit time: %dd %dh %dm %ds %dms" % time_to_hms(model.get("fit_time", "N/A"))),
                urwid.Divider(),
                urwid.Text("Ground Truth - P (%s), N (%s) - with Matches? %s" % (blocking_P, blocking_N, gt_text)),
                urwid.Divider(),
                urwid.Columns(simrow),
                urwid.Divider(),
                urwid.Text("Fusion Learner - %s = %s" % (model.get("best_classifier", "N/A"), model.get("best_params", "N/A"))),
                result_grid,
                urwid.Text("Blocking Scheme - max disjunction: %d, max conjunction: %d" %
                    (model.get("disjunctions", int(3)), model.get("conjunctions", int(2)))),
                urwid.Divider(),
                urwid.Columns(columns),
            ])
         )

    def metrics_info(self, button):
        metrics_columns = []
        for prefix, run, dataset in self.selected_reports:
            metrics = self.read_metrics(prefix, run, dataset)
            metrics_columns.append(self.metrics_columns(metrics))

        common_indexers = None
        for mcolumns in metrics_columns:
            if not common_indexers:
                common_indexers = set(mcolumns.keys())
            else:
                common_indexers.update(set(mcolumns.keys()))

        columns = []
        for indexer in sorted(common_indexers):
            for mcolumns in metrics_columns:
                if indexer in mcolumns:
                    columns.append(urwid.Pile(mcolumns[indexer]))

        self.open_box(
            urwid.ListBox([
                urwid.Text("Metrics."), urwid.Divider(),
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
            text += "Pairs completeness: %f\n" % measurements["pair_completeness"]
            text += "Pairs  quality:     %f\n" % measurements.get("pairs_quality", float('nan'))
            text += "Reduction ratio:    %f\n" % measurements["reduction_ratio"]
            columns_data[indexer].append(urwid.Text([text]))
            text =  "Query\n"
            text += "Recall:             %f\n" % measurements["recall"]
            text += "Precision:          %f\n" % measurements["precision"]
            text += "F1-Score:           %f\n" % measurements["f1_score"]
            text += "Average Precsion    %f\n" % measurements.get("average_precision", float('nan'))
            columns_data[indexer].append(urwid.Text([text]))
            text =  "Times\n"
            text += "Build time:         %dd %dh %dm %ds %dms\n" % time_to_hms(measurements["build_time"])
            text += "Build time sum:     %dd %dh %dm %ds %dms\n" % time_to_hms(measurements["build_time_sum"])
            text += "Query time:         %dd %dh %dm %ds %dms\n" % time_to_hms(measurements["query_time"])
            text += "Query time sum:     %dd %dh %dm %ds %dms\n" % time_to_hms(measurements["query_time_sum"])
            text += "Inserts mean:       %f\n" % measurements["inserts_mean"]
            text += "Queries mean:       %f\n" % measurements["queries_mean"]
            text += "Inserts (s):        %.0f\n" % measurements["inserts_sec"]
            text += "Queries (s):        %.0f\n" % measurements["queries_sec"]
            columns_data[indexer].append(urwid.Text([text]))
            text =  "Classification\n"
            text += "Best Classifier       %s\n"   % measurements['model'].get("best_classifier", "N/A")
            text += "Best Params           %s\n"   % measurements['model'].get("best_params", "N/A")
            text += "Best Score            %s\n"   % measurements['model'].get("best_score", "N/A")
            text += "Use Classifier        %s\n"   % measurements['model'].get("use_classifier", "N/A")
            text += "Use Full Simvector    %s\n"   % measurements['model'].get("use_fullvector", "False")
            text += "Use Partial Simvector %s\n"   % measurements['model'].get("use_parfullvector", "False")
            columns_data[indexer].append(urwid.Text([text]))
            text =  "Other\n"
            text += "Memory peak (MB)    %.2f" % measurements.get("build_memory_peak", float('nan'))
            columns_data[indexer].append(urwid.Text([text]))

        return columns_data

    def save_report(self, button):
        name = self.edit.edit_text
        prefix, run, dataset = self.selected_reports[-1]
        for report in glob.glob("%s/*%s_*%s*" % (prefix, run, dataset)):
            reportname = os.path.basename(report)
            parts = reportname.split('_')
            if reportname.startswith('mprofile'):
                filename = 'mprofile_%s_%s' % (name, '_'.join(parts[2:]))
            else:
                filename = '%s_%s' % (name, '_'.join(parts[1:]))

            os.renames(report, "./evaluation/%s" % filename)

        if not os.path.exists(prefix):
            os.mkdir(prefix)

        self.refresh_menu()

    def delete_report(self, button):
        prefix, run, dataset = self.selected_reports[-1]
        for report in glob.glob("%s/*%s_*%s*" % (prefix, run, dataset)):
            os.remove(report)

        self.refresh_menu()

    def read_metrics(self, prefix, run, dataset):
        metrics = {}
        model = ""
        for resultfile in glob.glob("%s/%s_*%s" % (prefix, run, dataset)):
            if resultfile.count("fit"):
                with open(resultfile) as fp:
                    model = json.load(fp)

        for resultfile in glob.glob("%s/%s_*%s" % (prefix, run, dataset)):
            if resultfile.count("fit"):
                continue

            with open(resultfile) as fp:
                measurements = json.load(fp)

            indexer_dataset = resultfile[resultfile.index('_') + 1:]
            indexer = indexer_dataset[:indexer_dataset.index('_')]
            measurements['model'] = model
            metrics[indexer] = measurements

        return metrics

    def show_mprof(self, button):
        indexer = button.label
        prefix, run, dataset = self.selected_reports[-1]
        subprocess.Popen([
            '/bin/sh', '-c',
             'mprof plot %s/mprofile_%s_%s_%s.dat' % (prefix,
                                                      run,
                                                      indexer,
                                                      dataset)
        ])

    def save_plots(self, button):
        self.open_box(self.menu('Saving plots...', []), small=True)
        self.mainloop.draw_screen()
        self.close_box()
        self.draw_plots(save=True)
        self.open_box(self.menu('Plots have been saved!', []), small=True)

    def show_plots(self, button):
        self.open_box(self.menu('Rendering plots (close all to continue)', []), small=True)
        self.mainloop.draw_screen()
        self.draw_plots()
        self.close_box()

    def draw_plots(self, save=False):
        memory_usage = defaultdict(lambda: defaultdict(dict))
        inserts_per_second = defaultdict(lambda: defaultdict(dict))
        queries_per_second = defaultdict(lambda: defaultdict(dict))
        index_build_time = defaultdict(lambda: defaultdict(dict))
        query_build_time = defaultdict(lambda: defaultdict(dict))
        insert_times = defaultdict(lambda: defaultdict(dict))
        query_times = defaultdict(lambda: defaultdict(dict))
        prc_curves = defaultdict(lambda: defaultdict(dict))

        for prefix, run, dataset in self.selected_reports:
            metrics = self.read_metrics(prefix, run, dataset)
            for indexer, measurements in metrics.items():
                insert_times[dataset][run][indexer] = measurements["insert_times"]
                query_times[dataset][run][indexer] = measurements["query_times"]

                if measurements['model'].get("use_classifier", True):
                    prc_curves[dataset][run][indexer] = (
                        np.array(measurements["prc_precisions"]),
                        np.array(measurements["prc_recalls"]),
                        np.array(measurements["prc_thresholds"]),
                        measurements["recall"],
                        measurements["precision"]
                    )
                else:
                    prc_curves[dataset][run][indexer] = (
                        np.empty(0),
                        np.empty(0),
                        np.empty(0),
                        measurements["recall"],
                        measurements["precision"]
                    )

                # Sort data by indexer and then by dataset
                memory_usage[dataset][indexer][run] = measurements.get("build_memory_peak", float('nan'))
                inserts_per_second[dataset][indexer][run] = measurements["inserts_sec"]
                queries_per_second[dataset][indexer][run] = measurements["queries_sec"]
                index_build_time[dataset][indexer][run] = measurements["build_time_sum"]
                query_build_time[dataset][indexer][run] = measurements["query_time_sum"]

        picture_names = []
        if "PRC" in self.plots:
            for dataset in prc_curves.keys():
                splt.draw_prc(prc_curves[dataset], dataset,
                              mod=int(self.plot_combine.edit_text),
                              repeat_colors=self.prc_repeat_colors.state)
                picture_names.append("%s_prc" % '_'.join(prc_curves[dataset].keys()))

        if "Inserts" in self.plots:
            for dataset in insert_times.keys():
                splt.draw_record_time_curve(insert_times[dataset], dataset, "insertion")
                picture_names.append("%s_tc_insert" % '_'.join(insert_times[dataset].keys()))

        if "Queries" in self.plots:
            for dataset in query_times.keys():
                splt.draw_record_time_curve(query_times[dataset], dataset, "query")
                picture_names.append("%s_tc_query" % '_'.join(query_times[dataset].keys()))

        if "Memory Usage" in self.plots:
            for dataset in memory_usage.keys():
                splt.draw_bar_chart(memory_usage[dataset],
                               "Memory usage (%s)" % dataset, "MiB",
                               landscape=self.bar_landscape.state)
                picture_names.append("%s_memusg" % '_'.join(memory_usage[dataset].keys()))

        if "Insert Time" in self.plots:
            for dataset in index_build_time.keys():
                splt.draw_bar_chart(index_build_time[dataset],
                               "Index build time (%s)" % dataset, "Seconds (s)",
                               landscape=self.bar_landscape.state)
                picture_names.append("%s_index_bt" % '_'.join(index_build_time[dataset].keys()))

        if "Query Time" in self.plots:
            for dataset in query_build_time.keys():
                splt.draw_bar_chart(query_build_time[dataset],
                               "Query time (%s)" % dataset, "Seconds (s)",
                               landscape=self.bar_landscape.state)
                picture_names.append("%s_query_bt" % '_'.join(index_build_time[dataset].keys()))

        if "Inserts per second" in self.plots:
            for dataset in inserts_per_second.keys():
                splt.draw_bar_chart(inserts_per_second[dataset],
                               "Inserts per second (%s)" % dataset, "Seconds (s)",
                               landscape=self.bar_landscape.state)
                picture_names.append("%s_index_ips" % '_'.join(index_build_time[dataset].keys()))

        if "Queries per second" in self.plots:
            for dataset in queries_per_second.keys():
                splt.draw_bar_chart(queries_per_second[dataset],
                               "Queries per second (%s)" % dataset, "Seconds (s)",
                               landscape=self.bar_landscape.state)
                picture_names.append("%s_query_ips" % '_'.join(index_build_time[dataset].keys()))

        # Show plots
        if not save:
            splt.show()
        else:
            splt.save(picture_names)


def time_to_hms(time):
    if not time == "N/A":
        sec = timedelta(seconds=time)
        d = datetime(1,1,1) + sec
        days = d.day - 1
        hours = d.hour
        minutes = d.minute
        seconds = d.second
        microsecond = d.microsecond / 1000
    else:
        days = "N/A"
        hours = "N/A"
        minutes = "N/A"
        seconds = "N/A"
        microsecond = "N/A"

    return days, hours, minutes, seconds, microsecond


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
            # os.remove(filename)
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

    # os.remove(filename)


@click.command(context_settings=dict(help_option_names=[u'-h', u'--help']))
def main():
    # Open menu with available reports
    top = JarvisMenu()
    mainloop = urwid.MainLoop(top, palette=[('reversed', 'standout', '')])
    top.set_mainloop(mainloop)
    mainloop.run()

if __name__ == "__main__":  # pragma: no cover
        main(prog_name="jarvis")
