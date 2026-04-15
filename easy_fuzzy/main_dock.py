import os
import math
import random
import datetime
from html import escape

import numpy as np
from osgeo import gdal, ogr

from qgis.core import QgsProject, QgsRasterLayer
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QPixmap
from qgis.PyQt.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QDoubleSpinBox,
    QProgressBar,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


gdal.UseExceptions()


def _get_plugin_logo_path():
    return os.path.join(os.path.dirname(__file__), "Icon_GAPEG.png")



class EasyFuzzyDock(QDockWidget):
    def __init__(self, iface):
        super().__init__("Easy Fuzzy QGIS")
        self.iface = iface
        self.setObjectName("EasyFuzzyPluginQGISDock")

        self.rasters = []
        self.training_input_vectors = []
        self.training_layers = []
        self.validation_layers = []
        self.training_merged_path = None
        self.validation_merged_path = None
        self.training_ranking_results = []
        self.last_training_report_html = None
        self.last_overlay_output = None
        self.last_reclass_output = None
        self.last_validation_html = None
        self.last_report_html = None

        self.main_widget = QWidget()
        self.setWidget(self.main_widget)

        self._build_ui()

    def _build_ui(self):
        main_layout = QHBoxLayout()
        self.main_widget.setLayout(main_layout)

        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.StyledPanel)
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        left_panel.setMinimumWidth(230)

        title = QLabel("Easy Fuzzy")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        left_layout.addWidget(title)

        self.menu_list = QListWidget()
        self.menu_list = QListWidget()
        self.menu_list.setMinimumHeight(320)
        self.menu_list.addItem("1. Project")
        self.menu_list.addItem("2. Input rasters")
        self.menu_list.addItem("3. Training & Validation")
        self.menu_list.addItem("4. Training analysis")
        self.menu_list.addItem("5. Fuzzy settings")
        self.menu_list.addItem("6. Fuzzy Membership Generation")
        self.menu_list.addItem("7. Fuzzy Overlay")
        self.menu_list.addItem("8. Results")
        self.menu_list.addItem("9. Sensitivity")
        self.menu_list.addItem("10. Reclassification")
        self.menu_list.addItem("11. Validation")
        self.menu_list.addItem("12. Generate report")
        self.menu_list.addItem("13. About")
        self.menu_list.currentRowChanged.connect(self._change_page)
        left_layout.addWidget(self.menu_list)

        left_layout.addStretch()

        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignCenter)
        logo_path = _get_plugin_logo_path()
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            if not pixmap.isNull():
                logo_label.setPixmap(
                    pixmap.scaledToWidth(180, Qt.SmoothTransformation)
                )
        left_layout.addWidget(logo_label)

        self.pages = QStackedWidget()

        self.page_project = self._build_project_page()
        self.page_inputs = self._build_inputs_page()
        self.page_split = self._build_split_page()
        self.page_training_analysis = self._build_training_analysis_page()
        self.page_membership = self._build_membership_page()
        self.page_fuzzy_membership = self._build_fuzzy_membership_run_page()
        self.page_overlay = self._build_overlay_page()
        self.page_results = self._build_results_page()
        self.page_sensitivity = self._build_sensitivity_page()
        self.page_reclass = self._build_reclass_page()
        self.page_validation = self._build_validation_page()
        self.page_report = self._build_report_page()
        self.page_about = self._build_about_page()

        self.pages.addWidget(self.page_project)
        self.pages.addWidget(self.page_inputs)
        self.pages.addWidget(self.page_split)
        self.pages.addWidget(self.page_training_analysis)
        self.pages.addWidget(self.page_membership)
        self.pages.addWidget(self.page_fuzzy_membership)
        self.pages.addWidget(self.page_overlay)
        self.pages.addWidget(self.page_results)
        self.pages.addWidget(self.page_sensitivity)
        self.pages.addWidget(self.page_reclass)
        self.pages.addWidget(self.page_validation)
        self.pages.addWidget(self.page_report)
        self.pages.addWidget(self.page_about)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.pages, stretch=1)

        self.menu_list.setCurrentRow(0)

    def _change_page(self, index):
        self.pages.setCurrentIndex(index)

    def _run_workflow(self):
        self._run_membership_generation()
        if self.last_overlay_output is None:
            self._run_overlay_only()

    def _run_membership_generation(self):
        if not self.rasters:
            QMessageBox.warning(self, "Warning", "No raster has been added to the workflow.")
            return

        not_configured = [r["name"] for r in self.rasters if not r.get("membership")]
        if not_configured:
            QMessageBox.warning(
                self,
                "Warning",
                "The following rasters have not yet been configured:\n\n" + "\n".join(not_configured)
            )
            return

        intermediate_dir = self.intermediate_edit.text().strip()
        if not intermediate_dir:
            QMessageBox.warning(self, "Warning", "Set the fuzzy raster folder in the Project tab.")
            return

        os.makedirs(intermediate_dir, exist_ok=True)

        try:
            total_rasters = max(1, len(self.rasters))
            self.membership_progress.setValue(0)
            self.membership_progress_label.setText("Preparing fuzzy memberships...")
            QApplication.processEvents()

            log_lines = []
            log_lines.append("FUZZY MEMBERSHIP EXECUTION")
            log_lines.append("")

            for idx, raster in enumerate(self.rasters, start=1):
                progress = int(5 + (idx - 1) * 85 / total_rasters)
                self.membership_progress.setValue(progress)
                self.membership_progress_label.setText(f"Fuzzification {idx}/{total_rasters}: {raster['name']}")
                QApplication.processEvents()

                out_path = os.path.join(intermediate_dir, f"{raster['name']}_fuzzy.tif")
                self._fuzzify_raster(
                    raster_path=raster["path"],
                    method=raster["membership"],
                    params=raster["params"],
                    output_path=out_path,
                )
                raster["fuzzy_path"] = out_path
                raster["status"] = "Fuzzified"
                log_lines.append(f"Fuzzy raster generated: {out_path}")

            self.last_overlay_output = None
            self._refresh_raster_table()
            self.membership_run_text.setPlainText("\n".join(log_lines))
            self.membership_progress.setValue(100)
            self.membership_progress_label.setText("Fuzzy memberships completed.")
            QApplication.processEvents()
            self.menu_list.setCurrentRow(5)

            QMessageBox.information(
                self,
                "Completed",
                "Fuzzy memberships executed successfully.\n\n"
                "Proceed to the Overlay step"
            )

        except Exception as e:
            self.membership_progress.setValue(0)
            self.membership_progress_label.setText("Membership execution stopped due to an error.")
            QMessageBox.critical(self, "Error", f"An error occurred during membership execution:\n\n{str(e)}")

    def _run_overlay_only(self):
        if not self.rasters:
            QMessageBox.warning(self, "Warning", "No raster has been added to the workflow.")
            return

        intermediate_dir = self.intermediate_edit.text().strip()
        final_dir = self.final_output_edit.text().strip()
        project_name = self.project_name_edit.text().strip() or "fuzzy_project"

        if not intermediate_dir:
            QMessageBox.warning(self, "Warning", "Set the fuzzy raster folder in the Project tab.")
            return

        if not final_dir:
            QMessageBox.warning(self, "Warning", "Set the final output folder in the Project tab.")
            return

        os.makedirs(intermediate_dir, exist_ok=True)
        os.makedirs(final_dir, exist_ok=True)

        fuzzy_paths = []
        missing = []
        for raster in self.rasters:
            fuzzy_path = raster.get("fuzzy_path")
            if not fuzzy_path:
                candidate = os.path.join(intermediate_dir, f"{raster['name']}_fuzzy.tif")
                if os.path.exists(candidate):
                    raster["fuzzy_path"] = candidate
                    fuzzy_path = candidate

            if not fuzzy_path or not os.path.exists(fuzzy_path):
                missing.append(self._get_raster_label(raster))
            else:
                fuzzy_paths.append(fuzzy_path)

        if missing:
            QMessageBox.warning(
                self,
                "Warning",
                "Run the Fuzzy membership module first. Missing fuzzy rasters for:\n\n" + "\n".join(missing)
            )
            return

        try:
            total_rasters = max(1, len(fuzzy_paths))
            self.workflow_progress.setValue(0)
            self.workflow_progress_label.setText("Preparing overlay...")
            QApplication.processEvents()

            reference_raster = self.reference_raster_edit.text().strip()
            if not reference_raster:
                reference_raster = fuzzy_paths[0]

            aligned_paths = []
            for i, path in enumerate(fuzzy_paths, start=1):
                progress = int(5 + (i - 1) * 55 / total_rasters)
                self.workflow_progress.setValue(progress)
                self.workflow_progress_label.setText(f"Aligning raster {i}/{total_rasters}")
                QApplication.processEvents()

                aligned_out = os.path.join(intermediate_dir, f"aligned_{i}_{os.path.basename(path)}")
                self._align_raster(
                    input_path=path,
                    reference_path=reference_raster,
                    output_path=aligned_out,
                    resampling_name=self.resampling_combo.currentText(),
                )
                aligned_paths.append(aligned_out)

            overlay_out = os.path.join(final_dir, f"{project_name}_fuzzy_overlay.tif")
            self.workflow_progress.setValue(70)
            self.workflow_progress_label.setText("Running final overlay...")
            QApplication.processEvents()
            self._overlay_rasters(
                raster_paths=aligned_paths,
                method=self.overlay_method.currentText(),
                gamma=self.gamma_spin.value(),
                output_path=overlay_out,
            )

            self.workflow_progress.setValue(90)
            self.workflow_progress_label.setText("Finishing and loading result...")
            QApplication.processEvents()

            self.last_overlay_output = overlay_out
            result_lines = []
            result_lines.append("FUZZY OVERLAY EXECUTION")
            result_lines.append("")
            result_lines.append(f"Overlay method: {self.overlay_method.currentText()}")
            if self.overlay_method.currentText() == "GAMMA":
                result_lines.append(f"Gamma: {self.gamma_spin.value()}")
            result_lines.append(f"Final raster: {overlay_out}")
            self.results_text.setPlainText("\n".join(result_lines))

            self.iface.addRasterLayer(overlay_out, os.path.basename(overlay_out))
            for raster in self.rasters:
                raster["status"] = "Completed"

            self._refresh_raster_table()
            self._show_workflow_summary()
            self.workflow_progress.setValue(100)
            self.workflow_progress_label.setText("Overlay execution completed.")
            QApplication.processEvents()
            self.menu_list.setCurrentRow(7)

            QMessageBox.information(
                self,
                "Completed",
                "Fuzzy overlay executed successfully.\n\n"
                f"Final raster saved in:\n{overlay_out}"
            )

        except Exception as e:
            self.last_overlay_output = None
            self.workflow_progress.setValue(0)
            self.workflow_progress_label.setText("Execution stopped due to an error.")
            QMessageBox.critical(self, "Error", f"An error occurred during execution:\n\n{str(e)}")

    def _build_project_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("Project / Working folder")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        form_box = QGroupBox("Project settings")
        form_layout = QFormLayout(form_box)

        self.project_name_edit = QLineEdit()
        self.workspace_edit = QLineEdit()
        self.intermediate_edit = QLineEdit()
        self.final_output_edit = QLineEdit()

        workspace_btn = QPushButton("Select folder")
        workspace_btn.clicked.connect(lambda: self._select_folder(self.workspace_edit))

        intermediate_btn = QPushButton("Select folder")
        intermediate_btn.clicked.connect(lambda: self._select_folder(self.intermediate_edit))

        final_btn = QPushButton("Select folder")
        final_btn.clicked.connect(lambda: self._select_folder(self.final_output_edit))

        workspace_layout = QHBoxLayout()
        workspace_layout.addWidget(self.workspace_edit)
        workspace_layout.addWidget(workspace_btn)

        intermediate_layout = QHBoxLayout()
        intermediate_layout.addWidget(self.intermediate_edit)
        intermediate_layout.addWidget(intermediate_btn)

        final_layout = QHBoxLayout()
        final_layout.addWidget(self.final_output_edit)
        final_layout.addWidget(final_btn)

        form_layout.addRow("Project name:", self.project_name_edit)
        form_layout.addRow("Working folder:", self._wrap_layout(workspace_layout))
        form_layout.addRow("Fuzzy raster folder:", self._wrap_layout(intermediate_layout))
        form_layout.addRow("Final output folder:", self._wrap_layout(final_layout))

        layout.addWidget(form_box)

        info = QTextEdit()
        info.setReadOnly(True)
        info.setPlainText(
            "In this step you define the project name and the folders that will be used.\n\n"
            "Sugestão:\n"
            "- Working folder: general project folder\n"
            "- Fuzzy raster folder: where the intermediate rasters will be saved\n"
            "- Final output folder: where the final overlay raster will be saved"
        )
        layout.addWidget(info)

        layout.addStretch()
        return page

    def _build_inputs_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("Input rasters")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        self.raster_table = QTableWidget()
        self.raster_table.setColumnCount(5)
        self.raster_table.setHorizontalHeaderLabels(["Name", "Variable name", "Path", "Fuzzy function", "Status"])
        self.raster_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.raster_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.raster_table)

        btn_layout = QHBoxLayout()

        add_btn = QPushButton("Add rasters from disk")
        add_btn.clicked.connect(self._add_rasters_from_disk)

        project_btn = QPushButton("Load rasters from project")
        project_btn.clicked.connect(self._add_rasters_from_project)

        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(self._remove_selected_raster)

        clear_btn = QPushButton("Clear list")
        clear_btn.clicked.connect(self._clear_rasters)

        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(project_btn)
        btn_layout.addWidget(remove_btn)
        btn_layout.addWidget(clear_btn)
        btn_layout.addStretch()

        layout.addLayout(btn_layout)

        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setPlainText(
            "Add here all the rasters that will be included in the fuzzy stream.\n\n"
            "You can:\n"
            "- Load rasters from disk\n"
            "- Import rasters already loaded in the QGIS project\n\n"
            "In the next step, you will define the membership function of each raster."
        )
        layout.addWidget(help_text)

        return page

    def _build_split_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("Training / Validation split")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        self.training_vector_table = QTableWidget()
        self.training_vector_table.setColumnCount(2)
        self.training_vector_table.setHorizontalHeaderLabels(["Layer name", "Path"])
        self.training_vector_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.training_vector_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.training_vector_table)

        controls = QHBoxLayout()
        add_btn = QPushButton("Add vectors from disk")
        add_btn.clicked.connect(self._add_training_vectors_from_disk)
        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(self._remove_selected_training_vector)
        clear_btn = QPushButton("Clear list")
        clear_btn.clicked.connect(self._clear_training_vectors)
        controls.addWidget(add_btn)
        controls.addWidget(remove_btn)
        controls.addWidget(clear_btn)
        controls.addStretch()
        layout.addLayout(controls)

        settings_box = QGroupBox("Split settings")
        settings_form = QFormLayout(settings_box)

        mode_widget = QWidget()
        mode_layout = QHBoxLayout(mode_widget)
        mode_layout.setContentsMargins(0, 0, 0, 0)

        self.split_mode_multi_radio = QRadioButton("Multiple layers")
        self.split_mode_single_radio = QRadioButton("Single layer with many features")
        self.split_mode_multi_radio.setChecked(True)
        self.split_mode_multi_radio.toggled.connect(self._update_split_mode_ui)
        self.split_mode_single_radio.toggled.connect(self._update_split_mode_ui)

        mode_layout.addWidget(self.split_mode_multi_radio)
        mode_layout.addWidget(self.split_mode_single_radio)
        mode_layout.addStretch()

        self.split_seed_spin = QSpinBox()
        self.split_seed_spin.setRange(0, 999999999)
        self.split_seed_spin.setValue(42)

        self.split_train_percent_spin = QSpinBox()
        self.split_train_percent_spin.setRange(1, 99)
        self.split_train_percent_spin.setValue(80)
        self.split_train_percent_spin.setSuffix(" %")

        settings_form.addRow("Input mode:", mode_widget)
        settings_form.addRow("Random seed:", self.split_seed_spin)
        settings_form.addRow("Training proportion:", self.split_train_percent_spin)
        layout.addWidget(settings_box)

        run_box = QGroupBox("Execution")
        run_layout = QVBoxLayout(run_box)
        self.split_progress = QProgressBar()
        self.split_progress.setRange(0, 100)
        self.split_progress.setValue(0)
        run_layout.addWidget(self.split_progress)
        split_btn = QPushButton("Run split")
        split_btn.clicked.connect(self._run_train_validation_split)
        run_layout.addWidget(split_btn)
        layout.addWidget(run_box)

        self.split_text = QTextEdit()
        self.split_text.setReadOnly(True)
        self.split_text.setPlainText(
            "Add the vector sample layers here. The plugin will randomly select whole layers for training and validation using the selected proportion."
        )
        layout.addWidget(self.split_text)
        self._update_split_mode_ui()
        return page

    def _build_training_analysis_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("Training analysis")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        info = QTextEdit()
        info.setReadOnly(True)
        info.setPlainText(
            "This module merges the training layers, reclassifies each input raster into five equal-interval classes, computes class representativeness inside the merged training samples, ranks the classes from the most to the least representative, and remaps them so that the most representative class becomes class 1 and the least representative becomes class 5."
        )
        layout.addWidget(info)

        box = QGroupBox("Outputs")
        form = QFormLayout(box)

        self.training_report_html_edit = QLineEdit()
        self.training_report_html_edit.setPlaceholderText("Leave blank to save automatically in the final output folder")
        report_btn = QPushButton("Save HTML")
        report_btn.clicked.connect(lambda: self._select_save_file(self.training_report_html_edit, "Save training analysis report", "HTML (*.html)"))
        report_layout = QHBoxLayout()
        report_layout.addWidget(self.training_report_html_edit)
        report_layout.addWidget(report_btn)

        form.addRow("Training report:", self._wrap_layout(report_layout))
        layout.addWidget(box)

        run_box = QGroupBox("Execution")
        run_layout = QVBoxLayout(run_box)
        self.training_progress = QProgressBar()
        self.training_progress.setRange(0, 100)
        self.training_progress.setValue(0)
        run_layout.addWidget(self.training_progress)
        run_btn = QPushButton("Run training analysis")
        run_btn.clicked.connect(self._run_training_analysis)
        run_layout.addWidget(run_btn)
        layout.addWidget(run_box)

        self.training_text = QTextEdit()
        self.training_text.setReadOnly(True)
        self.training_text.setPlainText("The training ranking report will appear here.")
        layout.addWidget(self.training_text)
        return page

    def _build_membership_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("Fuzzy settings")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        source_box = QGroupBox("Raster selection")
        source_layout = QVBoxLayout(source_box)

        source_info = QLabel(
            "Select which rasters will be used in the fuzzy membership stage. "
            "You may add original rasters, training-ranked rasters, or a combination of both."
        )
        source_info.setWordWrap(True)
        source_layout.addWidget(source_info)

        source_btn_layout_1 = QHBoxLayout()
        source_add_disk_btn = QPushButton("Add rasters from disk")
        source_add_disk_btn.clicked.connect(self._membership_add_rasters_from_disk)
        source_add_project_btn = QPushButton("Load rasters from project")
        source_add_project_btn.clicked.connect(self._membership_add_rasters_from_project)
        source_btn_layout_1.addWidget(source_add_disk_btn)
        source_btn_layout_1.addWidget(source_add_project_btn)
        source_layout.addLayout(source_btn_layout_1)

        source_btn_layout_2 = QHBoxLayout()
        source_replace_disk_btn = QPushButton("Replace with rasters from disk")
        source_replace_disk_btn.clicked.connect(self._membership_replace_rasters_from_disk)
        source_replace_project_btn = QPushButton("Replace with rasters from project")
        source_replace_project_btn.clicked.connect(self._membership_replace_rasters_from_project)
        source_btn_layout_2.addWidget(source_replace_disk_btn)
        source_btn_layout_2.addWidget(source_replace_project_btn)
        source_layout.addLayout(source_btn_layout_2)

        source_btn_layout_3 = QHBoxLayout()
        source_clear_btn = QPushButton("Clear current list")
        source_clear_btn.clicked.connect(self._clear_rasters)
        source_btn_layout_3.addWidget(source_clear_btn)
        source_btn_layout_3.addStretch()
        source_layout.addLayout(source_btn_layout_3)

        layout.addWidget(source_box)

        top_box = QGroupBox("Select raster and function")
        top_layout = QFormLayout(top_box)

        self.raster_selector_combo = QComboBox()
        self.raster_selector_combo.currentIndexChanged.connect(self._load_selected_raster_config)

        self.membership_combo = QComboBox()
        self.membership_combo.addItems([
            "Linear",
            "Large membership",
            "Small membership",
            "Gaussian",
            "Power membership",
        ])
        self.membership_combo.currentIndexChanged.connect(self._update_parameter_fields)

        self.variable_name_edit = QLineEdit()
        self.variable_name_edit.setPlaceholderText("Optional custom name for this variable")

        top_layout.addRow("Raster:", self.raster_selector_combo)
        top_layout.addRow("Fuzzy function:", self.membership_combo)
        top_layout.addRow("Variable name (optional):", self.variable_name_edit)

        layout.addWidget(top_box)

        self.parameter_box = QGroupBox("Parameters")
        self.parameter_form = QFormLayout(self.parameter_box)

        self.param_1 = QDoubleSpinBox()
        self.param_1.setDecimals(6)
        self.param_1.setRange(-999999999, 999999999)

        self.param_2 = QDoubleSpinBox()
        self.param_2.setDecimals(6)
        self.param_2.setRange(-999999999, 999999999)

        self.param_3 = QDoubleSpinBox()
        self.param_3.setDecimals(6)
        self.param_3.setRange(-999999999, 999999999)
        self.param_3.setValue(2.0)

        self.param_1_label = QLabel("Parameter 1")
        self.param_2_label = QLabel("Parameter 2")
        self.param_3_label = QLabel("Parameter 3")

        self.parameter_form.addRow(self.param_1_label, self.param_1)
        self.parameter_form.addRow(self.param_2_label, self.param_2)
        self.parameter_form.addRow(self.param_3_label, self.param_3)

        layout.addWidget(self.parameter_box)

        button_layout = QHBoxLayout()

        save_btn = QPushButton("Save settings for this raster")
        save_btn.clicked.connect(self._save_membership_config)

        next_btn = QPushButton("Next raster")
        next_btn.clicked.connect(self._go_to_next_raster)

        button_layout.addWidget(save_btn)
        button_layout.addWidget(next_btn)
        button_layout.addStretch()

        layout.addLayout(button_layout)

        self.membership_summary = QTextEdit()
        self.membership_summary.setReadOnly(True)
        self.membership_summary.setPlainText(
            "Choose a raster from the dropdown above and configure its fuzzy function."
        )
        layout.addWidget(self.membership_summary)

        self._update_parameter_fields()
        return page

    def _build_fuzzy_membership_run_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("Fuzzy membership")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        info_box = QGroupBox("Execution")
        info_layout = QVBoxLayout(info_box)

        info = QLabel(
            "This step generates the fuzzy membership rasters from the input rasters and the settings defined in the previous tab.\n"
            "Run this step first. After that, go to the Overlay tab to align the fuzzy rasters and calculate the final overlay."
        )
        info.setWordWrap(True)
        info_layout.addWidget(info)

        self.membership_progress_label = QLabel("Waiting for fuzzy membership execution.")
        self.membership_progress = QProgressBar()
        self.membership_progress.setRange(0, 100)
        self.membership_progress.setValue(0)
        info_layout.addWidget(self.membership_progress_label)
        info_layout.addWidget(self.membership_progress)

        btn_layout = QHBoxLayout()
        run_btn = QPushButton("Run fuzzy memberships")
        run_btn.clicked.connect(self._run_membership_generation)
        btn_layout.addWidget(run_btn)
        btn_layout.addStretch()
        info_layout.addLayout(btn_layout)

        layout.addWidget(info_box)

        self.membership_run_text = QTextEdit()
        self.membership_run_text.setReadOnly(True)
        self.membership_run_text.setPlainText(
            "The membership execution report will appear here."
        )
        layout.addWidget(self.membership_run_text)

        return page

    def _build_overlay_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("Fuzzy overlay")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        box = QGroupBox("Overlay settings")
        form = QFormLayout(box)

        self.overlay_method = QComboBox()
        self.overlay_method.addItems(["AND", "OR", "PRODUCT", "SUM", "GAMMA"])
        self.overlay_method.currentIndexChanged.connect(self._toggle_gamma)

        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setDecimals(4)
        self.gamma_spin.setRange(0.0, 1.0)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.setValue(0.9)

        self.reference_raster_edit = QLineEdit()
        ref_btn = QPushButton("Select raster")
        ref_btn.clicked.connect(
            lambda: self._select_file(
                self.reference_raster_edit,
                "Raster (*.tif *.img *.vrt)"
            )
        )

        ref_layout = QHBoxLayout()
        ref_layout.addWidget(self.reference_raster_edit)
        ref_layout.addWidget(ref_btn)

        self.resampling_combo = QComboBox()
        self.resampling_combo.addItems([
            "Nearest neighbour",
            "Bilinear",
            "Cubic",
            "Cubic spline",
            "Lanczos",
            "Average",
            "Mode",
        ])

        form.addRow("Overlay method:", self.overlay_method)
        form.addRow("Gamma:", self.gamma_spin)
        form.addRow("Reference raster:", self._wrap_layout(ref_layout))
        form.addRow("Resampling:", self.resampling_combo)

        layout.addWidget(box)

        run_box = QGroupBox("Execution")
        run_layout = QVBoxLayout(run_box)

        self.run_summary = QTextEdit()
        self.run_summary.setReadOnly(True)
        self.run_summary.setPlainText(
            "In this step, the plugin aligns the fuzzy rasters and runs the final overlay.\n"
            "Use the Fuzzy membership tab first to generate the intermediate fuzzy rasters."
        )

        self.workflow_progress_label = QLabel("Waiting for execution.")
        self.workflow_progress = QProgressBar()
        self.workflow_progress.setRange(0, 100)
        self.workflow_progress.setValue(0)

        button_layout = QHBoxLayout()

        preview_btn = QPushButton("Show workflow summary")
        preview_btn.clicked.connect(self._show_workflow_summary)

        run_btn = QPushButton("Run overlay")
        run_btn.clicked.connect(self._run_overlay_only)

        button_layout.addWidget(preview_btn)
        button_layout.addWidget(run_btn)
        button_layout.addStretch()

        run_layout.addWidget(self.run_summary)
        run_layout.addWidget(self.workflow_progress_label)
        run_layout.addWidget(self.workflow_progress)
        run_layout.addLayout(button_layout)

        layout.addWidget(run_box)
        layout.addStretch()

        self._toggle_gamma()
        return page

    def _build_results_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("Results")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setPlainText(
            "Workflow results will appear here."
        )
        layout.addWidget(self.results_text)

        return page

    def _build_sensitivity_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("Sensitivity analysis")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        config_box = QGroupBox("Settings")
        config_form = QFormLayout(config_box)

        self.sens_sample_spin = QSpinBox()
        self.sens_sample_spin.setRange(100, 1000000)
        self.sens_sample_spin.setSingleStep(1000)
        self.sens_sample_spin.setValue(10000)

        self.sens_threshold_spin = QDoubleSpinBox()
        self.sens_threshold_spin.setDecimals(4)
        self.sens_threshold_spin.setRange(0.0, 1.0)
        self.sens_threshold_spin.setSingleStep(0.01)
        self.sens_threshold_spin.setValue(0.05)

        config_form.addRow("Number of sampled pixels:", self.sens_sample_spin)
        config_form.addRow("Change threshold:", self.sens_threshold_spin)

        layout.addWidget(config_box)

        run_box = QGroupBox("Execution")
        run_layout = QVBoxLayout(run_box)

        info = QLabel(
            "This module runs a leave-one-out analysis.\n"
            "For each fuzzified variable, the overlay is recalculated without it and compared to the original result."
        )
        info.setWordWrap(True)
        run_layout.addWidget(info)

        self.sensitivity_progress_label = QLabel("Waiting for sensitivity analysis.")
        self.sensitivity_progress = QProgressBar()
        self.sensitivity_progress.setRange(0, 100)
        self.sensitivity_progress.setValue(0)
        run_layout.addWidget(self.sensitivity_progress_label)
        run_layout.addWidget(self.sensitivity_progress)

        btn_layout = QHBoxLayout()
        sens_btn = QPushButton("Run sensitivity analysis")
        sens_btn.clicked.connect(self._run_sensitivity_analysis)
        btn_layout.addWidget(sens_btn)
        btn_layout.addStretch()
        run_layout.addLayout(btn_layout)

        layout.addWidget(run_box)

        self.sensitivity_text = QTextEdit()
        self.sensitivity_text.setReadOnly(True)
        self.sensitivity_text.setPlainText(
            "The sensitivity report will appear here.\n\n"
            "Outputs:\n"
            "- absolute average difference\n"
            "- RMSE\n"
            "- correlation with the original model\n"
            "- % of pixels changed\n"
            "- automatic classification"
        )
        layout.addWidget(self.sensitivity_text)

        return page

    def _build_reclass_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("Reclassification")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        box = QGroupBox("Settings")
        form = QFormLayout(box)

        self.reclass_input_edit = QLineEdit()
        self.reclass_input_edit.setPlaceholderText("Leave blank to use the most recent final gamma raster")
        in_btn = QPushButton("Select raster")
        in_btn.clicked.connect(lambda: self._select_file(self.reclass_input_edit, "Raster (*.tif *.img *.vrt)"))
        in_layout = QHBoxLayout()
        in_layout.addWidget(self.reclass_input_edit)
        in_layout.addWidget(in_btn)

        self.reclass_method_combo = QComboBox()
        self.reclass_method_combo.addItems(["Equal intervals", "Jenks"])

        self.reclass_classes_spin = QSpinBox()
        self.reclass_classes_spin.setRange(2, 20)
        self.reclass_classes_spin.setValue(5)

        self.reclass_output_edit = QLineEdit()
        out_btn = QPushButton("Save as")
        out_btn.clicked.connect(lambda: self._select_save_file(self.reclass_output_edit, "Save reclassified raster", "GeoTIFF (*.tif)"))
        out_layout = QHBoxLayout()
        out_layout.addWidget(self.reclass_output_edit)
        out_layout.addWidget(out_btn)

        form.addRow("Raster de entrada:", self._wrap_layout(in_layout))
        form.addRow("Method:", self.reclass_method_combo)
        form.addRow("Number of classes:", self.reclass_classes_spin)
        form.addRow("Output raster:", self._wrap_layout(out_layout))
        layout.addWidget(box)

        run_box = QGroupBox("Execution")
        run_layout = QVBoxLayout(run_box)
        self.reclass_progress = QProgressBar()
        self.reclass_progress.setRange(0, 100)
        self.reclass_progress.setValue(0)
        run_layout.addWidget(self.reclass_progress)
        run_btn = QPushButton("Run reclassificação")
        run_btn.clicked.connect(self._run_reclassification)
        run_layout.addWidget(run_btn)
        layout.addWidget(run_box)

        self.reclass_text = QTextEdit()
        self.reclass_text.setReadOnly(True)
        self.reclass_text.setPlainText("The reclassification report will appear here.")
        layout.addWidget(self.reclass_text)
        return page

    def _build_validation_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("Validation")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        box = QGroupBox("Inputs")
        form = QFormLayout(box)

        self.validation_raster_edit = QLineEdit()
        self.validation_raster_edit.setPlaceholderText("Leave blank to use the most recent reclassified raster")
        val_raster_btn = QPushButton("Select raster")
        val_raster_btn.clicked.connect(lambda: self._select_file(self.validation_raster_edit, "Raster (*.tif *.img *.vrt)"))
        val_raster_layout = QHBoxLayout()
        val_raster_layout.addWidget(self.validation_raster_edit)
        val_raster_layout.addWidget(val_raster_btn)

        self.validation_vector_edit = QLineEdit()
        val_vector_btn = QPushButton("Select layer")
        val_vector_btn.clicked.connect(lambda: self._select_file(self.validation_vector_edit, "Vectors (*.shp *.gpkg *.geojson);;All files (*.*)"))
        val_vector_layout = QHBoxLayout()
        val_vector_layout.addWidget(self.validation_vector_edit)
        val_vector_layout.addWidget(val_vector_btn)

        self.validation_html_edit = QLineEdit()
        html_btn = QPushButton("Save HTML")
        html_btn.clicked.connect(lambda: self._select_save_file(self.validation_html_edit, "Save HTML report", "HTML (*.html)"))
        html_layout = QHBoxLayout()
        html_layout.addWidget(self.validation_html_edit)
        html_layout.addWidget(html_btn)

        form.addRow("Reclassified raster:", self._wrap_layout(val_raster_layout))
        form.addRow("Validation input layer:", self._wrap_layout(val_vector_layout))
        form.addRow("HTML report:", self._wrap_layout(html_layout))
        layout.addWidget(box)

        run_box = QGroupBox("Execution")
        run_layout = QVBoxLayout(run_box)
        self.validation_progress = QProgressBar()
        self.validation_progress.setRange(0, 100)
        self.validation_progress.setValue(0)
        run_layout.addWidget(self.validation_progress)
        val_btn = QPushButton("Run Validation")
        val_btn.clicked.connect(self._run_validation)
        run_layout.addWidget(val_btn)
        layout.addWidget(run_box)

        self.validation_text = QTextEdit()
        self.validation_text.setReadOnly(True)
        self.validation_text.setPlainText("The validation summary will appear here and the full report will be saved as HTML.")
        layout.addWidget(self.validation_text)
        return page

    def _build_report_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("Generate report")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        info = QTextEdit()
        info.setReadOnly(True)
        info.setPlainText(
            "This module generates a complete methodological report in HTML, written in scientific English and structured like the Methods section of a thesis, dissertation, or journal article.\n\n"
            "The report uses the project information, raster list, optional variable names, fuzzy functions, parameter values, overlay settings, sensitivity analysis settings, reclassification settings, and validation settings already defined in the workflow."
        )
        layout.addWidget(info)

        box = QGroupBox("Output")
        form = QFormLayout(box)

        self.report_html_edit = QLineEdit()
        self.report_html_edit.setPlaceholderText("Leave blank to save automatically in the final output folder")
        report_btn = QPushButton("Save HTML")
        report_btn.clicked.connect(lambda: self._select_save_file(self.report_html_edit, "Save methodological report", "HTML (*.html)"))
        report_layout = QHBoxLayout()
        report_layout.addWidget(self.report_html_edit)
        report_layout.addWidget(report_btn)
        form.addRow("HTML report:", self._wrap_layout(report_layout))
        layout.addWidget(box)

        run_box = QGroupBox("Execution")
        run_layout = QVBoxLayout(run_box)
        self.report_progress = QProgressBar()
        self.report_progress.setRange(0, 100)
        self.report_progress.setValue(0)
        run_layout.addWidget(self.report_progress)
        generate_btn = QPushButton("Generate report")
        generate_btn.clicked.connect(self._generate_report)
        run_layout.addWidget(generate_btn)
        layout.addWidget(run_box)

        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setPlainText("The methodological report summary will appear here after the HTML file is generated.")
        layout.addWidget(self.report_text)

        return page

    def _build_about_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("About")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignCenter)
        logo_path = _get_plugin_logo_path()
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            if not pixmap.isNull():
                logo_label.setPixmap(
                    pixmap.scaledToWidth(220, Qt.SmoothTransformation)
                )
        layout.addWidget(logo_label)

        about = QTextEdit()
        about.setReadOnly(True)
        about.setHtml(
            """
            <h3>Easy Fuzzy QGIS</h3>
            <p>Plugin under development for a complete fuzzy modeling workflow in QGIS.</p>

            <p><b>Objective:</b><br>
            Integrate, in a single organized interface, raster selection, fuzzy membership
            function settings, final overlay, and sensitivity analysis.</p>

            <p><b>Developed by:</b><br>
            Dr. Antonio Henrique Cordeiro Ramalho; Flávio Hebert da Silva Fonseca; Amaury Caldeira de Lima Gonçalves; Wesley Lopes Pinto; 
            Hana Saiumy Favacho dos Santos; Darliane Miranda da Rocha; Duanne Karine dos Anjos Colares; Yasmim Guedes da Silva; Camila
            Vitória Santos de Aquino; José Maria Franco Santos Junior; Giovanne Figueiredo da Rocha; Antonio Francisco Oliveira dos
            Santos<br>
            Grupo Amazônico de Pesquisas Geoespaciais (GAPEG)</p>

            <p><b>Version:</b> 1.6.0 - Split, Training Analysis, Reclassification and Validation</p>
            """
        )
        layout.addWidget(about)

        return page

    def _wrap_layout(self, layout):
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _select_folder(self, line_edit):
        folder = QFileDialog.getExistingDirectory(self, "Select folder")
        if folder:
            line_edit.setText(folder)

    def _select_file(self, line_edit, filter_text):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select file", "", filter_text)
        if file_path:
            line_edit.setText(file_path)

    def _select_save_file(self, line_edit, title, filter_text):
        file_path, _ = QFileDialog.getSaveFileName(self, title, "", filter_text)
        if file_path:
            line_edit.setText(file_path)

    def _raster_exists(self, path):
        return any(r["path"] == path for r in self.rasters)

    def _add_rasters_from_disk(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select rasters",
            "",
            "Raster files (*.tif *.img *.vrt *.asc);;All files (*.*)"
        )

        if not files:
            return

        added = 0
        for f in files:
            if self._raster_exists(f):
                continue

            name = os.path.splitext(os.path.basename(f))[0]
            self.rasters.append({
                "name": name,
                "path": f,
                "membership": "",
                "params": {},
                "variable_name": "",
                "status": "Pending",
            })
            added += 1

        self._refresh_raster_table()
        self._refresh_raster_dropdown()

        QMessageBox.information(self, "Completed", f"{added} raster(s) added.")

    def _add_rasters_from_project(self):
        layers = QgsProject.instance().mapLayers().values()
        added = 0

        for layer in layers:
            if isinstance(layer, QgsRasterLayer):
                path = layer.source()
                if self._raster_exists(path):
                    continue

                self.rasters.append({
                    "name": layer.name(),
                    "path": path,
                    "membership": "",
                    "params": {},
                    "variable_name": "",
                    "status": "Pending",
                })
                added += 1

        self._refresh_raster_table()
        self._refresh_raster_dropdown()

        QMessageBox.information(self, "Completed", f"{added} project raster(s) added.")

    def _remove_selected_raster(self):
        row = self.raster_table.currentRow()
        if row < 0:
            return

        del self.rasters[row]
        self._refresh_raster_table()
        self._refresh_raster_dropdown()
        self._clear_membership_form()

    def _clear_rasters(self):
        reply = QMessageBox.question(
            self,
            "Clear list",
            "Do you want to remove all rasters from the list?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.rasters = []
            self._refresh_raster_table()
            self._refresh_raster_dropdown()
            self._clear_membership_form()

    def _membership_add_rasters_from_disk(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select rasters",
            "",
            "Raster files (*.tif *.img *.vrt *.asc);;All files (*.*)"
        )

        if not files:
            return

        added = 0
        for f in files:
            if self._raster_exists(f):
                continue

            name = os.path.splitext(os.path.basename(f))[0]
            self.rasters.append({
                "name": name,
                "path": f,
                "membership": "",
                "params": {},
                "variable_name": "",
                "status": "Pending",
            })
            added += 1

        self._refresh_raster_table()
        self._refresh_raster_dropdown()

        QMessageBox.information(self, "Completed", f"{added} raster(s) added to Fuzzy settings.")

    def _membership_add_rasters_from_project(self):
        layers = QgsProject.instance().mapLayers().values()
        added = 0

        for layer in layers:
            if isinstance(layer, QgsRasterLayer):
                path = layer.source()
                if self._raster_exists(path):
                    continue

                self.rasters.append({
                    "name": layer.name(),
                    "path": path,
                    "membership": "",
                    "params": {},
                    "variable_name": "",
                    "status": "Pending",
                })
                added += 1

        self._refresh_raster_table()
        self._refresh_raster_dropdown()

        QMessageBox.information(self, "Completed", f"{added} project raster(s) added to Fuzzy settings.")

    def _membership_replace_rasters_from_disk(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select rasters",
            "",
            "Raster files (*.tif *.img *.vrt *.asc);;All files (*.*)"
        )

        if not files:
            return

        new_rasters = []
        for f in files:
            name = os.path.splitext(os.path.basename(f))[0]
            new_rasters.append({
                "name": name,
                "path": f,
                "membership": "",
                "params": {},
                "variable_name": "",
                "status": "Pending",
            })

        self.rasters = new_rasters
        self._refresh_raster_table()
        self._refresh_raster_dropdown()
        QMessageBox.information(self, "Completed", f"{len(new_rasters)} raster(s) loaded into Fuzzy settings.")

    def _membership_replace_rasters_from_project(self):
        layers = QgsProject.instance().mapLayers().values()
        new_rasters = []

        for layer in layers:
            if isinstance(layer, QgsRasterLayer):
                new_rasters.append({
                    "name": layer.name(),
                    "path": layer.source(),
                    "membership": "",
                    "params": {},
                    "variable_name": "",
                    "status": "Pending",
                })

        self.rasters = new_rasters
        self._refresh_raster_table()
        self._refresh_raster_dropdown()
        QMessageBox.information(self, "Completed", f"{len(new_rasters)} project raster(s) loaded into Fuzzy settings.")

    def _vector_exists(self, path):
        return any(v["path"] == path for v in self.training_input_vectors)

    def _add_training_vectors_from_disk(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select vectors",
            "",
            "Vectors (*.shp *.gpkg *.geojson *.json);;All files (*.*)"
        )

        if not files:
            return

        added = 0
        for f in files:
            if self._vector_exists(f):
                continue
            name = os.path.splitext(os.path.basename(f))[0]
            self.training_input_vectors.append({"name": name, "path": f})
            added += 1

        self._refresh_training_vector_table()
        QMessageBox.information(self, "Completed", f"{added} vector layer(s) added.")

    def _remove_selected_training_vector(self):
        row = self.training_vector_table.currentRow()
        if row < 0:
            return
        del self.training_input_vectors[row]
        self._refresh_training_vector_table()

    def _clear_training_vectors(self):
        reply = QMessageBox.question(
            self,
            "Clear list",
            "Do you want to remove all vector layers from the split list?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.training_input_vectors = []
            self.training_layers = []
            self.validation_layers = []
            self.training_merged_path = None
            self.validation_merged_path = None
            self.training_ranking_results = []
            self._refresh_training_vector_table()
            self._update_split_mode_ui()

    def _refresh_training_vector_table(self):
        self.training_vector_table.setRowCount(len(self.training_input_vectors))
        for row, item in enumerate(self.training_input_vectors):
            self.training_vector_table.setItem(row, 0, QTableWidgetItem(item["name"]))
            self.training_vector_table.setItem(row, 1, QTableWidgetItem(item["path"]))
        self.training_vector_table.resizeColumnsToContents()

    def _update_split_mode_ui(self):
        if not hasattr(self, "split_text"):
            return

        if self.split_mode_single_radio.isChecked():
            self.split_text.setPlainText(
                "Add a single vector layer containing multiple features. The plugin will randomly split the features inside that single layer into training and validation outputs using the selected proportion."
            )
        else:
            self.split_text.setPlainText(
                "Add the vector sample layers here. The plugin will randomly select whole layers for training and validation using the selected proportion."
            )

    def _get_split_output_dir(self):
        project_name = self.project_name_edit.text().strip() or "fuzzy_project"
        base_dir = self.workspace_edit.text().strip() or self.final_output_edit.text().strip() or self.intermediate_edit.text().strip()
        if not base_dir:
            base_dir = os.path.join(os.path.expanduser("~"), "EasyFuzzySplit")
        return os.path.join(base_dir, f"{project_name}_split_outputs")

    def _split_single_vector_layer_by_features(self, input_path, training_output_path, validation_output_path, seed, train_pct, layer_name="samples"):
        in_ds = gdal.OpenEx(input_path, gdal.OF_VECTOR)
        if in_ds is None:
            raise Exception(f"It was not possible to open the vector layer:\n{input_path}")

        in_layer = in_ds.GetLayer(0)
        if in_layer is None:
            raise Exception(f"The vector file does not contain a readable layer:\n{input_path}")

        feature_count = in_layer.GetFeatureCount()
        if feature_count < 2:
            raise Exception("The selected vector layer must contain at least two features to perform the split.")

        feature_ids = []
        in_layer.ResetReading()
        for feat in in_layer:
            feature_ids.append(feat.GetFID())

        rng = random.Random(seed)
        rng.shuffle(feature_ids)

        n_total = len(feature_ids)
        n_train = int(round(n_total * train_pct))
        n_train = max(1, min(n_total - 1, n_train))
        train_ids = set(feature_ids[:n_train])
        validation_ids = set(feature_ids[n_train:])

        self._write_split_vector_subset(input_path, training_output_path, train_ids, f"{layer_name}_training")
        self._write_split_vector_subset(input_path, validation_output_path, validation_ids, f"{layer_name}_validation")

        in_ds = None
        return {
            "total_features": n_total,
            "training_features": len(train_ids),
            "validation_features": len(validation_ids),
        }

    def _write_split_vector_subset(self, input_path, output_path, feature_ids, output_layer_name):
        driver_name = "GPKG"
        if os.path.exists(output_path):
            try:
                ogr.GetDriverByName(driver_name).DeleteDataSource(output_path)
            except Exception:
                try:
                    os.remove(output_path)
                except Exception:
                    pass

        in_ds = gdal.OpenEx(input_path, gdal.OF_VECTOR)
        if in_ds is None:
            raise Exception(f"It was not possible to open the vector layer:\n{input_path}")

        in_layer = in_ds.GetLayer(0)
        if in_layer is None:
            raise Exception(f"The vector file does not contain a readable layer:\n{input_path}")

        out_driver = ogr.GetDriverByName(driver_name)
        out_ds = out_driver.CreateDataSource(output_path)
        if out_ds is None:
            raise Exception(f"It was not possible to create the split output:\n{output_path}")

        srs = in_layer.GetSpatialRef()
        geom_type = in_layer.GetGeomType()
        out_layer = out_ds.CreateLayer(output_layer_name, srs=srs, geom_type=geom_type)
        if out_layer is None:
            raise Exception(f"It was not possible to create the output layer:\n{output_layer_name}")

        in_defn = in_layer.GetLayerDefn()
        for i in range(in_defn.GetFieldCount()):
            out_layer.CreateField(in_defn.GetFieldDefn(i))

        out_defn = out_layer.GetLayerDefn()
        in_layer.ResetReading()
        for feat in in_layer:
            if feat.GetFID() not in feature_ids:
                continue

            out_feat = ogr.Feature(out_defn)
            geom = feat.GetGeometryRef()
            if geom is not None:
                out_feat.SetGeometry(geom.Clone())

            for i in range(in_defn.GetFieldCount()):
                try:
                    out_feat.SetField(i, feat.GetField(i))
                except Exception:
                    pass

            if out_layer.CreateFeature(out_feat) != 0:
                raise Exception(f"It was not possible to write one of the split features to:\n{output_path}")
            out_feat = None

        out_ds = None
        in_ds = None

    def _run_train_validation_split(self):
        single_mode = self.split_mode_single_radio.isChecked()

        if single_mode:
            if len(self.training_input_vectors) != 1:
                QMessageBox.warning(self, "Warning", "In single-layer mode, add exactly one vector layer with multiple features.")
                return
        else:
            if len(self.training_input_vectors) < 2:
                QMessageBox.warning(self, "Warning", "Add at least two vector layers to perform the split.")
                return

        try:
            self.split_progress.setValue(5)
            seed = self.split_seed_spin.value()
            train_pct = self.split_train_percent_spin.value() / 100.0

            if single_mode:
                input_item = self.training_input_vectors[0]
                split_dir = self._get_split_output_dir()
                os.makedirs(split_dir, exist_ok=True)

                base_name = self._safe_name(input_item["name"])
                train_path = os.path.join(split_dir, f"{base_name}_training.gpkg")
                validation_path = os.path.join(split_dir, f"{base_name}_validation.gpkg")

                self.split_progress.setValue(30)
                split_info = self._split_single_vector_layer_by_features(
                    input_path=input_item["path"],
                    training_output_path=train_path,
                    validation_output_path=validation_path,
                    seed=seed,
                    train_pct=train_pct,
                    layer_name=base_name,
                )

                self.training_layers = [{"name": f"{input_item['name']}_training", "path": train_path}]
                self.validation_layers = [{"name": f"{input_item['name']}_validation", "path": validation_path}]

                self.split_progress.setValue(100)
                lines = [
                    "TRAINING / VALIDATION SPLIT",
                    "",
                    "Mode: Single layer with many features",
                    f"Random seed: {seed}",
                    f"Training proportion: {train_pct * 100:.1f}%",
                    f"Input layer: {input_item['name']}",
                    f"Total features: {split_info['total_features']}",
                    f"Training features: {split_info['training_features']}",
                    f"Validation features: {split_info['validation_features']}",
                    "",
                    f"Training output: {train_path}",
                    f"Validation output: {validation_path}",
                ]
            else:
                items = list(self.training_input_vectors)
                rng = random.Random(seed)
                rng.shuffle(items)

                n_total = len(items)
                n_train = int(round(n_total * train_pct))
                n_train = max(1, min(n_total - 1, n_train))
                n_validation = n_total - n_train

                self.training_layers = items[:n_train]
                self.validation_layers = items[n_train:]

                self.split_progress.setValue(100)
                lines = [
                    "TRAINING / VALIDATION SPLIT",
                    "",
                    "Mode: Multiple layers",
                    f"Random seed: {seed}",
                    f"Training proportion: {train_pct * 100:.1f}%",
                    f"Total layers: {n_total}",
                    f"Training layers: {n_train}",
                    f"Validation layers: {n_validation}",
                    "",
                    "Training selection:",
                ]
                for item in self.training_layers:
                    lines.append(f" - {item['name']}")
                lines.append("")
                lines.append("Validation selection:")
                for item in self.validation_layers:
                    lines.append(f" - {item['name']}")

            self.split_text.setPlainText("\n".join(lines))
            self.menu_list.setCurrentRow(2)
            QMessageBox.information(self, "Completed", "Split completed successfully.\n\nYou can now run the Training analysis module.")
        except Exception as e:
            self.split_progress.setValue(0)
            QMessageBox.critical(self, "Error", f"An error occurred during the split:\n\n{str(e)}")

    def _run_training_analysis(self):
        if not self.rasters:
            QMessageBox.warning(self, "Warning", "Add the input rasters before running the training analysis.")
            return
        if not self.training_layers or not self.validation_layers:
            QMessageBox.warning(self, "Warning", "Run the Training / Validation split first.")
            return

        base_dir = self.final_output_edit.text().strip() or self.workspace_edit.text().strip() or self.intermediate_edit.text().strip()
        if not base_dir:
            QMessageBox.warning(self, "Warning", "Set the working folder or the final output folder in the Project tab.")
            return

        project_name = self.project_name_edit.text().strip() or "fuzzy_project"
        analysis_dir = os.path.join(base_dir, f"{project_name}_training_analysis")
        os.makedirs(analysis_dir, exist_ok=True)

        html_path = self.training_report_html_edit.text().strip()
        if not html_path:
            html_path = os.path.join(analysis_dir, f"{project_name}_training_analysis.html")
            self.training_report_html_edit.setText(html_path)

        try:
            self.training_progress.setValue(5)
            self.training_merged_path = os.path.join(analysis_dir, f"{project_name}_training.gpkg")
            self.validation_merged_path = os.path.join(analysis_dir, f"{project_name}_validation.gpkg")
            self._merge_vector_layers(self.training_layers, self.training_merged_path)
            self.training_progress.setValue(20)
            self._merge_vector_layers(self.validation_layers, self.validation_merged_path)
            self.validation_vector_edit.setText(self.validation_merged_path)
            self.training_progress.setValue(30)

            results = []
            total = max(1, len(self.rasters))

            for idx, raster in enumerate(self.rasters, start=1):
                progress = int(30 + (idx - 1) * 55 / total)
                self.training_progress.setValue(progress)
                QApplication.processEvents()

                label = self._get_raster_label(raster)
                arr = self._read_raster_array(raster["path"])
                valid = arr[~np.isnan(arr)]
                if valid.size == 0:
                    raise Exception(f"Raster without valid pixels: {label}")

                min_val = float(np.nanmin(valid))
                max_val = float(np.nanmax(valid))
                if min_val == max_val:
                    raise Exception(f"Raster with constant value: {label}")

                breaks = np.linspace(min_val, max_val, 6)
                class_arr = self._apply_breaks(arr, breaks)
                training_mask = self._rasterize_validation(self.training_merged_path, raster["path"])
                class_summary = self._compute_class_distribution(class_arr, training_mask)
                ranked_summary, rank_map = self._rank_class_summary(class_summary)
                remapped_arr = self._remap_classes(class_arr, rank_map)

                reclass_path = os.path.join(analysis_dir, f"{self._safe_name(label)}_training_ranked.tif")
                ds = gdal.Open(raster["path"])
                self._write_int_raster_like(ds, remapped_arr, reclass_path)
                ds = None

                if os.path.exists(reclass_path):
                    self.iface.addRasterLayer(reclass_path, os.path.basename(reclass_path))

                results.append({
                    "raster_name": raster["name"],
                    "label": label,
                    "source_path": raster["path"],
                    "breaks": breaks.tolist(),
                    "class_summary": class_summary,
                    "ranked_summary": ranked_summary,
                    "rank_map": rank_map,
                    "output_path": reclass_path,
                    "training_pixels": int(np.sum(training_mask & (~np.isnan(class_arr)))),
                })

            self.training_ranking_results = results
            self.training_progress.setValue(90)

            html = self._build_training_analysis_html(project_name, html_path, results)
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)
            self.last_training_report_html = html_path

            if self.training_merged_path and os.path.exists(self.training_merged_path):
                self.iface.addVectorLayer(self.training_merged_path, os.path.basename(self.training_merged_path), "ogr")
            if self.validation_merged_path and os.path.exists(self.validation_merged_path):
                self.iface.addVectorLayer(self.validation_merged_path, os.path.basename(self.validation_merged_path), "ogr")

            lines = [
                "TRAINING ANALYSIS",
                "",
                f"Merged training vector: {self.training_merged_path}",
                f"Merged validation vector: {self.validation_merged_path}",
                "",
            ]
            for result in results:
                lines.append(f"Raster: {result['label']}")
                for item in result["ranked_summary"]:
                    lines.append(
                        f" - Rank {item['new_class']}: original class {item['class']} | pixels = {item['count']} | percentage = {item['percentage']:.2f}%"
                    )
                lines.append(f" - Output raster: {result['output_path']}")
                lines.append("")
            lines.append(f"HTML report: {html_path}")

            self.training_text.setPlainText("\n".join(lines))
            self.training_progress.setValue(100)
            self.menu_list.setCurrentRow(3)
            QMessageBox.information(self, "Completed", "Training analysis completed successfully.\n\nThe training rasters were saved to disk in the project output folder and loaded into QGIS.\nThe merged validation layer has already been sent to the Validation tab.")
        except Exception as e:
            self.training_progress.setValue(0)
            QMessageBox.critical(self, "Error", f"An error occurred during the training analysis:\n\n{str(e)}")

    def _merge_vector_layers(self, input_layers, output_path):
        if not input_layers:
            raise Exception("There are no vector layers to merge.")

        driver_name = "GPKG"
        if os.path.exists(output_path):
            try:
                ogr.GetDriverByName(driver_name).DeleteDataSource(output_path)
            except Exception:
                try:
                    os.remove(output_path)
                except Exception:
                    pass

        out_driver = ogr.GetDriverByName(driver_name)
        out_ds = out_driver.CreateDataSource(output_path)
        if out_ds is None:
            raise Exception(f"It was not possible to create the merged vector file:\n{output_path}")

        out_layer = None

        for item in input_layers:
            in_ds = gdal.OpenEx(item["path"], gdal.OF_VECTOR)
            if in_ds is None:
                raise Exception(f"It was not possible to open the vector layer:\n{item['path']}")
            in_layer = in_ds.GetLayer(0)
            if in_layer is None:
                raise Exception(f"The vector file does not contain a readable layer:\n{item['path']}")

            if out_layer is None:
                srs = in_layer.GetSpatialRef()
                geom_type = in_layer.GetGeomType()
                layer_name = os.path.splitext(os.path.basename(output_path))[0]
                out_layer = out_ds.CreateLayer(layer_name, srs=srs, geom_type=geom_type)
                if out_layer is None:
                    raise Exception("It was not possible to create the output vector layer.")

            in_defn = in_layer.GetLayerDefn()
            current_out_defn = out_layer.GetLayerDefn()
            existing_field_names = {
                current_out_defn.GetFieldDefn(i).GetNameRef()
                for i in range(current_out_defn.GetFieldCount())
            }

            for i in range(in_defn.GetFieldCount()):
                field_defn = in_defn.GetFieldDefn(i)
                field_name = field_defn.GetNameRef()
                if field_name not in existing_field_names:
                    out_layer.CreateField(field_defn)
                    existing_field_names.add(field_name)

            if "source_name" not in existing_field_names:
                source_field = ogr.FieldDefn("source_name", ogr.OFTString)
                source_field.SetWidth(254)
                out_layer.CreateField(source_field)
                existing_field_names.add("source_name")

            out_defn = out_layer.GetLayerDefn()
            out_field_names = {
                out_defn.GetFieldDefn(i).GetNameRef()
                for i in range(out_defn.GetFieldCount())
            }

            in_layer.ResetReading()
            for feat in in_layer:
                out_feat = ogr.Feature(out_defn)
                geom = feat.GetGeometryRef()
                if geom is not None:
                    out_feat.SetGeometry(geom.Clone())

                for i in range(in_defn.GetFieldCount()):
                    field_name = in_defn.GetFieldDefn(i).GetNameRef()
                    if field_name in out_field_names:
                        try:
                            out_feat.SetField(field_name, feat.GetField(i))
                        except Exception:
                            pass

                if "source_name" in out_field_names:
                    out_feat.SetField("source_name", str(item["name"]))

                if out_layer.CreateFeature(out_feat) != 0:
                    raise Exception(f"It was not possible to append a feature from layer: {item['name']}")
                out_feat = None
            in_ds = None

        out_ds = None
    def _compute_class_distribution(self, class_arr, training_mask):
        valid = (~np.isnan(class_arr)) & training_mask
        total = int(np.sum(valid))
        if total == 0:
            raise Exception("The merged training layer does not overlap valid raster cells.")

        summary = []
        for cls in range(1, 6):
            count = int(np.sum((class_arr == float(cls)) & valid))
            percentage = (count / total) * 100.0 if total else 0.0
            summary.append({"class": cls, "count": count, "percentage": percentage})
        return summary

    def _rank_class_summary(self, class_summary):
        ranked = sorted(class_summary, key=lambda x: (-x["count"], x["class"]))
        rank_map = {}
        ranked_output = []
        for new_class, item in enumerate(ranked, start=1):
            rank_map[item["class"]] = new_class
            ranked_output.append({
                "new_class": new_class,
                "class": item["class"],
                "count": item["count"],
                "percentage": item["percentage"],
            })
        return ranked_output, rank_map

    def _remap_classes(self, arr, rank_map):
        result = np.full(arr.shape, np.nan, dtype=np.float32)
        valid = ~np.isnan(arr)
        result[valid] = arr[valid]
        for original_class, new_class in rank_map.items():
            result[arr == float(original_class)] = float(new_class)
        return result

    def _build_training_analysis_html(self, project_name, html_path, results):
        now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        sections = [
            f"<h1>Training analysis report</h1><p><strong>Project:</strong> {escape(project_name)}<br><strong>Generated on:</strong> {escape(now)}</p>",
            f"<p><strong>Merged training layer:</strong> {escape(self.training_merged_path or '')}<br><strong>Merged validation layer:</strong> {escape(self.validation_merged_path or '')}</p>",
        ]

        for result in results:
            rows = []
            for item in result["ranked_summary"]:
                rows.append(
                    f"<tr><td>{item['new_class']}</td><td>{item['class']}</td><td>{item['count']}</td><td>{item['percentage']:.2f}%</td></tr>"
                )

            breaks = result["breaks"]
            interval_rows = []
            for i in range(1, len(breaks)):
                interval_rows.append(
                    f"<tr><td>{i}</td><td>{breaks[i-1]:.6f}</td><td>{breaks[i]:.6f}</td></tr>"
                )

            remap_rows = []
            for original_class, new_class in sorted(result["rank_map"].items()):
                remap_rows.append(f"<tr><td>{original_class}</td><td>{new_class}</td></tr>")

            sections.append(f"""
            <h2>{escape(result['label'])}</h2>
            <p><strong>Source raster:</strong> {escape(result['source_path'])}<br>
            <strong>Training pixels used:</strong> {result['training_pixels']}<br>
            <strong>Ranked output raster:</strong> {escape(result['output_path'])}</p>
            <h3>Equal-interval classes</h3>
            <table>
                <thead><tr><th>Original class</th><th>Minimum</th><th>Maximum</th></tr></thead>
                <tbody>{''.join(interval_rows)}</tbody>
            </table>
            <h3>Representativeness ranking</h3>
            <table>
                <thead><tr><th>New rank</th><th>Original class</th><th>Pixels inside training samples</th><th>Percentage</th></tr></thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
            <h3>Class remapping</h3>
            <table>
                <thead><tr><th>Original class</th><th>New class</th></tr></thead>
                <tbody>{''.join(remap_rows)}</tbody>
            </table>
            """)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Training analysis report</title>
<style>body{{font-family:Arial,sans-serif;margin:24px;line-height:1.5}}h1,h2,h3{{color:#1f2937}}table{{border-collapse:collapse;width:100%;margin:12px 0 24px 0}}th,td{{border:1px solid #d1d5db;padding:8px;text-align:left}}th{{background:#f3f4f6}}</style>
</head>
<body>
{''.join(sections)}
</body>
</html>"""

    def _get_raster_label(self, raster):
        custom_name = raster.get("variable_name", "").strip()
        return custom_name or raster["name"]

    def _refresh_raster_table(self):
        self.raster_table.setRowCount(len(self.rasters))

        for row, raster in enumerate(self.rasters):
            self.raster_table.setItem(row, 0, QTableWidgetItem(raster["name"]))
            self.raster_table.setItem(row, 1, QTableWidgetItem(raster.get("variable_name", "")))
            self.raster_table.setItem(row, 2, QTableWidgetItem(raster["path"]))
            self.raster_table.setItem(row, 3, QTableWidgetItem(raster["membership"]))
            self.raster_table.setItem(row, 4, QTableWidgetItem(raster["status"]))

        self.raster_table.resizeColumnsToContents()

    def _refresh_raster_dropdown(self):
        current_text = self.raster_selector_combo.currentText()
        self.raster_selector_combo.blockSignals(True)
        self.raster_selector_combo.clear()

        for raster in self.rasters:
            self.raster_selector_combo.addItem(self._get_raster_label(raster))

        idx = self.raster_selector_combo.findText(current_text)
        if idx >= 0:
            self.raster_selector_combo.setCurrentIndex(idx)
        elif self.rasters:
            self.raster_selector_combo.setCurrentIndex(0)

        self.raster_selector_combo.blockSignals(False)

        if self.rasters:
            self._load_selected_raster_config()
        else:
            self._clear_membership_form()

    def _load_selected_raster_config(self):
        idx = self.raster_selector_combo.currentIndex()
        if idx < 0 or idx >= len(self.rasters):
            self._clear_membership_form()
            return

        raster = self.rasters[idx]

        membership = raster.get("membership", "")
        if membership:
            combo_idx = self.membership_combo.findText(membership)
            if combo_idx >= 0:
                self.membership_combo.setCurrentIndex(combo_idx)

        params = raster.get("params", {})
        self.param_1.setValue(params.get("param_1", 0.0))
        self.param_2.setValue(params.get("param_2", 0.0))
        self.param_3.setValue(params.get("param_3", 2.0))
        self.variable_name_edit.setText(raster.get("variable_name", ""))

        self._update_parameter_fields()
        self._update_membership_summary()

    def _clear_membership_form(self):
        self.param_1.setValue(0.0)
        self.param_2.setValue(0.0)
        self.param_3.setValue(2.0)
        self.variable_name_edit.clear()
        self.membership_summary.setPlainText("Choose a raster from the dropdown above and configure its fuzzy function.")

    def _update_parameter_fields(self):
        method = self.membership_combo.currentText()

        self.param_3.hide()
        self.param_3_label.hide()

        if method == "Linear":
            self.param_1_label.setText("Minimum value")
            self.param_2_label.setText("Maximum value")

        elif method == "Large membership":
            self.param_1_label.setText("Midpoint")
            self.param_2_label.setText("Spread")

        elif method == "Small membership":
            self.param_1_label.setText("Midpoint")
            self.param_2_label.setText("Spread")

        elif method == "Gaussian":
            self.param_1_label.setText("Midpoint")
            self.param_2_label.setText("Spread")

        elif method == "Power membership":
            self.param_1_label.setText("Low bound")
            self.param_2_label.setText("High bound")
            self.param_3_label.setText("Exponent")
            self.param_3.show()
            self.param_3_label.show()

    def _save_membership_config(self):
        idx = self.raster_selector_combo.currentIndex()
        if idx < 0 or idx >= len(self.rasters):
            QMessageBox.warning(self, "Warning", "No raster was selected.")
            return

        method = self.membership_combo.currentText()
        params = {
            "param_1": self.param_1.value(),
            "param_2": self.param_2.value(),
        }

        if method == "Power membership":
            params["param_3"] = self.param_3.value()

        self.rasters[idx]["membership"] = method
        self.rasters[idx]["params"] = params
        self.rasters[idx]["variable_name"] = self.variable_name_edit.text().strip()
        self.rasters[idx]["status"] = "Configured"

        self._refresh_raster_table()
        self._refresh_raster_dropdown()
        self._update_membership_summary()

        QMessageBox.information(self, "Success", f"Settings saved to: {self._get_raster_label(self.rasters[idx])}")

    def _go_to_next_raster(self):
        idx = self.raster_selector_combo.currentIndex()
        if idx < 0:
            return

        if idx < self.raster_selector_combo.count() - 1:
            self.raster_selector_combo.setCurrentIndex(idx + 1)
        else:
            QMessageBox.information(self, "End", "You reached the last raster in the list.")

    def _update_membership_summary(self):
        idx = self.raster_selector_combo.currentIndex()
        if idx < 0 or idx >= len(self.rasters):
            return

        raster = self.rasters[idx]
        text = [
            f"Raster: {raster['name']}",
            f"Variable name: {raster.get('variable_name', '') or 'Not defined'}",
            f"Path: {raster['path']}",
            f"Fuzzy function: {raster.get('membership', '')}",
            "Parameters:",
        ]

        if raster.get("params"):
            for k, v in raster["params"].items():
                text.append(f" - {k}: {v}")
        else:
            text.append(" - No parameters saved yet.")

        self.membership_summary.setPlainText("\n".join(text))

    def _toggle_gamma(self):
        use_gamma = self.overlay_method.currentText() == "GAMMA"
        self.gamma_spin.setEnabled(use_gamma)

    def _show_workflow_summary(self):
        lines = []
        lines.append("FUZZY WORKFLOW SUMMARY")
        lines.append("")

        lines.append(f"Project: {self.project_name_edit.text()}")
        lines.append(f"Working folder: {self.workspace_edit.text()}")
        lines.append(f"Fuzzy folder: {self.intermediate_edit.text()}")
        lines.append(f"Final folder: {self.final_output_edit.text()}")
        lines.append("")

        lines.append("RASTERS:")
        if not self.rasters:
            lines.append(" - No raster added.")
        else:
            for i, r in enumerate(self.rasters, start=1):
                lines.append(f"{i}. {self._get_raster_label(r)}")
                if r.get("variable_name"):
                    lines.append(f"   Original raster name: {r['name']}")
                lines.append(f"   Path: {r['path']}")
                lines.append(f"   Function: {r['membership']}")
                if r.get("params"):
                    for k, v in r["params"].items():
                        lines.append(f"   {k}: {v}")
                lines.append(f"   Status: {r['status']}")
                if r.get("fuzzy_path"):
                    lines.append(f"   Fuzzy raster: {r['fuzzy_path']}")
                lines.append("")

        lines.append("OVERLAY:")
        lines.append(f" - Method: {self.overlay_method.currentText()}")
        if self.overlay_method.currentText() == "GAMMA":
            lines.append(f" - Gamma: {self.gamma_spin.value()}")
        lines.append(f" - Reference raster: {self.reference_raster_edit.text()}")
        lines.append(f" - Resampling: {self.resampling_combo.currentText()}")

        self.run_summary.setPlainText("\n".join(lines))

    def _run_sensitivity_analysis(self):
        if not self.last_overlay_output or not os.path.exists(self.last_overlay_output):
            QMessageBox.warning(
                self,
                "Warning",
                "Run the full fuzzy model before running the sensitivity analysis."
            )
            return

        if len(self.rasters) < 2:
            QMessageBox.warning(
                self,
                "Warning",
                "Sensitivity analysis requires at least two variables."
            )
            return

        fuzzy_paths = []
        missing = []
        for raster in self.rasters:
            fp = raster.get("fuzzy_path")
            if not fp or not os.path.exists(fp):
                missing.append(self._get_raster_label(raster))
            else:
                fuzzy_paths.append(fp)

        if missing:
            QMessageBox.warning(
                self,
                "Warning",
                "The following fuzzy rasters were not found:\n\n" + "\n".join(missing)
            )
            return

        try:
            total_rasters = max(1, len(self.rasters))
            self.sensitivity_progress.setValue(0)
            self.sensitivity_progress_label.setText("Preparing sensitivity analysis...")
            QApplication.processEvents()

            sample_size = self.sens_sample_spin.value()
            threshold = self.sens_threshold_spin.value()

            self.sensitivity_progress.setValue(10)
            self.sensitivity_progress_label.setText("Reading reference raster...")
            QApplication.processEvents()

            y = self._read_raster_array(self.last_overlay_output)
            valid_mask = ~np.isnan(y)
            valid_indices = np.flatnonzero(valid_mask.ravel())

            if valid_indices.size == 0:
                raise Exception("The final raster has no valid pixels for analysis.")

            n_samples = min(sample_size, valid_indices.size)
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(valid_indices, size=n_samples, replace=False)
            y_sample = y.ravel()[sample_idx]

            lines = []
            lines.append("SENSITIVITY ANALYSIS - VERSION 1")
            lines.append("")
            lines.append(f"Reference result: {self.last_overlay_output}")
            lines.append(f"Sampled pixels: {n_samples}")
            lines.append(f"Change threshold: {threshold}")
            lines.append("")

            rows = []
            aligned_original = self._get_aligned_fuzzy_paths()
            self.sensitivity_progress.setValue(20)
            self.sensitivity_progress_label.setText("Running leave-one-out comparisons...")
            QApplication.processEvents()

            for i, raster in enumerate(self.rasters):
                loo_paths = [p for j, p in enumerate(aligned_original) if j != i]

                temp_out = os.path.join(
                    self.final_output_edit.text().strip(),
                    f"sensitivity_without_{self._safe_name(self._get_raster_label(raster))}.tif"
                )

                progress = int(20 + (i * 60 / total_rasters))
                self.sensitivity_progress.setValue(progress)
                self.sensitivity_progress_label.setText(
                    f"Analyzing variable {i + 1}/{total_rasters}: {self._get_raster_label(raster)}"
                )
                QApplication.processEvents()

                self._overlay_rasters(
                    raster_paths=loo_paths,
                    method=self.overlay_method.currentText(),
                    gamma=self.gamma_spin.value(),
                    output_path=temp_out,
                )

                y_loo = self._read_raster_array(temp_out).ravel()[sample_idx]

                diff = np.abs(y_sample - y_loo)
                mean_abs = float(np.nanmean(diff))
                rmse = float(np.sqrt(np.nanmean((y_sample - y_loo) ** 2)))

                valid_pair = (~np.isnan(y_sample)) & (~np.isnan(y_loo))
                if np.sum(valid_pair) > 2:
                    corr = float(np.corrcoef(y_sample[valid_pair], y_loo[valid_pair])[0, 1])
                else:
                    corr = np.nan

                pct_changed = float(np.mean(diff > threshold) * 100.0)

                classification = self._classify_sensitivity(mean_abs, pct_changed)

                rows.append((self._get_raster_label(raster), mean_abs, rmse, corr, pct_changed, classification))

            rows.sort(key=lambda x: x[1], reverse=True)
            self.sensitivity_progress.setValue(85)
            self.sensitivity_progress_label.setText("Building sensitivity report...")
            QApplication.processEvents()

            for name, mean_abs, rmse, corr, pct_changed, classification in rows:
                lines.append(f"{name}")
                lines.append(f" - Absolute average difference: {mean_abs:.6f}")
                lines.append(f" - RMSE: {rmse:.6f}")
                lines.append(f" - Correlation with the original model: {corr:.6f}" if not np.isnan(corr) else " - Correlation with the original model: undefined")
                lines.append(f" - % of pixels changed above the threshold: {pct_changed:.2f}%")
                lines.append(f" - Classification: {classification}")
                lines.append("")

            self.sensitivity_text.setPlainText("\n".join(lines))
            self.menu_list.setCurrentRow(8)
            self.sensitivity_progress.setValue(100)
            self.sensitivity_progress_label.setText("Sensitivity analysis completed.")
            QApplication.processEvents()

            QMessageBox.information(
                self,
                "Completed",
                "Sensitivity analysis concluída.\n\n"
                "The report has been updated in the Sensitivity tab."
            )

        except Exception as e:
            self.sensitivity_progress.setValue(0)
            self.sensitivity_progress_label.setText("Sensitivity analysis stopped due to an error.")
            QMessageBox.critical(self, "Error", f"An error occurred in the sensitivity analysis:\n\n{str(e)}")


    def _classify_sensitivity(self, mean_abs_diff, pct_changed):
        if mean_abs_diff < 0.01 and pct_changed < 5.0:
            return "Irrelevant"
        if mean_abs_diff < 0.05 and pct_changed < 20.0:
            return "Moderate"
        return "Important"

    def _get_aligned_fuzzy_paths(self):
        intermediate_dir = self.intermediate_edit.text().strip()
        aligned_paths = []

        for i, raster in enumerate(self.rasters, start=1):
            path = os.path.join(intermediate_dir, f"aligned_{i}_{os.path.basename(raster['fuzzy_path'])}")
            if os.path.exists(path):
                aligned_paths.append(path)
            else:
                aligned_paths.append(raster["fuzzy_path"])

        return aligned_paths

    def _safe_name(self, text):
        return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in text)

    def _read_raster_array(self, raster_path):
        ds = gdal.Open(raster_path)
        if ds is None:
            raise Exception(f"It was not possible to open the raster:\n{raster_path}")
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray().astype(np.float32)
        nodata = band.GetNoDataValue()
        if nodata is not None:
            if np.isnan(nodata):
                arr[np.isnan(arr)] = np.nan
            else:
                arr[arr == nodata] = np.nan
        ds = None
        return arr

    def _run_reclassification(self):
        raster_path = self.reclass_input_edit.text().strip() or self.last_overlay_output
        if not raster_path or not os.path.exists(raster_path):
            QMessageBox.warning(self, "Warning", "Select a valid input raster or run the overlay first.")
            return

        output_path = self.reclass_output_edit.text().strip()
        if not output_path:
            base_dir = self.final_output_edit.text().strip() or os.path.dirname(raster_path)
            project_name = self.project_name_edit.text().strip() or "fuzzy_project"
            output_path = os.path.join(base_dir, f"{project_name}_reclass.tif")
            self.reclass_output_edit.setText(output_path)

        try:
            self.reclass_progress.setValue(5)
            arr = self._read_raster_array(raster_path)
            valid = arr[~np.isnan(arr)]
            if valid.size == 0:
                raise Exception("The input raster does not contain valid pixels.")

            self.reclass_progress.setValue(30)
            n_classes = self.reclass_classes_spin.value()
            method = self.reclass_method_combo.currentText()

            if method == "Equal intervals":
                breaks = np.linspace(0.0, 1.0, n_classes + 1)
            else:
                sample = valid
                if sample.size > 10000:
                    rng = np.random.default_rng(42)
                    sample = rng.choice(sample, size=10000, replace=False)
                breaks = np.array(self._jenks_breaks(sample.astype(float), n_classes), dtype=float)
                breaks[0] = min(breaks[0], float(np.nanmin(valid)))
                breaks[-1] = max(breaks[-1], float(np.nanmax(valid)))

            self.reclass_progress.setValue(60)
            reclass = self._apply_breaks(arr, breaks)
            ds = gdal.Open(raster_path)
            self._write_int_raster_like(ds, reclass, output_path)
            ds = None
            self.last_reclass_output = output_path
            self.validation_raster_edit.setText(output_path)

            lines = ["RECLASSIFICAÇÃO", "", f"Method: {method}", f"Classes: {n_classes}", f"Output: {output_path}", "", "Intervals:"]
            for i in range(1, len(breaks)):
                lines.append(f"Classe {i}: {breaks[i-1]:.6f} a {breaks[i]:.6f}")
            self.reclass_text.setPlainText("\n".join(lines))
            self.reclass_progress.setValue(100)
            self.iface.addRasterLayer(output_path, os.path.basename(output_path))
            self.menu_list.setCurrentRow(9)
            QMessageBox.information(self, "Completed", f"Reclassification done.\n\nRaster saved in:\n{output_path}")
        except Exception as e:
            self.reclass_progress.setValue(0)
            QMessageBox.critical(self, "Error", f"Error in reclassification:\n\n{str(e)}")

    def _run_validation(self):
        raster_path = self.validation_raster_edit.text().strip()
        if not raster_path:
            raster_path = self.last_reclass_output

        vector_path = self.validation_vector_edit.text().strip()

        if not raster_path or not os.path.exists(raster_path):
            QMessageBox.warning(self, "Warning", "Select a valid reclassified raster or perform the reclassification first.")
            return

        if not vector_path or not os.path.exists(vector_path):
            QMessageBox.warning(self, "Warning", "Select the validation input layer.")
            return

        html_path = self.validation_html_edit.text().strip()
        if not html_path:
            base_dir = self.final_output_edit.text().strip() or os.path.dirname(raster_path)
            project_name = self.project_name_edit.text().strip() or "fuzzy_project"
            html_path = os.path.join(base_dir, f"{project_name}_validation_report.html")
            self.validation_html_edit.setText(html_path)

        try:
            self.validation_progress.setValue(10)

            class_arr = self._read_raster_array(raster_path)

            self.validation_progress.setValue(35)

            validation_mask = self._rasterize_validation(vector_path, raster_path)

            self.validation_progress.setValue(60)

            class_summary, total_inside = self._compute_validation_class_proportions(class_arr, validation_mask)

            self.validation_progress.setValue(80)

            html = self._build_validation_html(raster_path, vector_path, class_summary, total_inside)

            if os.path.dirname(html_path):
                os.makedirs(os.path.dirname(html_path), exist_ok=True)

            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)

            self.last_validation_html = html_path

            lines = [
                "VALIDATION",
                "",
                f"Reclassified raster: {raster_path}",
                f"Validation layer: {vector_path}",
                f"Pixels/cells inside validation area: {total_inside}",
                "",
                "Proportion of classes inside validation area:",
            ]

            for item in class_summary:
                lines.append(
                    f" - Class {item['class']}: {item['count']} cells ({item['percentage']:.2f}%)"
                )

            lines.append("")
            lines.append(f"HTML report: {html_path}")

            self.validation_text.setPlainText("\n".join(lines))
            self.validation_progress.setValue(100)
            self.menu_list.setCurrentRow(10)

            QMessageBox.information(
                self,
                "Completed",
                f"Validation done.\n\nReport saved in:\n{html_path}"
            )

        except Exception as e:
            self.validation_progress.setValue(0)
            QMessageBox.critical(self, "Error", f"Validation error:\n\n{str(e)}")

    def _jenks_breaks(self, data, n_classes):
        data = np.sort(np.asarray(data, dtype=float))
        if data.size == 0:
            raise Exception("No valid data for Jenks.")
        if n_classes < 2:
            raise Exception("The number of classes must be at least 2.")
        if data.size < n_classes:
            raise Exception("The number of classes is greater than the number of valid pixels sampled.")

        mat1 = np.zeros((data.size + 1, n_classes + 1), dtype=int)
        mat2 = np.full((data.size + 1, n_classes + 1), np.inf, dtype=float)

        for i in range(1, n_classes + 1):
            mat1[0, i] = 1
            mat2[0, i] = 0.0

        for l in range(1, data.size + 1):
            s1 = s2 = w = 0.0
            for m in range(1, l + 1):
                i3 = l - m + 1
                val = data[i3 - 1]
                s2 += val * val
                s1 += val
                w += 1
                v = s2 - (s1 * s1) / w
                if i3 > 1:
                    for j in range(2, n_classes + 1):
                        if mat2[l, j] >= v + mat2[i3 - 1, j - 1]:
                            mat1[l, j] = i3
                            mat2[l, j] = v + mat2[i3 - 1, j - 1]
            mat1[l, 1] = 1
            mat2[l, 1] = v

        k = data.size
        kclass = [0.0] * (n_classes + 1)
        kclass[n_classes] = data[-1]
        count_num = n_classes
        while count_num >= 2:
            idx = mat1[k, count_num] - 2
            kclass[count_num - 1] = data[max(idx, 0)]
            k = mat1[k, count_num] - 1
            count_num -= 1
        kclass[0] = data[0]
        return kclass

    def _apply_breaks(self, arr, breaks):
        out = np.full(arr.shape, np.nan, dtype=np.float32)
        valid = ~np.isnan(arr)
        vals = arr[valid]
        classes = np.digitize(vals, breaks[1:-1], right=True) + 1
        out[valid] = classes.astype(np.float32)
        return out

    def _write_int_raster_like(self, reference_ds, array, output_path):
        gt = reference_ds.GetGeoTransform()
        proj = reference_ds.GetProjection()
        rows, cols = array.shape
        nodata_value = -9999
        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(output_path, cols, rows, 1, gdal.GDT_Int16, options=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"])
        if out_ds is None:
            raise Exception(f"It was not possible to create the output raster:\n{output_path}")
        out_ds.SetGeoTransform(gt)
        out_ds.SetProjection(proj)
        band = out_ds.GetRasterBand(1)
        band.WriteArray(np.where(np.isnan(array), nodata_value, array).astype(np.int16))
        band.SetNoDataValue(nodata_value)
        band.ComputeStatistics(False)
        band.FlushCache()
        out_ds.FlushCache()
        out_ds = None

    def _rasterize_validation(self, vector_path, reference_raster_path):
        ref_ds = gdal.Open(reference_raster_path)
        if ref_ds is None:
            raise Exception("It was not possible to open the reclassified reference raster.")
        gt = ref_ds.GetGeoTransform()
        proj = ref_ds.GetProjection()
        cols = ref_ds.RasterXSize
        rows = ref_ds.RasterYSize

        mem_drv = gdal.GetDriverByName("MEM")
        mask_ds = mem_drv.Create("", cols, rows, 1, gdal.GDT_Byte)
        mask_ds.SetGeoTransform(gt)
        mask_ds.SetProjection(proj)
        mask_ds.GetRasterBand(1).Fill(0)

        vec_ds = gdal.OpenEx(vector_path, gdal.OF_VECTOR)
        if vec_ds is None:
            raise Exception("The validation vector layer could not be opened.")

        layer = vec_ds.GetLayer(0)
        gdal.RasterizeLayer(mask_ds, [1], layer, burn_values=[1], options=["ALL_TOUCHED=TRUE"])
        arr = mask_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
        vec_ds = None
        mask_ds = None
        ref_ds = None
        return arr.astype(bool)

    def _compute_validation_class_proportions(self, class_arr, validation_mask):
        valid_inside = (~np.isnan(class_arr)) & validation_mask
        total_inside = int(np.sum(valid_inside))

        if total_inside == 0:
            raise Exception("The validation layer does not overlap valid raster cells.")

        class_values = class_arr[valid_inside]
        unique_classes = sorted(int(v) for v in np.unique(class_values))

        class_summary = []
        for cls in unique_classes:
            count = int(np.sum(class_values == float(cls)))
            percentage = (count / total_inside) * 100.0
            class_summary.append({
                "class": cls,
                "count": count,
                "percentage": percentage,
            })

        return class_summary, total_inside

    def _build_validation_html(self, raster_path, vector_path, class_summary, total_inside):
        rows = []
        for item in class_summary:
            rows.append(
                f"<tr><td>{item['class']}</td><td>{item['count']}</td><td>{item['percentage']:.2f}%</td></tr>"
            )

        now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        return f"""<!DOCTYPE html>
    <html lang='en'>
    <head>
    <meta charset='utf-8'>
    <title>Validation Report</title>
    <style>
    body{{font-family:Arial,sans-serif;margin:24px;line-height:1.45}}
    h1,h2{{color:#1f2937}}
    table{{border-collapse:collapse;width:100%;margin-top:12px}}
    th,td{{border:1px solid #ccc;padding:8px;text-align:left}}
    th{{background:#f3f4f6}}
    .small{{color:#555}}
    </style>
    </head>
    <body>
    <h1>Validation Report</h1>
    <p class='small'>Generated on {now}</p>

    <h2>Inputs</h2>
    <p>
    <b>Reclassified raster:</b> {raster_path}<br>
    <b>Validation layer:</b> {vector_path}<br>
    <b>Total cells/pixels inside validation area:</b> {total_inside}
    </p>

    <h2>Class proportions inside the validation area</h2>
    <table>
    <thead>
    <tr>
    <th>Class</th>
    <th>Cells/pixels inside validation area</th>
    <th>Proportion inside validation area</th>
    </tr>
    </thead>
    <tbody>
    {''.join(rows)}
    </tbody>
    </table>

    </body>
    </html>"""

    def _describe_membership_parameters(self, method, params):
        p1 = params.get("param_1")
        p2 = params.get("param_2")
        p3 = params.get("param_3")

        if method == "Linear":
            return f"A linear membership transformation was defined using a minimum value of {p1} and a maximum value of {p2}."
        if method == "Large membership":
            return f"A large membership function was adopted, with midpoint equal to {p1} and spread equal to {p2}, so that higher original values progressively received higher suitability scores."
        if method == "Small membership":
            return f"A small membership function was adopted, with midpoint equal to {p1} and spread equal to {p2}, so that lower original values progressively received higher suitability scores."
        if method == "Gaussian":
            return f"A Gaussian membership function was used with midpoint equal to {p1} and spread equal to {p2}, emphasizing intermediate values around the selected center."
        if method == "Power membership":
            return f"A power membership function was applied with lower bound equal to {p1}, upper bound equal to {p2}, and exponent equal to {p3}."
        return "A membership function was configured, but its parameters could not be described automatically."

    def _describe_overlay_method(self, method, gamma):
        if method == "AND":
            return "The fuzzy overlay stage adopted the AND operator, which retained the minimum membership value among the aligned fuzzy rasters for each pixel, thereby representing a restrictive decision rule."
        if method == "OR":
            return "The fuzzy overlay stage adopted the OR operator, which retained the maximum membership value among the aligned fuzzy rasters for each pixel, thereby representing an optimistic decision rule."
        if method == "PRODUCT":
            return "The fuzzy overlay stage adopted the PRODUCT operator, which multiplied the aligned membership values pixel by pixel, emphasizing cumulative penalization when one or more criteria showed low suitability."
        if method == "SUM":
            return "The fuzzy overlay stage adopted the SUM operator, calculated as one minus the product of the complements, in order to combine criteria while reducing excessive penalization associated with simple multiplication."
        if method == "GAMMA":
            return f"The fuzzy overlay stage adopted the GAMMA operator with gamma = {gamma}, combining the PRODUCT and SUM behaviours in a balanced manner and allowing intermediate compensation among criteria."
        return "The fuzzy overlay operator was informed in the workflow, but its methodological description could not be automatically expanded."

    def _html_list_from_items(self, items):
        return "".join(f"<li>{item}</li>" for item in items)

    def _generate_report(self):
        if not self.rasters:
            QMessageBox.warning(self, "Warning", "Add at least one raster before generating the report.")
            return

        html_path = self.report_html_edit.text().strip()
        if not html_path:
            base_dir = self.final_output_edit.text().strip() or self.workspace_edit.text().strip() or os.path.dirname(self.rasters[0]["path"])
            project_name = self.project_name_edit.text().strip() or "fuzzy_project"
            html_path = os.path.join(base_dir, f"{project_name}_methodology_report.html")
            self.report_html_edit.setText(html_path)

        try:
            self.report_progress.setValue(10)
            project_name = self.project_name_edit.text().strip() or "Unnamed project"
            overlay_method = self.overlay_method.currentText()
            gamma = self.gamma_spin.value()
            reference_raster = self.reference_raster_edit.text().strip()
            resampling = self.resampling_combo.currentText()
            reclass_method = self.reclass_method_combo.currentText()
            reclass_classes = self.reclass_classes_spin.value()
            sample_size = self.sens_sample_spin.value()
            threshold = self.sens_threshold_spin.value()
            validation_layer = self.validation_vector_edit.text().strip() or "an external reference vector layer"
            now = datetime.datetime.now().strftime("%d %B %Y, %H:%M")

            split_mode = "single" if self.split_mode_single_radio.isChecked() else "multiple"
            train_percent = self.split_train_percent_spin.value()
            validation_percent = 100 - train_percent
            random_seed = self.split_seed_spin.value()

            def format_variable_list(names):
                if not names:
                    return ""
                if len(names) == 1:
                    return names[0]
                if len(names) == 2:
                    return f"{names[0]} and {names[1]}"
                return ", ".join(names[:-1]) + f", and {names[-1]}"

            def membership_sentence(raster):
                label = escape(self._get_raster_label(raster))
                method = raster.get("membership", "") or "Not configured"
                params = raster.get("params", {})
                p1 = params.get("param_1")
                p2 = params.get("param_2")
                p3 = params.get("param_3")

                if method == "Linear":
                    return (
                        f"The variable <strong>{label}</strong> was transformed using a linear membership function "
                        f"defined by the thresholds {p1} and {p2}."
                    )
                if method == "Large membership":
                    return (
                        f"The variable <strong>{label}</strong> was transformed using a large membership function, "
                        f"with midpoint = {p1} and spread = {p2}, assigning progressively higher membership values to higher input values."
                    )
                if method == "Small membership":
                    return (
                        f"The variable <strong>{label}</strong> was transformed using a small membership function, "
                        f"with midpoint = {p1} and spread = {p2}, assigning progressively higher membership values to lower input values."
                    )
                if method == "Gaussian":
                    return (
                        f"The variable <strong>{label}</strong> was transformed using a Gaussian membership function, "
                        f"centred at {p1} with spread = {p2}."
                    )
                if method == "Power membership":
                    return (
                        f"The variable <strong>{label}</strong> was transformed using a power membership function, "
                        f"with lower bound = {p1}, upper bound = {p2}, and exponent = {p3}."
                    )
                return (
                    f"The variable <strong>{label}</strong> was included in the workflow, although its membership "
                    f"function parameters were not fully configured in the current project."
                )

            def build_overlay_paragraph(method, gamma_value):
                if method == "AND":
                    return (
                        "The fuzzy layers were integrated using the AND operator, which represents a restrictive "
                        "combination logic in which low suitability values exert strong control over the final result."
                    )
                if method == "OR":
                    return (
                        "The fuzzy layers were integrated using the OR operator, which represents a permissive "
                        "combination logic and emphasizes the highest suitability responses among the criteria."
                    )
                if method == "PRODUCT":
                    return (
                        "The fuzzy layers were integrated using the PRODUCT operator, which intensifies restrictive "
                        "effects by progressively reducing the final suitability values when one or more criteria present low membership."
                    )
                if method == "SUM":
                    return (
                        "The fuzzy layers were integrated using the SUM operator, which favours cumulative responses "
                        "and increases the compensatory effect among criteria."
                    )
                if method == "GAMMA":
                    return (
                        f"The fuzzy layers were integrated using the GAMMA operator with a gamma coefficient of {gamma_value}, "
                        "combining the restrictive behaviour of AND-like logic with the compensatory behaviour of OR-like logic. "
                        "This configuration produced a predominantly compensatory integration while preserving the influence of less suitable criteria."
                    )
                return "The fuzzy layers were integrated using the operator selected by the user in the overlay module."

            self.report_progress.setValue(25)
            variable_names = [escape(self._get_raster_label(r)) for r in self.rasters]
            variable_names_text = format_variable_list(variable_names)
            database_items = [
                f'<li><strong>{escape(self._get_raster_label(r))}</strong> <span style="color:#475569;">(source raster: {escape(r["name"])})</span></li>'
                for r in self.rasters
            ]
            processing_sentences = [membership_sentence(r) for r in self.rasters]

            if split_mode == "multiple":
                split_paragraph = (
                    "Model training was based on representative vector samples organised as multiple independent input layers. "
                    f"These layers were randomly shuffled using a fixed random seed ({random_seed}) and then partitioned into "
                    f"{train_percent}% for training and {validation_percent}% for validation. "
                    "Under this strategy, each vector layer was treated as an individual sampling unit."
                )
            else:
                split_paragraph = (
                    "Model training was based on a single vector layer containing multiple features. "
                    f"The internal features were randomly shuffled using a fixed random seed ({random_seed}) and subsequently partitioned into "
                    f"{train_percent}% for training and {validation_percent}% for validation, generating independent subsets for calibration and validation."
                )

            processing_paragraph = " ".join(processing_sentences) if processing_sentences else ""

            if reference_raster:
                alignment_paragraph = (
                    f"Before overlay, all fuzzy rasters were spatially aligned to the reference raster <strong>{escape(reference_raster)}</strong> "
                    f"using the <strong>{escape(resampling)}</strong> resampling method, ensuring pixel-to-pixel correspondence among the criteria."
                )
            else:
                alignment_paragraph = (
                    f"Before overlay, all fuzzy rasters were spatially aligned to the first available fuzzy raster as the reference grid, "
                    f"using the <strong>{escape(resampling)}</strong> resampling method, ensuring pixel-to-pixel correspondence among the criteria."
                )

            overlay_paragraph = build_overlay_paragraph(overlay_method, gamma)

            if reclass_method == "Equal intervals" and reclass_classes == 5:
                reclass_table = """
                <table>
                    <thead>
                        <tr><th>Fuzzy set</th><th>Classes</th></tr>
                    </thead>
                    <tbody>
                        <tr><td>0.0 – 0.2</td><td>Very low</td></tr>
                        <tr><td>0.2 – 0.4</td><td>Low</td></tr>
                        <tr><td>0.4 – 0.6</td><td>Moderate</td></tr>
                        <tr><td>0.6 – 0.8</td><td>High</td></tr>
                        <tr><td>0.8 – 1.0</td><td>Very High</td></tr>
                    </tbody>
                </table>
                """
            else:
                reclass_table = ""

            variable_table_rows = []
            for idx, raster in enumerate(self.rasters, start=1):
                label = escape(self._get_raster_label(raster))
                original = escape(raster['name'])
                method = escape(raster.get('membership', '') or 'Not configured')
                params = raster.get('params', {})
                params_text = ", ".join(f"{escape(str(k))} = {escape(str(v))}" for k, v in params.items()) if params else "Not configured"
                variable_table_rows.append(
                    f"<tr><td>{idx}</td><td>{label}</td><td>{original}</td><td>{method}</td><td>{params_text}</td></tr>"
                )

            self.report_progress.setValue(60)
            sections = [
                f"<h1>3. MATERIALS AND METHODS</h1><p><strong>Project:</strong> {escape(project_name)}<br><strong>Generated on:</strong> {escape(now)}</p>",
                (
                    "<h2>3.1 General structure of the modelling procedure</h2>"
                    f"<p>This study was developed based on multicriteria spatial modelling using fuzzy logic. The modelling procedure was carried out in a Geographic Information System (GIS) environment, integrating biophysical and anthropogenic variables relevant to the proposed analysis. The input database consisted of the following variables: <strong>{variable_names_text}</strong>. All variables were converted to raster format and standardised in terms of spatial resolution, geographic extent, and reference system, ensuring compatibility for joint processing.</p>"
                ),
                (
                    "<h2>3.2 Sampling basis and data partitioning</h2>"
                    f"<p>The model was constructed from representative vector samples of the phenomenon under analysis. Data partitioning into training and validation subsets was performed by means of controlled random sampling, ensuring reproducibility of the workflow. {split_paragraph} This procedure resulted in two independent subsets, one used for model calibration and the other reserved exclusively for validation, thereby preserving independence between stages and reducing the likelihood of overfitting.</p>"
                ),
                (
                    "<h2>3.3 Training analysis and variable hierarchy</h2>"
                    "<p>The training stage was designed to identify the relationship between the input variables and the analysed phenomenon based on the spatial distribution of the training samples. Initially, each raster variable was reclassified into five equal-interval classes considering the range of observed values. Subsequently, the training areas were overlaid onto the reclassified rasters, allowing the frequency of occurrence of each class inside the training samples to be quantified. On the basis of this distribution, the classes were ranked according to their relative representativeness and then remapped onto an ordinal scale from 1 to 5, where class 1 corresponds to the highest representativeness and class 5 to the lowest. This procedure transformed the empirical information derived from the samples into a gradient of relative importance, which served as the basis for the fuzzification stage.</p>"
                ),
                (
                    "<h2>3.4 Variable fuzzification</h2>"
                    "<p>The reclassified variables were transformed into continuous fuzzy membership surfaces, with values normalised to the interval from 0 to 1. In this framework, the most representative classes received membership values closer to 1, whereas the least representative classes approached 0, reproducing the expected suitability gradient in the modelling procedure. In conceptual terms, the decreasing linear behaviour adopted in the workflow expresses an inverse monotonic relationship between the ranked classes and the membership degree, so that the higher the representativeness observed in training, the higher the fuzzy response assigned to the criterion (Ramalho et al., 2023; Aragão et al., 2023). (RAMALHO, A. H. C. et al. Optimal allocation model of forest fire detection towers in protected areas based on fire occurrence risk: Where and how to act? Canadian Journal of Forest Research, v. 54, n. 1, 2024. DOI: 10.1139/cjfr-2023-0084; ARAGÃO, M. de A. et al. Risk of forest fires occurrence on a transition island Amazon-Cerrado: Where to act? Forest Ecology and Management, v. 536, n. 1, 2023. DOI: 10.1016/j.foreco.2023.120858) </p>"
                    f"<p>{processing_paragraph}</p>"
                ),
                (
                    "<h2>3.5 Integration of variables by fuzzy overlay</h2>"
                    f"<p>{alignment_paragraph}</p>"
                    f"<p>{overlay_paragraph}</p>"
                    "<p>The result of this stage was a continuous raster representing the integrated response of the model for the complete set of analysed variables.</p>"
                ),
                (
                    "<h2>3.6 Sensitivity analysis</h2>"
                    f"<p>The individual influence of the variables on the model response was evaluated through a leave-one-out sensitivity analysis. In this procedure, the model was recalculated successively with the removal of one variable at a time, and the resulting surfaces were compared with the complete model. The comparisons were based on the absolute mean difference, the root mean square error (RMSE), the Pearson correlation coefficient, and the percentage of sampled pixels whose change exceeded the threshold of <strong>{threshold}</strong>. Up to <strong>{sample_size}</strong> valid pixels were used in this stage. Based on these indicators, the variables were classified according to their relative importance into three categories: relevant, moderately relevant, and irrelevant. This stage enabled the robustness of the modelling framework to be assessed and supported the interpretation of the final suitability response.</p>"
                ),
                (
                    "<h2>3.7 Reclassification of the final result</h2>"
                    f"<p>The continuous map resulting from the fuzzy overlay was reclassified using the <strong>{escape(reclass_method)}</strong> method into <strong>{reclass_classes}</strong> classes. This procedure was adopted to facilitate cartographic interpretation of the final model response, converting the continuous suitability surface into discrete classes without compromising its analytical coherence (Da Silva et al., 2026). (DA SILVA, B. L. et al. Fuzzy modeling in a GIS environment for identifying the seasonality of forest fire risk in a protected area in the Brazilian Amazon. Remote Sensing Applications: Society and Environment, v. 41, n. 1, 2026. DOI: 10.1016/j.rsase.2025.101846)</p>"
                    f"{reclass_table}"
                ),
                (
                    "<h2>3.8 Model validation</h2>"
                    f"<p>Model validation was performed using the independent subset of samples previously reserved for this purpose. The validation layer used in the workflow was <strong>{escape(validation_layer)}</strong>. The validation samples were superimposed on the reclassified suitability map, making it possible to quantify the frequency and percentage distribution of the validation occurrences in each suitability class. The quality of the model was interpreted from the concentration of validation samples in the higher suitability classes, particularly those representing stronger agreement with the set of criteria incorporated into the fuzzy model.</p>"
                ),
                f"<h2>Configured variables</h2><ul>{''.join(database_items)}</ul><table><thead><tr><th>#</th><th>Variable name</th><th>Original raster name</th><th>Membership function</th><th>Parameters</th></tr></thead><tbody>{''.join(variable_table_rows)}</tbody></table>",
            ]

            outputs = []
            if self.last_overlay_output:
                outputs.append(f"<li>Final overlay raster: <code>{escape(self.last_overlay_output)}</code></li>")
            if self.last_reclass_output:
                outputs.append(f"<li>Reclassified raster: <code>{escape(self.last_reclass_output)}</code></li>")
            if self.last_training_report_html:
                outputs.append(f"<li>Training analysis report: <code>{escape(self.last_training_report_html)}</code></li>")
            if self.last_validation_html:
                outputs.append(f"<li>Validation report: <code>{escape(self.last_validation_html)}</code></li>")
            outputs.append(f"<li>Methodology report: <code>{escape(html_path)}</code></li>")
            sections.append(f"<h2>Generated outputs</h2><ul>{''.join(outputs)}</ul>")

            html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Materials and Methods</title>
<style>
body{{font-family:Arial,sans-serif;margin:32px;line-height:1.7;color:#1e293b;background:#ffffff;}}
h1,h2{{color:#0f172a;}}
p,li{{font-size:14px;text-align:justify;}}
ul,ol{{margin-top:8px;margin-bottom:18px;}}
code{{background:#f1f5f9;padding:2px 5px;border-radius:4px;}}
table{{border-collapse:collapse;width:100%;margin-top:14px;margin-bottom:22px;}}
th,td{{border:1px solid #cbd5e1;padding:8px;vertical-align:top;text-align:left;font-size:13px;}}
th{{background:#f8fafc;}}
.section{{margin-bottom:28px;}}
</style>
</head>
<body>
{''.join(f'<div class="section">{section}</div>' for section in sections)}
</body>
</html>"""

            self.report_progress.setValue(85)
            if os.path.dirname(html_path):
                os.makedirs(os.path.dirname(html_path), exist_ok=True)
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)
            self.last_report_html = html_path

            summary_lines = [
                "METHODOLOGY REPORT GENERATED",
                "",
                f"Project: {project_name}",
                f"Variables described: {len(self.rasters)}",
                f"Overlay method: {overlay_method}",
                f"Report path: {html_path}",
                "",
                "The Generate report module was updated with the scientific Materials and Methods model in English, using the variable names captured from the Fuzzy settings module.",
            ]
            self.report_text.setPlainText("\n".join(summary_lines))
            self.report_progress.setValue(100)
            self.menu_list.setCurrentRow(11)
            QMessageBox.information(self, "Completed", f"Methodology report generated successfully.\n\nSaved in:\n{html_path}")

        except Exception as e:
            self.report_progress.setValue(0)
            QMessageBox.critical(self, "Error", f"Error while generating the methodology report:\n\n{str(e)}")

    def _fuzzify_raster(self, raster_path, method, params, output_path):
        ds = gdal.Open(raster_path)
        if ds is None:
            raise Exception(f"It was not possible to open the raster:\n{raster_path}")

        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()

        cols = ds.RasterXSize
        rows = ds.RasterYSize
        gt = ds.GetGeoTransform()
        proj = ds.GetProjection()
        nodata_value = -9999.0

        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(
            output_path,
            cols,
            rows,
            1,
            gdal.GDT_Float32,
            options=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"],
        )
        if out_ds is None:
            raise Exception(f"It was not possible to create the output raster:\n{output_path}")

        out_ds.SetGeoTransform(gt)
        out_ds.SetProjection(proj)
        out_band = out_ds.GetRasterBand(1)
        out_band.SetNoDataValue(nodata_value)

        block_rows = 32

        for yoff in range(0, rows, block_rows):
            ysize = min(block_rows, rows - yoff)

            arr = band.ReadAsArray(0, yoff, cols, ysize).astype(np.float32)

            if nodata is not None:
                if np.isnan(nodata):
                    arr[np.isnan(arr)] = np.nan
                else:
                    arr[arr == nodata] = np.nan

            if method == "Linear":
                result = self._fuzzy_linear(arr, params["param_1"], params["param_2"])
            elif method == "Large membership":
                result = self._fuzzy_large(arr, params["param_1"], params["param_2"])
            elif method == "Small membership":
                result = self._fuzzy_small(arr, params["param_1"], params["param_2"])
            elif method == "Gaussian":
                result = self._fuzzy_gaussian(arr, params["param_1"], params["param_2"])
            elif method == "Power membership":
                result = self._fuzzy_power(arr, params["param_1"], params["param_2"], params.get("param_3", 2.0))
            else:
                raise Exception(f"Fuzzy function not supported: {method}")

            out_band.WriteArray(np.where(np.isnan(result), nodata_value, result), 0, yoff)

        out_band.FlushCache()
        out_ds.FlushCache()
        out_ds = None
        ds = None

    def _align_raster(self, input_path, reference_path, output_path, resampling_name):
        ref_ds = gdal.Open(reference_path)
        if ref_ds is None:
            raise Exception(f"The reference raster could not be opened:\n{reference_path}")

        gt = ref_ds.GetGeoTransform()
        x_min = gt[0]
        y_max = gt[3]
        pixel_width = gt[1]
        pixel_height = abs(gt[5])
        width = ref_ds.RasterXSize
        height = ref_ds.RasterYSize
        x_max = x_min + (width * pixel_width)
        y_min = y_max - (height * pixel_height)
        projection = ref_ds.GetProjection()

        resampling_map = {
            "Nearest neighbour": gdal.GRA_NearestNeighbour,
            "Bilinear": gdal.GRA_Bilinear,
            "Cubic": gdal.GRA_Cubic,
            "Cubic spline": gdal.GRA_CubicSpline,
            "Lanczos": gdal.GRA_Lanczos,
            "Average": gdal.GRA_Average,
            "Mode": gdal.GRA_Mode,
        }

        warp_options = gdal.WarpOptions(
            format="GTiff",
            outputBounds=(x_min, y_min, x_max, y_max),
            width=width,
            height=height,
            dstSRS=projection,
            resampleAlg=resampling_map.get(resampling_name, gdal.GRA_NearestNeighbour),
            multithread=True,
            dstNodata=np.nan,
            outputType=gdal.GDT_Float32,
        )

        gdal.Warp(output_path, input_path, options=warp_options)
        ref_ds = None

    def _overlay_rasters(self, raster_paths, method, gamma, output_path):
        datasets = []
        bands = []

        try:
            for path in raster_paths:
                ds = gdal.Open(path)
                if ds is None:
                    raise Exception(f"It was not possible to open the aligned raster:\n{path}")
                datasets.append(ds)
                bands.append(ds.GetRasterBand(1))

            first_ds = datasets[0]
            cols = first_ds.RasterXSize
            rows = first_ds.RasterYSize
            gt = first_ds.GetGeoTransform()
            proj = first_ds.GetProjection()
            nodata_value = -9999.0

            driver = gdal.GetDriverByName("GTiff")
            out_ds = driver.Create(
                output_path,
                cols,
                rows,
                1,
                gdal.GDT_Float32,
                options=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"],
            )
            if out_ds is None:
                raise Exception(f"It was not possible to create the output raster:\n{output_path}")

            out_ds.SetGeoTransform(gt)
            out_ds.SetProjection(proj)
            out_band = out_ds.GetRasterBand(1)
            out_band.SetNoDataValue(nodata_value)

            block_rows = 32

            for yoff in range(0, rows, block_rows):
                ysize = min(block_rows, rows - yoff)

                arrays = []
                mask = None

                for band in bands:
                    arr = band.ReadAsArray(0, yoff, cols, ysize).astype(np.float32)
                    nodata = band.GetNoDataValue()

                    if nodata is not None:
                        if np.isnan(nodata):
                            arr[np.isnan(arr)] = np.nan
                        else:
                            arr[arr == nodata] = np.nan

                    current_mask = np.isnan(arr)
                    if mask is None:
                        mask = current_mask.copy()
                    else:
                        mask |= current_mask

                    arrays.append(arr)

                if method == "AND":
                    result = arrays[0].copy()
                    for arr in arrays[1:]:
                        result = np.fmin(result, arr)

                elif method == "OR":
                    result = arrays[0].copy()
                    for arr in arrays[1:]:
                        result = np.fmax(result, arr)

                elif method == "PRODUCT":
                    result = np.ones((ysize, cols), dtype=np.float32)
                    for arr in arrays:
                        result *= np.where(np.isnan(arr), 1.0, arr)

                elif method == "SUM":
                    prod_component = np.ones((ysize, cols), dtype=np.float32)
                    for arr in arrays:
                        prod_component *= (1.0 - np.where(np.isnan(arr), 0.0, arr))
                    result = 1.0 - prod_component

                elif method == "GAMMA":
                    product = np.ones((ysize, cols), dtype=np.float32)
                    sum_component_prod = np.ones((ysize, cols), dtype=np.float32)

                    for arr in arrays:
                        safe_for_product = np.where(np.isnan(arr), 1.0, arr)
                        safe_for_sum = np.where(np.isnan(arr), 0.0, arr)

                        product *= safe_for_product
                        sum_component_prod *= (1.0 - safe_for_sum)

                    sum_component = 1.0 - sum_component_prod
                    result = np.power(sum_component, gamma) * np.power(product, 1.0 - gamma)

                else:
                    raise Exception(f"Overlay method not supported: {method}")

                result = np.where(mask, np.nan, result).astype(np.float32)
                out_band.WriteArray(np.where(np.isnan(result), nodata_value, result), 0, yoff)

            out_band.FlushCache()
            out_ds.FlushCache()
            out_ds = None

        finally:
            for ds in datasets:
                ds = None

    def _write_raster_like(self, reference_ds, array, output_path):
        gt = reference_ds.GetGeoTransform()
        proj = reference_ds.GetProjection()
        rows, cols = array.shape
        nodata_value = -9999.0

        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(
            output_path,
            cols,
            rows,
            1,
            gdal.GDT_Float32,
            options=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"],
        )
        if out_ds is None:
            raise Exception(f"It was not possible to create the output raster:\n{output_path}")

        out_ds.SetGeoTransform(gt)
        out_ds.SetProjection(proj)
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(np.where(np.isnan(array), nodata_value, array))
        out_band.SetNoDataValue(nodata_value)
        out_band.FlushCache()
        out_ds.FlushCache()
        out_ds = None

    def _fuzzy_linear(self, arr, low, high):
        result = np.full(arr.shape, np.nan, dtype=np.float32)
        valid = ~np.isnan(arr)

        if low == high:
            raise Exception("In a linear function, the minimum and maximum values ​​cannot be equal.")

        if low < high:
            temp = (arr[valid] - low) / (high - low)
        else:
            temp = (low - arr[valid]) / (low - high)

        result[valid] = np.clip(temp, 0.0, 1.0)
        return result

    def _fuzzy_large(self, arr, midpoint, spread):
        result = np.full(arr.shape, np.nan, dtype=np.float32)
        valid = ~np.isnan(arr)

        if midpoint == 0:
            raise Exception("In the Large function, the midpoint cannot be zero.")
        if spread <= 0:
            raise Exception("In the Large function, the spread must be greater than zero.")

        x = arr[valid]
        result[valid] = 1.0 / (1.0 + np.power((x / midpoint), -spread))
        return np.clip(result, 0.0, 1.0)

    def _fuzzy_small(self, arr, midpoint, spread):
        result = np.full(arr.shape, np.nan, dtype=np.float32)
        valid = ~np.isnan(arr)

        if midpoint == 0:
            raise Exception("In the Small function, the midpoint cannot be zero.")
        if spread <= 0:
            raise Exception("In the Small function, the spread must be greater than zero.")

        x = arr[valid]
        result[valid] = 1.0 / (1.0 + np.power((x / midpoint), spread))
        return np.clip(result, 0.0, 1.0)

    def _fuzzy_gaussian(self, arr, midpoint, spread):
        result = np.full(arr.shape, np.nan, dtype=np.float32)
        valid = ~np.isnan(arr)

        if spread <= 0:
            raise Exception("In a Gaussian function, the spread must be greater than zero.")

        x = arr[valid]
        result[valid] = np.exp(-((x - midpoint) ** 2) / (2.0 * (spread ** 2)))
        return np.clip(result, 0.0, 1.0)

    def _fuzzy_power(self, arr, low, high, exponent):
        result = np.full(arr.shape, np.nan, dtype=np.float32)
        valid = ~np.isnan(arr)

        if low == high:
            raise Exception("In the Power function, low and high cannot be the same.")
        if exponent <= 0:
            raise Exception("NIn the Power function, the exponent must be greater than zero.")

        if low < high:
            temp = (arr[valid] - low) / (high - low)
        else:
            temp = (low - arr[valid]) / (low - high)

        result[valid] = np.power(np.clip(temp, 0.0, 1.0), exponent)
        return np.clip(result, 0.0, 1.0)
