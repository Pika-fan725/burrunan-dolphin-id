"""A small desktop interface for the dorsal-fin identification program."""

import base64
import csv
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2

import fin_curvature
import identify


PROGRAM_FOLDER = os.path.dirname(os.path.abspath(__file__))
IMAGE_FILETYPES = [
    ('Image files', '*.jpg *.jpeg *.png *.bmp *.tif *.tiff'),
    ('All files', '*.*'),
]


class DatasetViewer(tk.Toplevel):
    """Browse every live image and send any pair to the comparison window."""

    def __init__(self, interface):
        super().__init__(interface)
        self.interface = interface
        self.title('Dataset Viewer')
        self.geometry('900x610')
        self.minsize(720, 500)
        self.preview = None

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(1, weight=1)

        heading = ttk.Frame(self, padding=(12, 12, 12, 6))
        heading.grid(row=0, column=0, columnspan=2, sticky='ew')
        heading.columnconfigure(0, weight=1)
        ttk.Label(heading, text='Select one image to inspect or two images to compare.').grid(row=0, column=0, sticky='w')
        ttk.Button(heading, text='Refresh', command=self.refresh).grid(row=0, column=1, padx=(8, 0))

        self.images = ttk.Treeview(self, columns=('folder', 'file'), show='headings', selectmode='extended')
        self.images.heading('folder', text='Folder')
        self.images.heading('file', text='Photo')
        self.images.column('folder', width=130, anchor='w')
        self.images.column('file', width=300, anchor='w')
        self.images.grid(row=1, column=0, sticky='nsew', padx=(12, 6), pady=6)
        self.images.bind('<<TreeviewSelect>>', self.show_preview)

        self.preview_label = ttk.Label(self, anchor='center')
        self.preview_label.grid(row=1, column=1, sticky='nsew', padx=(6, 12), pady=6)

        controls = ttk.Frame(self, padding=(12, 6, 12, 12))
        controls.grid(row=2, column=0, columnspan=2, sticky='ew')
        controls.columnconfigure(0, weight=1)
        ttk.Button(controls, text='Compare selected', command=self.compare_selected).grid(row=0, column=1)

        self.refresh()

    def dataset_paths(self):
        folders = (
            ('Catalogue', self.interface.catalogue_folder.get()),
            ('Sightings', identify.SIGHTINGS_FOLDER),
            ('Human review', identify.REVIEW_FOLDER),
        )
        paths = []
        for label, folder in folders:
            if os.path.isdir(folder):
                paths.extend((label, path) for path in identify.list_images(folder))
        return paths

    def refresh(self):
        self.images.delete(*self.images.get_children())
        for index, (folder, path) in enumerate(self.dataset_paths()):
            self.images.insert('', tk.END, iid=str(index), values=(folder, os.path.basename(path)), tags=(path,))
        self.preview_label.configure(image='')
        self.preview = None

    def selected_paths(self):
        paths = []
        for item in self.images.selection():
            paths.append(self.images.item(item, 'tags')[0])
        return paths

    def show_preview(self, _event=None):
        paths = self.selected_paths()
        if not paths:
            return
        image = cv2.imread(paths[0])
        if image is None:
            return
        description = fin_curvature.describe_fin(image)
        if description is not None and self.interface.show_outline.get():
            image = fin_curvature.draw_outline(image, description)
        self.preview = self.interface.preview_image(image, os.path.basename(paths[0]), 480, 440)
        self.preview_label.configure(image=self.preview)

    def compare_selected(self):
        paths = self.selected_paths()
        if len(paths) != 2:
            messagebox.showinfo('Compare photos', 'Select exactly two photos to compare.')
            return
        self.interface.open_comparison(paths[0], paths[1])


class DolphinInterface(tk.Tk):
    """Desktop controls around the matching code in identify.py."""

    def __init__(self):
        super().__init__()
        self.title('Dolphin Fin Identifier')
        self.geometry('1240x860')
        self.minsize(1040, 720)

        self.catalogue_folder = tk.StringVar(value=identify.CATALOGUE_FOLDER)
        # The traced outline can hide the real fin edge, so it can be switched off.
        self.show_outline = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value='Choose sighting photos, then select Identify.')
        self.sighting_paths = []
        self.results = []
        self.worker_messages = queue.Queue()
        self.preview_images = []

        self._build()
        self.load_sightings_folder(quiet=True)

    def load_sightings_folder(self, quiet=False):
        """Fill the list with everything in the sightings/ folder."""
        added = 0
        for path in identify.list_images(identify.SIGHTINGS_FOLDER):
            if path not in self.sighting_paths:
                self.sighting_paths.append(path)
                self.sightings_list.insert(tk.END, os.path.basename(path))
                added += 1
        if not quiet:
            self.status.set('Added %d photo%s from the sightings folder.' % (added, '' if added == 1 else 's'))

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        settings = ttk.Frame(self, padding=(14, 12, 14, 8))
        settings.grid(row=0, column=0, sticky='ew')
        settings.columnconfigure(1, weight=1)
        ttk.Label(settings, text='Known-fin catalogue').grid(row=0, column=0, sticky='w', padx=(0, 8))
        ttk.Entry(settings, textvariable=self.catalogue_folder).grid(row=0, column=1, sticky='ew')
        ttk.Button(settings, text='Choose folder', command=self.choose_catalogue).grid(row=0, column=2, padx=(8, 0))

        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.grid(row=1, column=0, sticky='nsew', padx=14, pady=8)

        left = ttk.Frame(main, padding=4)
        right = ttk.Frame(main, padding=4)
        main.add(left, weight=1)
        main.add(right, weight=3)

        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)
        ttk.Label(left, text='Sighting photos').grid(row=0, column=0, sticky='w', pady=(0, 6))
        self.sightings_list = tk.Listbox(left, selectmode=tk.EXTENDED, exportselection=False)
        self.sightings_list.grid(row=1, column=0, sticky='nsew')
        buttons = ttk.Frame(left)
        buttons.grid(row=2, column=0, sticky='ew', pady=(8, 0))
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        ttk.Button(buttons, text='Add photos', command=self.add_sightings).grid(row=0, column=0, sticky='ew', padx=(0, 4))
        ttk.Button(buttons, text='Remove selected', command=self.remove_sightings).grid(row=0, column=1, sticky='ew', padx=(4, 0))
        ttk.Button(buttons, text='Reload sightings folder', command=self.load_sightings_folder).grid(
            row=1, column=0, columnspan=2, sticky='ew', pady=(4, 0))

        # Every catalogue dolphin's score for the selected sighting, so the
        # runner-up is visible - a close second is the main reason a result
        # should be checked by a person.
        left.rowconfigure(4, weight=1)
        ttk.Label(left, text='Scores for selected sighting').grid(row=3, column=0, sticky='w', pady=(14, 6))
        self.ranking_table = ttk.Treeview(left, columns=('dolphin', 'score', 'side'), show='headings', height=6)
        for name, title, width in (('dolphin', 'Dolphin', 110), ('score', 'Score', 70), ('side', 'Side', 100)):
            self.ranking_table.heading(name, text=title)
            self.ranking_table.column(name, width=width, anchor='w')
        self.ranking_table.grid(row=4, column=0, sticky='nsew')

        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        right.rowconfigure(3, weight=3)
        ttk.Label(right, text='Identification results').grid(row=0, column=0, sticky='w', pady=(0, 6))
        self.result_table = ttk.Treeview(
            right, columns=('photo', 'candidate', 'score', 'decision'), show='headings', selectmode='browse')
        for name, title, width in (
                ('photo', 'Sighting', 260), ('candidate', 'Candidate', 130),
                ('score', 'Score', 90), ('decision', 'Decision', 145)):
            self.result_table.heading(name, text=title)
            self.result_table.column(name, width=width, anchor='w')
        self.result_table.grid(row=1, column=0, sticky='nsew')
        self.result_table.bind('<<TreeviewSelect>>', self.show_selected_result)
        self.result_table.tag_configure('candidate_match', background='#dcf3dc')
        self.result_table.tag_configure('human_review', background='#fbeccc')
        self.result_table.tag_configure('new_or_unusable', background='#e6e6e6')

        ttk.Checkbutton(right, text='Show traced outline on photos', variable=self.show_outline,
                        command=self.show_selected_result).grid(row=2, column=0, sticky='w', pady=(10, 0))
        self.previews = ttk.Frame(right)
        self.previews.grid(row=3, column=0, sticky='nsew', pady=(6, 0))
        for column in range(2):
            self.previews.columnconfigure(column, weight=1)
        self.sighting_preview = ttk.Label(self.previews, anchor='center')
        self.match_preview = ttk.Label(self.previews, anchor='center')
        self.sighting_preview.grid(row=0, column=0, sticky='nsew', padx=(0, 6))
        self.match_preview.grid(row=0, column=1, sticky='nsew', padx=(6, 0))
        # The notch-pattern graphs are what the decision is actually made on:
        # a notch in the fin is a dip in the line. Two photos of the same
        # dolphin should show the same dips in the same places.
        self.sighting_graph = ttk.Label(self.previews, anchor='center')
        self.match_graph = ttk.Label(self.previews, anchor='center')
        self.sighting_graph.grid(row=1, column=0, sticky='nsew', padx=(0, 6), pady=(6, 0))
        self.match_graph.grid(row=1, column=1, sticky='nsew', padx=(6, 0), pady=(6, 0))

        footer = ttk.Frame(self, padding=(14, 8, 14, 12))
        footer.grid(row=2, column=0, sticky='ew')
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.status).grid(row=0, column=0, sticky='w')
        self.export_button = ttk.Button(footer, text='Export review CSV', command=self.export_csv, state=tk.DISABLED)
        self.export_button.grid(row=0, column=1, padx=(8, 0))
        ttk.Button(footer, text='Browse dataset', command=self.open_dataset_viewer).grid(row=0, column=2, padx=(8, 0))
        self.compare_button = ttk.Button(footer, text='Compare', command=self.compare_selected_result, state=tk.DISABLED)
        self.compare_button.grid(row=0, column=3, padx=(8, 0))
        self.identify_button = ttk.Button(footer, text='Identify', command=self.start_identification)
        self.identify_button.grid(row=0, column=4, padx=(8, 0))

    def choose_catalogue(self):
        folder = filedialog.askdirectory(initialdir=self.catalogue_folder.get() or PROGRAM_FOLDER)
        if folder:
            self.catalogue_folder.set(folder)

    def add_sightings(self):
        paths = filedialog.askopenfilenames(initialdir=PROGRAM_FOLDER, filetypes=IMAGE_FILETYPES)
        for path in paths:
            if path not in self.sighting_paths:
                self.sighting_paths.append(path)
                self.sightings_list.insert(tk.END, os.path.basename(path))

    def remove_sightings(self):
        indexes = list(self.sightings_list.curselection())
        for index in reversed(indexes):
            del self.sighting_paths[index]
            self.sightings_list.delete(index)

    def start_identification(self):
        catalogue_folder = self.catalogue_folder.get()
        if not os.path.isdir(catalogue_folder):
            messagebox.showerror('Catalogue folder', 'Choose a folder containing known fin photos.')
            return
        if not self.sighting_paths:
            messagebox.showinfo('Sighting photos', 'Add at least one side-on fin photo first.')
            return

        self.identify_button.configure(state=tk.DISABLED)
        self.export_button.configure(state=tk.DISABLED)
        self.compare_button.configure(state=tk.DISABLED)
        self.status.set('Tracing fins and comparing notch patterns...')
        self.results = []
        self.result_table.delete(*self.result_table.get_children())
        thread = threading.Thread(target=self._identify_worker,
                                  args=(catalogue_folder, list(self.sighting_paths)), daemon=True)
        thread.start()
        self.after(80, self.receive_worker_message)

    def _identify_worker(self, catalogue_folder, paths):
        try:
            catalogue = identify.load_catalogue(catalogue_folder)
            if not catalogue:
                raise ValueError('The chosen catalogue contains no readable image files.')
            results = [identify.identify(path, catalogue, save_temp=True, method='notches') for path in paths]
            self.worker_messages.put(('complete', results))
        except Exception as error:
            self.worker_messages.put(('error', str(error)))

    def receive_worker_message(self):
        try:
            message, payload = self.worker_messages.get_nowait()
        except queue.Empty:
            self.after(80, self.receive_worker_message)
            return

        self.identify_button.configure(state=tk.NORMAL)
        if message == 'error':
            self.status.set('Identification could not be completed.')
            messagebox.showerror('Identification error', payload)
            return

        self.results = payload
        for index, result in enumerate(self.results):
            identify.save_for_human_review(result)
            row = identify.result_row(result)
            candidate = row['candidate_id'] or 'No confident match'
            self.result_table.insert('', tk.END, iid=str(index), tags=(row['decision'],), values=(
                os.path.basename(result['path']), candidate,
                row['best_score_percent'] + '%', row['decision'].replace('_', ' ')))
        self.export_button.configure(state=tk.NORMAL)
        self.status.set('Finished %d sighting%s. Select a result to inspect it.' %
                        (len(self.results), '' if len(self.results) == 1 else 's'))
        if self.results:
            self.result_table.selection_set('0')
            self.show_selected_result()

    def preview_image(self, image, title, max_width=410, max_height=290):
        """Turn an OpenCV image into a Tk-compatible PNG without Pillow."""
        height, width = image.shape[:2]
        scale = min(max_width / width, max_height / height, 1.0)
        resized = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))),
                             interpolation=cv2.INTER_AREA)
        title_bar = 28
        canvas = cv2.copyMakeBorder(resized, title_bar, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
        cv2.putText(canvas, title, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        ok, encoded = cv2.imencode('.png', canvas)
        if not ok:
            return None
        return tk.PhotoImage(data=base64.b64encode(encoded.tobytes()).decode('ascii'))

    def show_selected_result(self, _event=None):
        selected = self.result_table.selection()
        if not selected:
            return
        result = self.results[int(selected[0])]
        if not result['ranked']:
            return

        name, score, match = result['ranked'][0]
        left = result['image']
        right, match_description = identify.match_view(result, name)
        if self.show_outline.get():
            if result['description'] is not None:
                left = fin_curvature.draw_outline(left, result['description'])
            if match_description is not None:
                right = fin_curvature.draw_outline(right, match_description)

        other_side = result['mirrored'].get(name)
        match_title = 'Closest match: %s (%.1f%%)%s%s' % (
            name, score * 100, ' - other side' if other_side else '',
            ' - mirrored copy' if result.get('flipped', {}).get(name) else '')
        self.preview_images = [self.preview_image(left, 'Sighting - trailing edge traced from the tip'),
                               self.preview_image(right, match_title)]
        self.sighting_preview.configure(image=self.preview_images[0])
        self.match_preview.configure(image=self.preview_images[1])

        # Notch-pattern graphs under the photos.
        graphs = []
        for description, title in ((result['description'], 'Sighting notch pattern'),
                                   (match_description, 'Match notch pattern')):
            if description is None:
                graphs.append(None)
                continue
            graph = fin_curvature.draw_curvature(description, width=410, height=110)
            graphs.append(self.preview_image(graph, title, max_width=410, max_height=140))
        self.preview_images.extend(graphs)
        self.sighting_graph.configure(image=graphs[0] or '')
        self.match_graph.configure(image=graphs[1] or '')

        # Every dolphin's score, best first.
        self.ranking_table.delete(*self.ranking_table.get_children())
        for dolphin, dolphin_score, _entry in result['ranked']:
            side = 'other side' if result['mirrored'].get(dolphin) else 'same side'
            self.ranking_table.insert('', tk.END, values=(dolphin, '%.1f%%' % (dolphin_score * 100), side))

        self.compare_button.configure(state=tk.NORMAL)

    def open_dataset_viewer(self):
        DatasetViewer(self)

    def comparison_image(self, path):
        """Load one photo with the isolated fin outline drawn over it."""
        image = cv2.imread(path)
        if image is None:
            raise ValueError("Could not open '%s'." % path)
        description = fin_curvature.describe_fin(image)
        if description is not None and self.show_outline.get():
            return fin_curvature.draw_outline(image, description), description
        return image, description

    def open_comparison(self, left_path, right_path):
        """Show two source photos and the fin-only score used for review."""
        try:
            left, left_description = self.comparison_image(left_path)
            right, right_description = self.comparison_image(right_path)
        except ValueError as error:
            messagebox.showerror('Compare photos', str(error))
            return

        window = tk.Toplevel(self)
        window.title('Fin Comparison')
        window.geometry('1060x620')
        window.minsize(760, 480)
        window.columnconfigure(0, weight=1)
        window.columnconfigure(1, weight=1)
        window.rowconfigure(1, weight=1)

        if left_description is not None and right_description is not None:
            score, _ = fin_curvature.curvature_similarity(
                left_description['profile'], right_description['profile'])
            mirrored = fin_curvature.opposite_sides(left_description, right_description)
            message = 'Fin-only notch similarity: %.1f%%%s' % (
                score * 100, ' (opposite side)' if mirrored else '')
        else:
            message = 'One or both fins could not be isolated for comparison.'
        ttk.Label(window, text=message, anchor='center').grid(row=0, column=0, columnspan=2, sticky='ew', pady=(12, 4))

        left_photo = self.preview_image(left, os.path.basename(left_path), 500, 500)
        right_photo = self.preview_image(right, os.path.basename(right_path), 500, 500)
        left_label = ttk.Label(window, image=left_photo, anchor='center')
        right_label = ttk.Label(window, image=right_photo, anchor='center')
        left_label.grid(row=1, column=0, sticky='nsew', padx=(12, 6), pady=(4, 12))
        right_label.grid(row=1, column=1, sticky='nsew', padx=(6, 12), pady=(4, 12))
        # Keep Tk image data alive for the lifetime of the comparison window.
        window.preview_images = (left_photo, right_photo)

    def compare_selected_result(self):
        selected = self.result_table.selection()
        if not selected:
            messagebox.showinfo('Compare photos', 'Select an identification result first.')
            return
        result = self.results[int(selected[0])]
        if not result['ranked']:
            return
        self.open_comparison(result['path'], result['ranked'][0][2]['path'])

    def export_csv(self):
        path = filedialog.asksaveasfilename(
            initialdir=PROGRAM_FOLDER, defaultextension='.csv',
            filetypes=[('CSV file', '*.csv')], initialfile='dolphin_match_review.csv')
        if not path:
            return
        rows = [identify.result_row(result) for result in self.results]
        with open(path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self.status.set('Review CSV saved: %s' % os.path.basename(path))


if __name__ == '__main__':
    DolphinInterface().mainloop()
