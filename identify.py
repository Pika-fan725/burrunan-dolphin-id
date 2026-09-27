"""
Burrunan Dolphin Identification Program
=======================================

This is Part 1 of the Gippsland Lakes project - the part that goes in the
final display product.

The outline says the identification stage works like this:

    1. Take the sighting image.
    2. Run Prewitt edge detection on it.
    3. Save the result temporarily.
    4. Compare the edge with other images.
    5. The highest similarity is likely the same dolphin.

Burrunan dolphins are identified by the SHAPE OF THE DORSAL FIN SEEN FROM THE
SIDE. Every fin picks up its own pattern of nicks and notches along its
trailing edge over the dolphin's life, and that outline works like a
fingerprint. Because it is the outline that carries the information, edge
detection is exactly the right tool - it throws away the lighting, the water
colour and the weather on the day, and keeps the shape.

This matters for the photos you feed in:

  * The fin must be SIDE ON. A fin photographed from the front or behind shows
    the flat face of the fin instead of its outline, and cannot be identified
    by any method - it is a rejected sighting, not a hard one.
  * Use a clear, side-on view with the fin reasonably visible. The program
    isolates the fin silhouette before comparing, so waves and water texture
    are excluded from the match.
  * A dolphin photographed from its LEFT side gives a mirror image of the same
    fin photographed from its RIGHT. The program handles this by always tracing
    the trailing edge from the tip downwards, which gives the same notch
    sequence from either side. It still reports which side each photo shows.

How to use it
-------------
    catalogue/   put side-on fin photos of dolphins you already know in here.
                 Name them <DolphinName>_<number>.jpg, for example:
                     PPB001_01.jpg, PPB001_02.jpg, PPB002_01.jpg
                 (more than one photo per dolphin gives a better result)

    sightings/   put the new fin photos from the monitoring buoys in here.

Then run:
    python3 identify.py                     -> identifies every new sighting
    python3 identify.py sightings/new.jpg   -> identifies just one photo
    python3 identify.py new.jpg --no-display  -> no pop-up windows
"""

import os
import sys
import glob
import argparse
import csv
import shutil

import cv2
import numpy as np

import fin_curvature


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# The folders live next to this file, so the program works the same no matter
# which directory it is run from.
PROGRAM_FOLDER = os.path.dirname(os.path.abspath(__file__))

CATALOGUE_FOLDER = os.path.join(PROGRAM_FOLDER, 'catalogue')    # dolphins we already know
SIGHTINGS_FOLDER = os.path.join(PROGRAM_FOLDER, 'sightings')    # new photos to identify
TEMP_FOLDER = os.path.join(PROGRAM_FOLDER, 'temp_edges')        # step 3: edge results
REVIEW_FOLDER = os.path.join(PROGRAM_FOLDER, 'human_review')    # needs a person to assess

# Every edge image is squashed to this size before comparing, so that a photo
# taken from far away can still be compared against a close-up one.
COMPARE_SIZE = (256, 256)

# How similar two fins have to be before we are willing to call it a match.
# 1.0 would be a perfect match, 0.0 would be no relationship at all.
#
# 'notches' was calibrated on the showcase set of real NOAA re-sightings
# (11 Sep 2026): 10 catalogued dolphins, each with a second photo taken on a
# different day, plus two sightings of dolphins NOT in the catalogue.
#   * every true match scored 64-85%;
#   * the uncatalogued sightings scored at most 50% against any fin;
#   * the one exception is NOAA0423, whose two photos are seven years apart
#     and whose fin gained a large new notch in between - it scores ~35%
#     and is refused, which is honest.
# 55% sits in the gap. On unfiltered photos (36 NOAA individuals, no quality
# check) the true fin ranked first only 31% of the time, so this threshold is
# only trustworthy for reasonably clear, side-on fin photos. A result near
# the line must be checked by a person.
MATCH_THRESHOLD = {'notches': 0.55, 'shape': 0.55}

# Image types the program will accept.
IMAGE_TYPES = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')


# ---------------------------------------------------------------------------
# Step 2: Prewitt edge detection
# ---------------------------------------------------------------------------

def prewitt_edges(image):
    """Run Prewitt edge detection on a colour image and return the edge map.

    This converts to grayscale, slides two masks over the image (one that finds
    vertical edges and one that finds
    horizontal edges), then combine them with Pythagoras' theorem to get the
    overall edge strength at every pixel.
    """

    # Convert the image to grayscale - edges are about brightness changes, so
    # the colour information is not needed.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert to float64 so the maths below does not get rounded off.
    k1 = gray.astype(np.float64)

    # The two Prewitt masks. The first one finds edges running down the image
    # (changes left to right), the second finds edges running across it.
    p_msk_x = np.array([[-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]], dtype=np.float64)

    p_msk_y = np.array([[-1, -1, -1],
                        [0,  0,  0],
                        [1,  1,  1]], dtype=np.float64)

    # Slide each mask over the image. cv2.filter2D actually does correlation,
    # so the mask is flipped first to make it a true convolution.
    kx = cv2.filter2D(k1, -1, cv2.flip(p_msk_x, -1), borderType=cv2.BORDER_CONSTANT)
    ky = cv2.filter2D(k1, -1, cv2.flip(p_msk_y, -1), borderType=cv2.BORDER_CONSTANT)

    # Combine the two directions into one edge strength per pixel.
    edges = np.sqrt(kx ** 2 + ky ** 2)

    return edges


def edges_to_image(edges):
    """Turn a float edge map into a normal 8-bit image that can be saved/shown."""
    # Scale the strongest edge to 255 so the picture is easy to see.
    return cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# ---------------------------------------------------------------------------
# Step 4: comparing two edge images
# ---------------------------------------------------------------------------

def prepare_for_comparison(edges):
    """Get an edge map ready to be compared against another one.

    Two things have to happen first:
      - resize, so every fin is measured at the same scale
      - blur slightly, so a fin that is a few pixels off still lines up
        instead of scoring zero
    """
    resized = cv2.resize(edges, COMPARE_SIZE, interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    return blurred.astype(np.float64)


def prepare_fin_shape(description):
    """Prepare edges from the isolated fin silhouette, never the water photo."""
    silhouette = fin_curvature.silhouette_edges(description)
    if silhouette is None:
        return None
    return prepare_for_comparison(silhouette)


def fin_edges(description):
    """Run Prewitt detection on the binary fin mask, never the photo itself."""
    mask_as_colour = cv2.cvtColor(description['mask'], cv2.COLOR_GRAY2BGR)
    return prewitt_edges(mask_as_colour)


def similarity(edges_a, edges_b):
    """Return how alike two prepared edge maps are, from -1.0 to 1.0.

    This uses normalised cross-correlation. In plain terms: it lines the two
    edge images up pixel by pixel and asks "when one image is bright here, is
    the other one bright here too?". Because both images are normalised first,
    one photo simply being brighter or sharper than the other does not change
    the score - only the shape does.

        1.0  = the two edge patterns are identical
        0.0  = the two edge patterns have nothing in common
    """
    a = edges_a - edges_a.mean()
    b = edges_b - edges_b.mean()

    bottom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    if bottom == 0:
        # One of the images is completely blank, so there is nothing to match.
        return 0.0

    return float((a * b).sum() / bottom)


# ---------------------------------------------------------------------------
# Loading images and the catalogue
# ---------------------------------------------------------------------------

def list_images(folder):
    """Return a sorted list of every image file inside a folder."""
    files = []
    for pattern in IMAGE_TYPES:
        files.extend(glob.glob(os.path.join(folder, pattern)))
        files.extend(glob.glob(os.path.join(folder, pattern.upper())))
    return sorted(set(files))


def dolphin_name(path):
    """Work out which dolphin a catalogue photo belongs to, from its filename.

    'catalogue/Ripple_02.jpg'                ->  'Ripple'
    'catalogue/Notch.png'                    ->  'Notch'
    'catalogue/0472_2007_Jun_27_20D0067.JPG' ->  '0472'   (NOAA naming)
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    if '_' in stem:
        stem = stem.split('_', 1)[0]
    return stem.replace('-', ' ').strip()


def load_catalogue(folder=CATALOGUE_FOLDER):
    """Run edge detection over every known dolphin photo, ready for comparing.

    Returns a list of dictionaries, one per catalogue photo.
    """
    catalogue = []

    for path in list_images(folder):
        image = cv2.imread(path)
        if image is None:
            print("  ! Skipping '%s' - it could not be opened." % path)
            continue

        # Trace the fin outline and measure its curvature. This is what the
        # notch matching runs on. An unusable photo is not compared: letting a
        # fallback use the original picture would match water and waves.
        description = fin_curvature.describe_fin(image)
        if description is None:
            print("  ! Skipping '%s' - its fin could not be isolated from water."
                  % os.path.basename(path))
            continue

        edges = fin_edges(description)
        prepared = prepare_fin_shape(description)

        # Also trace the photo mirrored left-to-right. The trailing-edge trace
        # should come out the same either way, but the fin-finding step is not
        # perfectly even-handed, so the mirrored copy is a second chance for a
        # fin that traced poorly the first time. A sighting is scored against
        # both and the better one counts.
        description_flipped = fin_curvature.describe_fin(cv2.flip(image, 1))

        catalogue.append({
            'name': dolphin_name(path),
            'path': path,
            'image': image,
            'edges': edges,
            'prepared': prepared,
            # The same fin seen from the dolphin's other side, so a sighting
            # taken from the opposite side can still be matched.
            'prepared_mirrored': cv2.flip(prepared, 1),
            'description': description,
            'description_flipped': description_flipped,
        })

    return catalogue


# ---------------------------------------------------------------------------
# The identification itself (steps 1 to 5)
# ---------------------------------------------------------------------------

def identify(sighting_path, catalogue, save_temp=True, method='notches', use_flipped=True):
    """Identify one sighting photo against the catalogue.

    `method` picks how two fins are compared:
        'notches' - trace the fin outline and compare its curvature pattern.
                    This is the one that actually keys on nicks and notches,
                    and it is the default.
        'shape'   - the original whole-picture edge comparison. Kept so the two
                    can be shown side by side, because it is a good
                    demonstration of why the notch method was needed.

    `use_flipped` also scores each catalogue fin's mirrored trace and keeps
    the better result. It is switched off by the self-test, which needs to
    know whether a mirrored photo can be matched to the ORIGINAL trace.

    Returns a dictionary holding the sighting, its edge map, and every
    catalogue dolphin ranked from most similar to least similar.
    """

    # --- Step 1: take the sighting image -----------------------------------
    sighting = cv2.imread(sighting_path)
    if sighting is None:
        raise FileNotFoundError("Could not open the sighting image '%s'." % sighting_path)

    # --- Step 2: run the edge detection program ----------------------------
    # Trace the sighting's fin outline, ready for water-free comparison.
    description = fin_curvature.describe_fin(sighting)
    if description is None:
        raise ValueError("Could not isolate a dorsal fin from this sighting. "
                         "Use a clear side-on fin photo.")

    sighting_edges = fin_edges(description)

    # --- Step 3: save the result temporarily -------------------------------
    temp_path = None
    if save_temp:
        os.makedirs(TEMP_FOLDER, exist_ok=True)
        name = os.path.splitext(os.path.basename(sighting_path))[0]
        temp_path = os.path.join(TEMP_FOLDER, name + '_edges.png')
        cv2.imwrite(temp_path, edges_to_image(sighting_edges))

    # --- Step 4: compare the isolated fin with the other fins --------------
    prepared = prepare_fin_shape(description)

    scores = {}   # best score for each dolphin
    best_photo = {}
    was_mirrored = {}
    used_flipped = {}
    for entry in catalogue:
        use_notches = method == 'notches'
        flipped = False

        if use_notches:
            # Compare the two notch patterns. Both are traced from the tip,
            # so a photo from the dolphin's other side needs no special
            # handling - but it is still worth reporting.
            score, _ = fin_curvature.curvature_similarity(
                description['profile'], entry['description']['profile'])
            mirrored = fin_curvature.opposite_sides(description, entry['description'])

            # ...and against the trace of the mirrored catalogue photo.
            if use_flipped and entry['description_flipped'] is not None:
                flipped_score, _ = fin_curvature.curvature_similarity(
                    description['profile'], entry['description_flipped']['profile'])
                if flipped_score > score:
                    score, flipped = flipped_score, True
        else:
            # Compare isolated fin silhouettes only. Also compare against the
            # mirror, since the sighting may be of the other side.
            same_side = similarity(prepared, entry['prepared'])
            other_side = similarity(prepared, entry['prepared_mirrored'])
            score = max(same_side, other_side)
            mirrored = other_side > same_side

        # A dolphin can have several photos in the catalogue. Keep its best one,
        # because only one of the angles needs to match the sighting.
        if score > scores.get(entry['name'], -2.0):
            scores[entry['name']] = score
            best_photo[entry['name']] = entry
            was_mirrored[entry['name']] = mirrored
            used_flipped[entry['name']] = flipped

    # --- Step 5: highest similarity is likely the same dolphin -------------
    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)

    return {
        'path': sighting_path,
        'image': sighting,
        'edges': sighting_edges,
        'temp_path': temp_path,
        'ranked': [(name, score, best_photo[name]) for name, score in ranked],
        'mirrored': was_mirrored,
        'flipped': used_flipped,
        'description': description,
        'method': method,
    }


def match_view(result, name):
    """The catalogue photo and trace that a match was actually scored on.

    If the mirrored copy of the catalogue photo gave the better score, that
    is what should be shown, so the drawn outline lines up with the picture.
    """
    entry = next(e for n, _s, e in result['ranked'] if n == name)
    if result.get('flipped', {}).get(name) and entry.get('description_flipped') is not None:
        return cv2.flip(entry['image'], 1), entry['description_flipped']
    return entry['image'], entry['description']


# ---------------------------------------------------------------------------
# Showing and reporting the result
# ---------------------------------------------------------------------------

def print_result(result):
    """Print the identification result to the screen."""
    print("\nSighting: %s" % result['path'])
    if result['temp_path']:
        print("  Edge result saved to: %s" % result['temp_path'])

    ranked = result['ranked']
    if not ranked:
        print("  No dolphins in the catalogue to compare against.")
        return

    best_name, best_score, _ = ranked[0]

    print("  Similarity to each known dolphin:")
    for name, score, _ in ranked:
        bar = '#' * int(max(score, 0) * 30)
        print("    %-18s %6.1f%%  %s" % (name, score * 100, bar))

    threshold = MATCH_THRESHOLD[result['method']]

    if best_score >= threshold:
        side = " - seen from the dolphin's other side" if result['mirrored'].get(best_name) else ""
        print("  ==> Most likely: %s  (%.1f%% similar%s)" % (best_name, best_score * 100, side))
        if result.get('flipped', {}).get(best_name):
            print("      (best score came from the mirrored copy of the catalogue photo)")

        # If second place is nearly as close, the result is not trustworthy.
        if len(ranked) > 1 and best_score - ranked[1][1] < 0.10:
            print("      Note: %s scored almost the same, so this one should be"
                  % ranked[1][0])
            print("      checked by hand before it is saved to the database.")
    else:
        print("  ==> No confident match (best was %s at %.1f%%)."
              % (best_name, best_score * 100))
        print("      This is possibly a dolphin that has not been catalogued yet.")

    if result.get('review_path'):
        print("      Saved for human review: %s" % result['review_path'])


def result_row(result):
    """Return one database-ready record for a sighting result.

    This is deliberately just a hand-off record, rather than a replacement for
    a catalogue database. Ambiguous and rejected matches still need a person to
    inspect the photos before a sighting is attached to an individual.
    """
    ranked = result['ranked']
    if not ranked:
        return None

    best_name, best_score, _ = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else None
    threshold = MATCH_THRESHOLD[result['method']]
    ambiguous = runner_up is not None and best_score - runner_up < 0.10

    if best_score < threshold:
        decision = 'new_or_unusable'
        candidate = ''
    elif ambiguous:
        decision = 'human_review'
        candidate = best_name
    else:
        decision = 'candidate_match'
        candidate = best_name

    return {
        'sighting_file': os.path.abspath(result['path']),
        'candidate_id': candidate,
        'best_score_percent': '%.1f' % (best_score * 100),
        'runner_up_score_percent': '' if runner_up is None else '%.1f' % (runner_up * 100),
        'method': result['method'],
        'mirrored': 'yes' if result['mirrored'].get(best_name) else 'no',
        'matched_flipped_photo': 'yes' if result.get('flipped', {}).get(best_name) else 'no',
        'decision': decision,
        'review_file': result.get('review_path', ''),
    }


def save_for_human_review(result):
    """Copy rejected or ambiguous sightings into the separate review queue.

    The catalogue is only for confirmed identities. A program result below the
    confidence threshold, or too close to its runner-up, is copied here for a
    human to inspect instead of being silently saved as a known dolphin.
    """
    row = result_row(result)
    if row is None or row['decision'] == 'candidate_match':
        return None

    os.makedirs(REVIEW_FOLDER, exist_ok=True)
    source = os.path.abspath(result['path'])
    destination = os.path.join(REVIEW_FOLDER, os.path.basename(source))
    if not os.path.exists(destination):
        shutil.copy2(source, destination)
    result['review_path'] = destination
    return destination


def append_results_csv(path, results):
    """Append identification results for later human/database review."""
    rows = [result_row(result) for result in results]
    rows = [row for row in rows if row is not None]
    if not rows:
        return

    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def label(image, text):
    """Put a title bar above an image so the display windows are readable."""
    image = image.copy()
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    bar = np.zeros((28, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return np.vstack([bar, image])


def show_result(result, width=300):
    """Show the sighting and its best match side by side."""
    ranked = result['ranked']
    if not ranked:
        return

    best_name, best_score, best_entry = ranked[0]

    def fit(image):
        """Scale an image so every panel is the same width."""
        if len(image.shape) == 2 or image.dtype != np.uint8:
            image = edges_to_image(image.astype(np.float64))
        scale = width / image.shape[1]
        return cv2.resize(image, (width, int(image.shape[0] * scale)))

    # Say so on screen when the match was to the dolphin's other side.
    mirrored = result['mirrored'].get(best_name)
    match_title = 'Best match: %s%s' % (best_name, ' (other side)' if mirrored else '')

    sighting_desc = result['description']
    match_image, match_desc = match_view(result, best_name)

    if result['method'] == 'notches' and sighting_desc and match_desc:
        # Show the traced outlines and the curvature graphs, because those are
        # what the decision was actually made on. The graphs are the useful bit:
        # a notch in the fin shows up as a dip in the line.
        panels = [
            label(fit(fin_curvature.draw_outline(result['image'], sighting_desc)),
                  'Sighting - fin traced'),
            label(fin_curvature.draw_curvature(sighting_desc, width),
                  'Its notch pattern'),
            label(fit(fin_curvature.draw_outline(match_image, match_desc)),
                  match_title),
            label(fin_curvature.draw_curvature(match_desc, width),
                  '%.1f%% similar' % (best_score * 100)),
        ]
    else:
        panels = [
            label(fit(result['image']), 'Sighting'),
            label(fit(result['edges']), 'Sighting edges'),
            label(fit(best_entry['image']), match_title),
            label(fit(best_entry['edges']), '%.1f%% similar' % (best_score * 100)),
        ]

    # Pad the panels to the same height so they can sit next to each other.
    tallest = max(panel.shape[0] for panel in panels)
    padded = []
    for panel in panels:
        gap = tallest - panel.shape[0]
        padded.append(cv2.copyMakeBorder(panel, 0, gap, 0, 4, cv2.BORDER_CONSTANT, value=(40, 40, 40)))

    cv2.imshow('Dolphin Identification - %s' % os.path.basename(result['path']),
               np.hstack(padded))


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Identify Burrunan dolphins from their dorsal fin edges.')
    parser.add_argument('sighting', nargs='*',
                        help='sighting image(s) to identify '
                             '(default: everything in the sightings folder)')
    parser.add_argument('--catalogue', default=CATALOGUE_FOLDER,
                        help='folder of known dolphin photos')
    parser.add_argument('--no-display', action='store_true',
                        help='do not open the comparison windows')
    parser.add_argument('--method', choices=['notches', 'shape'], default='notches',
                        help="'notches' compares the fin outline's curvature "
                             "(default); 'shape' is the older whole-picture "
                             "comparison, kept for comparison")
    parser.add_argument('--results-csv',
                        help='append candidate matches to this CSV for human/database review')
    parser.add_argument('--no-review', action='store_true',
                        help='do not copy rejected or ambiguous sightings to human_review/')
    parser.add_argument('--self-test', action='store_true',
                        help='feed every catalogue photo back in mirrored and check '
                             'it identifies itself against the un-mirrored traces')
    args = parser.parse_args()

    # Make sure the folders the program expects actually exist.
    for folder in (args.catalogue, SIGHTINGS_FOLDER):
        os.makedirs(folder, exist_ok=True)

    # Load the known dolphins and run edge detection over all of them.
    print("Loading the catalogue from '%s'..." % args.catalogue)
    catalogue = load_catalogue(args.catalogue)
    if not catalogue:
        print("The catalogue is empty. Put some known dolphin photos in '%s',"
              % args.catalogue)
        print("named like Ripple_01.jpg, and run the program again.")
        return 1
    print("Loaded %d photo(s) of %d dolphin(s)."
          % (len(catalogue), len({e['name'] for e in catalogue})))

    if args.self_test:
        return self_test(catalogue)

    # Work out which sightings to identify.
    sightings = args.sighting or list_images(SIGHTINGS_FOLDER)
    if not sightings:
        print("No sightings to identify. Put the new photos in '%s'."
              % SIGHTINGS_FOLDER)
        return 1

    results = []
    for path in sightings:
        try:
            result = identify(path, catalogue, method=args.method)
        except (FileNotFoundError, ValueError) as error:
            print("\n! %s" % error)
            continue

        if not args.no_review:
            save_for_human_review(result)
        print_result(result)
        results.append(result)

        if not args.no_display:
            show_result(result)

    if args.results_csv:
        append_results_csv(args.results_csv, results)
        print("\nResults saved for review: %s" % os.path.abspath(args.results_csv))

    # Hold the windows open until a key is pressed.
    if not args.no_display:
        print("\nPress any key on an image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


def self_test(catalogue):
    """Check the mirror handling: does each fin still match itself flipped?

    Every catalogue photo is mirrored left-to-right and identified against
    the catalogue's ORIGINAL traces only. Because the trailing edge is always
    traced from the tip, a mirrored photo should produce the same notch
    pattern and match itself at or near 100%. A fin that fails here is one
    the program traces differently depending on which way it faces - which
    is worth knowing before trusting it on a real other-side sighting.
    """
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    print("\nSelf-test: each catalogue photo mirrored, matched against the originals")
    passed = 0
    for entry in catalogue:
        flipped_path = os.path.join(TEMP_FOLDER, 'selftest_' + os.path.basename(entry['path']))
        cv2.imwrite(flipped_path, cv2.flip(entry['image'], 1))
        try:
            result = identify(flipped_path, catalogue, save_temp=False, use_flipped=False)
        except ValueError:
            print("  %-12s FAIL - fin could not be traced in the mirrored photo" % entry['name'])
            continue
        top_name, top_score, _ = result['ranked'][0]
        own_score = dict((n, s) for n, s, _e in result['ranked']).get(entry['name'], 0.0)
        ok = top_name == entry['name']
        passed += ok
        print("  %-12s %s  scored %5.1f%% against itself, ranked %s"
              % (entry['name'], 'ok  ' if ok else 'FAIL', own_score * 100,
                 '1st' if ok else 'behind %s (%.1f%%)' % (top_name, top_score * 100)))
    print("\n%d of %d fins recognised themselves mirrored." % (passed, len(catalogue)))
    return 0 if passed == len(catalogue) else 1


if __name__ == '__main__':
    sys.exit(main())
