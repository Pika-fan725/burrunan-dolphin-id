"""
Fin outline and curvature matching
==================================

Burrunan dolphins are told apart by the nicks and notches cut into the
TRAILING EDGE of their dorsal fin over their life. This module turns that edge
into a pattern that can be compared between photographs.

The first version of the program compared whole edge-detected pictures. When
that was measured it turned out to be almost blind to notches: two large
notches cut into a fin only dropped the score from 100% to 93%, while a totally
different dolphin dropped it to 41-58%. It was really measuring the broad
triangle of the fin, which most dolphins share. This module does what the
professional systems (CurvRank, finFindR) do instead:

    1. Separate the fin from the water.
    2. Find the tip of the fin and trace DOWN THE TRAILING EDGE to the base.
    3. Walk along that edge and measure the CURVATURE at every step.
    4. Compare two fins by lining up their curvature patterns.

Curvature is measured the "integral" way: put a small circle on a point of the
outline and ask what fraction of that circle is inside the fin.

    a straight edge     ->  about 0.5, half the circle is inside
    a notch cut in      ->  less than 0.5, the circle is mostly outside
    a bump sticking out ->  more than 0.5

So a notch becomes a dip in a graph, and identifying a dolphin turns into
comparing two wiggly lines. This is measured at several circle sizes at once,
so both tiny nicks and large chunks out of the fin are picked up.

Why only the trailing edge, and why tip-first
---------------------------------------------
The leading edge is smooth on nearly every dolphin and carries almost no
identity. The trailing edge is where the notches are. Tracing it from the tip
every time also solves the "other side" problem for free: a fin photographed
from the dolphin's left is a mirror image of the same fin from its right, but
the walk from tip to base along the trailing edge produces the same sequence
of notches either way. Which side the dolphin was seen from is still recorded,
because it is useful to know, but the comparison no longer depends on it.
"""

import cv2
import numpy as np


# How many points to sample along the traced edge. More points means finer
# detail but slower comparing.
OUTLINE_POINTS = 200

# The circle sizes used to measure curvature, as a fraction of the fin's height.
# Small circles catch small nicks, large circles catch big missing chunks.
CURVATURE_SCALES = (0.04, 0.08, 0.16)

# Photos are shrunk to this many pixels on their longest side while the fin is
# being separated from the water. It makes that step fast and means a 4000px
# photo and a 600px photo are treated the same way.
WORK_SIZE = 500

# The comparison distance at which two fins are treated as having nothing in
# common. See curvature_similarity() for how this number was measured.
ZERO_AT = 0.06


# ---------------------------------------------------------------------------
# Step 1: separating the fin from the water
# ---------------------------------------------------------------------------

def fin_mask(image):
    """Return a black-and-white mask with the dolphin in white, water in black.

    No single rule separates a fin from water in every photo. A fin can be
    darker than the water or paler than it, the water can be blue, green or
    grey, and wave shadows can be darker than everything. So several ways of
    splitting the picture are tried, every blob each one produces is scored on
    how much it looks like a fin, and the best-scoring blob wins. That blob's
    boundary is then tidied up with GrabCut, which uses the colours on each side
    of the boundary to decide exactly where it runs.

    The mask is returned at the photo's original size, upright, untouched by
    any rotation - so it lines up with the photo pixel for pixel.
    """
    height, width = image.shape[:2]

    # Work on a shrunken copy: quicker, and photo size stops mattering.
    scale = min(1.0, WORK_SIZE / float(max(height, width)))
    small = cv2.resize(image, (max(1, int(round(width * scale))),
                               max(1, int(round(height * scale)))),
                       interpolation=cv2.INTER_AREA) if scale < 1.0 else image.copy()

    # Every plausible blob from every way of splitting the photo, with its
    # fin-likeness score and which split it came from.
    found = []
    for split_index, candidate in enumerate(threshold_candidates(small)):
        for blob in blobs(candidate):
            score = fin_likeness(blob)
            if np.isfinite(score):
                found.append((split_index, blob, score))
    if not found:
        return None

    # Consensus bonus. A single split can be fooled - a glare streak down the
    # face of a fin can make brightness alone cut the fin in half, and the
    # half still looks fairly fin-shaped. But two different ways of looking
    # at the photo rarely make the same mistake, so a blob that another split
    # also found is much more likely to be the real dolphin.
    best_blob = None
    best_score = -np.inf
    for split_index, blob, score in found:
        agreement = max([overlap(blob, other) for other_split, other, _ in found
                         if other_split != split_index] or [0.0])
        total = score + 2.0 * agreement
        if total > best_score:
            best_score, best_blob = total, blob

    # Tidy the boundary. GrabCut can occasionally make things worse (for
    # example by swallowing a patch of water), so its result is only kept if it
    # still looks at least as fin-like as what went in.
    refined = grabcut_refine(small, best_blob)
    if refined is not None and fin_likeness(refined) >= fin_likeness(best_blob) - 0.25:
        best_blob = refined

    # A fin has no holes in it, so any hole is a glare spot or a shadow that
    # got mislabelled. Fill them by redrawing just the outer boundary.
    best_blob = fill_holes(best_blob)

    # Back to the photo's real size.
    if scale < 1.0:
        full = cv2.resize(best_blob, (width, height), interpolation=cv2.INTER_LINEAR)
        return np.uint8(full > 127) * 255
    return best_blob


def fill_holes(mask):
    """Fill any holes inside the largest blob, keeping its outer edge as is."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return mask
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    return filled


def threshold_candidates(small):
    """Yield several rough fin/water splits of the same photo.

    Each split is a plain Otsu threshold on one channel, tried both ways round
    because the fin might be the dark part or the light part:

        grey brightness   - the obvious one; dark fin against bright water
        blue-vs-yellow    - water is bluish, a dolphin is grey-brown, so this
                            separates them even when they are equally bright
        colour saturation - water is often more colourful than the dolphin
    """
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    lab_b = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[:, :, 2], (5, 5), 0)
    sat = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1], (5, 5), 0)

    kernel = np.ones((5, 5), np.uint8)
    for channel in (gray, lab_b, sat):
        for flag in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
            _, mask = cv2.threshold(channel, 0, 255, flag + cv2.THRESH_OTSU)
            # Close small gaps and remove speckles.
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            yield mask


def overlap(blob_a, blob_b):
    """How much two blobs coincide: shared area over combined area, 0 to 1."""
    a, b = blob_a > 0, blob_b > 0
    union = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / float(union) if union else 0.0


def blobs(mask):
    """Yield each separate white blob of a mask that is a plausible size."""
    height, width = mask.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA] / float(height * width)
        # A close-up can have the dolphin filling nearly the whole frame, so
        # the upper limit is generous. Blobs that big are still judged on
        # shape by fin_likeness().
        if 0.03 < area < 0.95:
            yield np.uint8(labels == label) * 255


def fin_likeness(blob):
    """Score how much a blob looks like a dorsal fin. Higher is more fin-like.

    Returns -inf for blobs that cannot be a fin. The clues used:

        pointed   a fin comes to a tip: its top few rows are much narrower
                  than its widest row. A patch of water is not pointed.
        big       the fin is the subject of the photo, so it takes up a fair
                  share of the frame. Scraps of foam do not.
        tall      likewise, it spans a good fraction of the picture's height.
        solid     a fin is one solid lump. Wave shadows are stringy streaks.
                  (A fin on a back scores modestly here - the hull spans the
                  gap between tip and back - so this is a nudge, not a veto.)
        grounded  the rest of the dolphin is under the water, so a real fin
                  often reaches the bottom of the picture.
        not sky   a blob touching the top of the picture is probably water
                  or sky that got split the wrong way round.
    """
    height, width = blob.shape
    rows_filled = np.count_nonzero(blob, axis=1)
    filled_rows = np.where(rows_filled > 0)[0]
    if len(filled_rows) < 10:
        return -np.inf

    top, bottom = filled_rows[0], filled_rows[-1]
    blob_height = bottom - top + 1
    area_px = float(rows_filled.sum())
    area = area_px / (height * width)

    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return -np.inf
    hull_area = cv2.contourArea(cv2.convexHull(max(contours, key=cv2.contourArea)))
    solidity = area_px / hull_area if hull_area > 0 else 0.0
    if solidity < 0.35:
        return -np.inf

    # How narrow the top 15% of the blob is, compared with its widest row.
    band = rows_filled[top: top + max(3, int(blob_height * 0.15))]
    pointed = 1.0 - band.max() / float(rows_filled.max())

    score = 2.0 * pointed
    score += 2.0 * min(1.0, area / 0.25)
    score += 1.0 * min(1.0, blob_height / (0.4 * height))
    score += solidity
    score += 0.3 if bottom >= height - 2 else 0.0
    score -= 1.5 if top <= 1 else 0.0
    score -= 2.0 * max(0.0, area - 0.6)
    return score


def grabcut_refine(small, blob):
    """Let GrabCut redraw the blob's boundary using the photo's colours.

    The blob says roughly where the dolphin is. GrabCut builds a colour model
    of "dolphin" from inside it and "water" from around it, then decides pixel
    by pixel near the boundary which model fits better. This is what recovers a
    pale fin edge that a plain brightness threshold chopped off.
    """
    height, width = blob.shape
    reach = max(5, int(round(min(height, width) * 0.04)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach, reach))

    sure_fg = cv2.erode(blob, kernel, iterations=2)
    maybe = cv2.dilate(blob, kernel, iterations=2)

    labels = np.full(blob.shape, cv2.GC_BGD, np.uint8)
    labels[maybe > 0] = cv2.GC_PR_BGD
    labels[blob > 0] = cv2.GC_PR_FGD
    labels[sure_fg > 0] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small, labels, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None

    result = np.uint8((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)) * 255
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Keep only the piece that overlaps the original blob.
    count, comp, stats, _ = cv2.connectedComponentsWithStats(result, 8)
    if count < 2:
        return None
    overlap = [np.count_nonzero((comp == i) & (blob > 0)) for i in range(1, count)]
    keep = 1 + int(np.argmax(overlap))
    if overlap[keep - 1] == 0:
        return None

    refined = np.uint8(comp == keep) * 255
    # Refuse a result that changed size dramatically - it has wandered off.
    ratio = np.count_nonzero(refined) / float(max(1, np.count_nonzero(blob)))
    if not 0.6 < ratio < 1.6:
        return None
    return refined


def silhouette_edges(description):
    """Return a water-free edge image of the detected fin silhouette.

    Used only by the older whole-shape comparison (identify.py --method shape).
    The edges come from the binary mask rather than the photograph, so
    reflections, foam and waves cannot affect that score either.
    """
    mask = description['mask']
    points = cv2.findNonZero(mask)
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    padding = max(2, int(round(max(width, height) * 0.03)))
    x1, y1 = max(0, x - padding), max(0, y - padding)
    x2, y2 = min(mask.shape[1], x + width + padding), min(mask.shape[0], y + height + padding)
    cropped = mask[y1:y2, x1:x2]
    return cv2.morphologyEx(cropped, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))


# ---------------------------------------------------------------------------
# Step 2: tracing the trailing edge, from the tip down to the base
# ---------------------------------------------------------------------------

def fin_outline(mask):
    """Trace the trailing edge from the tip of the fin down to its base.

    Returns (points, facing). `points` runs tip-first. `facing` is 'left' or
    'right' - which way the dolphin is heading in the photo, worked out from
    which side of the fin the trailing edge is on.

    The tip is the highest point of the outline. From there the outline can be
    followed two ways: down the leading edge or down the trailing edge. Which
    is which is decided by trailing_direction(). The trace then follows the
    outline until it reaches the edge of the picture or turns back upwards,
    and cut_at_back() trims off any of the dolphin's back that got included.
    """
    contour = largest_contour(mask)
    if contour is None:
        return None
    count = len(contour)
    tip = int(np.argmin(contour[:, 1]))
    mask_height = float(contour[:, 1].max() - contour[:, 1].min())
    if mask_height < 10:
        return None

    step = trailing_direction(mask, contour, tip)
    if step is None:
        return None

    # Walk the outline from the tip in the trailing direction.
    height, width = mask.shape
    indices = [tip]
    lowest = contour[tip, 1]
    for offset in range(1, int(count * 0.6)):
        index = (tip + step * offset) % count
        x, y = contour[index]
        indices.append(index)
        lowest = max(lowest, y)
        if x <= 0 or y <= 0 or x >= width - 1 or y >= height - 1:
            break                                   # reached the picture edge
        if y < lowest - 0.10 * mask_height:
            break                                   # started climbing back up
    edge = contour[indices].astype(np.float64)
    if len(edge) < 20:
        return None

    edge = cut_at_back(edge)
    if edge is None or len(edge) < 20:
        return None
    if edge[:, 1].max() - edge[:, 1].min() < 0.25 * mask_height:
        return None                                 # too short to be a fin edge

    # The trailing edge is on the tail side of the fin, so if it lies on the
    # right of the fin the dolphin is heading left, and vice versa.
    facing = 'left' if side_of_points(mask, edge) == 'right' else 'right'

    # Smooth away the staircase jaggedness that comes from tracing around square
    # pixels. Without this the curvature graph is full of noise that has nothing
    # to do with the dolphin. The window is small so real notches survive.
    edge = smooth(edge, 5)

    points = resample(edge, OUTLINE_POINTS)
    if points is None:
        return None
    return points, facing


def trailing_direction(mask, contour, tip):
    """Which way round the outline from the tip is the trailing edge?

    Returns +1 or -1 (the step to add to the contour index), or None.

    Two clues. A dorsal fin sweeps backwards, so its tip leans towards the
    tail and the trailing edge is on the side it leans to - that is used
    whenever the lean is clear. Otherwise the trailing edge is the side with
    the deeper scoop below the tip, since the leading edge is a smooth convex
    curve and the trailing edge is concave.
    """
    count = len(contour)
    probe = max(5, int(count * 0.05))
    forward = contour[[(tip + i) % count for i in range(1, probe + 1)]]
    backward = contour[[(tip - i) % count for i in range(1, probe + 1)]]
    # Which direction heads right from the tip?
    rightward = +1 if forward[:, 0].mean() > backward[:, 0].mean() else -1

    wanted = lean_of(mask, contour, tip)
    if wanted is None:
        scoops = tip_scoops(contour, tip)
        if not scoops:
            return None
        depth, near, far_end = max(scoops)
        mask_height = float(contour[:, 1].max() - contour[:, 1].min())
        if depth < 0.03 * mask_height:
            return None
        points = contour[ring_range(near, far_end, count)] if near <= far_end \
            else contour[ring_range(far_end, near, count)]
        wanted = side_of_points(mask, points)

    return rightward if wanted == 'right' else -rightward


def largest_contour(mask):
    """The outline of the biggest blob, as an (N, 2) array of x, y points."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    return contour if len(contour) >= 40 else None


def ring_range(a, b, count):
    """Indices going forward from a to b around a closed outline."""
    if a <= b:
        return np.arange(a, b + 1)
    return np.concatenate([np.arange(a, count), np.arange(0, b + 1)])


def tip_scoops(contour, tip):
    """List the concavities of an outline that begin near the tip.

    Each entry is (depth, near_end, far_end): the two ends are indices into
    the contour, near_end being the one close to the tip.
    """
    hull = cv2.convexHull(contour.reshape(-1, 1, 2), returnPoints=False)
    try:
        defects = cv2.convexityDefects(contour.reshape(-1, 1, 2), hull)
    except cv2.error:
        return []
    if defects is None:
        return []

    mask_height = float(contour[:, 1].max() - contour[:, 1].min())
    reach = 0.25 * mask_height
    tip_point = contour[tip].astype(np.float64)

    scoops = []
    for start, end, _far, depth in defects.reshape(-1, 4):
        start, end = int(start), int(end)
        to_start = np.linalg.norm(contour[start] - tip_point)
        to_end = np.linalg.norm(contour[end] - tip_point)
        if min(to_start, to_end) > reach:
            continue
        near, far_end = (start, end) if to_start <= to_end else (end, start)
        scoops.append((depth / 256.0, near, far_end))
    return scoops


def lean_of(mask, contour, tip):
    """Which way does the tip lean - 'left', 'right', or None if it is upright?

    Compares the tip's x position with the centre of the blade just below
    it. Only the top of the fin is looked at: further down the fin widens
    into the back, which would drag the centre towards the tail and give the
    wrong answer. A fin that leans clearly is easy to call; a nearly upright
    one returns None so the caller can fall back on other clues.
    """
    tip_x, tip_y = contour[tip]
    mask_height = contour[:, 1].max() - contour[:, 1].min()
    band = mask[int(tip_y): int(tip_y + 0.15 * mask_height) + 1]
    filled = np.flatnonzero(band.any(axis=0))
    if len(filled) == 0:
        return None
    # Centre and width of the fin in that band of rows.
    rows_x = [np.flatnonzero(row) for row in band]
    centres = [0.5 * (r[0] + r[-1]) for r in rows_x if len(r)]
    widths = [r[-1] - r[0] for r in rows_x if len(r)]
    if not centres:
        return None
    lean = tip_x - float(np.median(centres))
    if abs(lean) < 0.08 * max(widths):
        return None
    return 'right' if lean > 0 else 'left'


def side_of_points(mask, points):
    """Are these outline points on the left or right boundary of the blob?

    For each point, look along its row of the mask and see whether the point
    is nearer the leftmost or the rightmost filled pixel. Majority wins.
    """
    height, width = mask.shape
    votes_right = 0
    votes_left = 0
    for x, y in points:
        row = int(round(y))
        if not 0 <= row < height:
            continue
        filled = np.flatnonzero(mask[row])
        if len(filled) == 0:
            continue
        if abs(x - filled[-1]) < abs(x - filled[0]):
            votes_right += 1
        else:
            votes_left += 1
    return 'right' if votes_right >= votes_left else 'left'


def cut_at_back(edge):
    """Trim the traced edge where it stops being fin and becomes back.

    The trailing edge heads steeply downwards. When it reaches the base it
    turns and follows the dolphin's back, which runs much closer to level and
    stays that way to the end of the trace. So the trace is cut at the start
    of the final level stretch. Looking for the LAST level stretch matters: a
    big notch can have a level floor partway down, and that must not be
    mistaken for the back. If the trace never levels out (a photo cropped
    tightly to the fin) the whole trace is kept.
    """
    if len(edge) < 20:
        return edge

    steps = np.diff(edge, axis=0)
    lengths = np.sqrt((steps ** 2).sum(axis=1))
    travelled = np.concatenate([[0.0], np.cumsum(lengths)])
    total = travelled[-1]
    if total <= 0:
        return edge

    # Direction of travel, averaged over a short window so that a single
    # notch does not look like a turn.
    window = max(5, int(len(steps) * 0.05))
    kernel = np.ones(window) / window
    dx = np.convolve(steps[:, 0], kernel, mode='same')
    dy = np.convolve(steps[:, 1], kernel, mode='same')
    level = (np.abs(dy) < np.abs(dx) * 0.85).astype(np.float64)   # ~40 degrees
    level_share = np.convolve(level, kernel, mode='same')

    # The very end of the trace can run along the edge of the picture, which
    # is not part of the dolphin at all. Skip past that first.
    end = len(steps)
    while end > 0 and level_share[end - 1] <= 0.6:
        end -= 1
    if end == 0:
        return edge                       # never levelled out

    # Then walk back through the level stretch to find where it started.
    cut = end
    while cut > 0 and level_share[cut - 1] > 0.6:
        cut -= 1
    if travelled[cut] < total * 0.35:
        return edge                       # levelled out too early to be the back
    if end - cut < window:
        return edge                       # too short a stretch to trust
    return edge[: cut + 1]


def smooth(points, window):
    """Run a small moving average along a traced line."""
    if len(points) < window * 2:
        return points

    kernel = np.ones(window) / window
    x = np.convolve(points[:, 0], kernel, mode='same')
    y = np.convolve(points[:, 1], kernel, mode='same')

    # The ends get dragged inwards by the averaging, so put them back.
    half = window // 2
    x[:half], x[-half:] = points[:half, 0], points[-half:, 0]
    y[:half], y[-half:] = points[:half, 1], points[-half:, 1]

    return np.stack([x, y], axis=1)


def resample(points, count):
    """Space `count` points evenly along a traced line.

    The tracing gives points that are bunched up in places, which would make the
    curvature graph stretch and squash depending on the photo. Spacing them
    evenly by distance travelled fixes that.
    """
    steps = np.sqrt(((np.diff(points, axis=0)) ** 2).sum(axis=1))
    travelled = np.concatenate([[0], np.cumsum(steps)])
    if travelled[-1] == 0:
        return None

    wanted = np.linspace(0, travelled[-1], count)
    x = np.interp(wanted, travelled, points[:, 0])
    y = np.interp(wanted, travelled, points[:, 1])
    return np.stack([x, y], axis=1)


# ---------------------------------------------------------------------------
# Step 3: measuring the curvature along the edge
# ---------------------------------------------------------------------------

def curvature_profile(mask, outline):
    """Measure how curved the edge is at every point, at several scales.

    Returns a table with one row per outline point and one column per circle
    size. Each number is the fraction of that circle which is inside the fin.
    """
    # The circle sizes are set from the height of the traced edge, so that a
    # close-up photo and a distant photo of the same fin give the same numbers.
    fin_height = outline[:, 1].max() - outline[:, 1].min()
    if fin_height < 10:
        return None

    inside = (mask > 0).astype(np.float64)
    height, width = inside.shape

    profile = np.zeros((len(outline), len(CURVATURE_SCALES)))

    for column, scale in enumerate(CURVATURE_SCALES):
        radius = max(2, int(round(fin_height * scale)))

        # Work out which offsets within the square are actually in the circle.
        yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        disc = (xx ** 2 + yy ** 2) <= radius ** 2
        disc_area = float(disc.sum())

        for row, (px, py) in enumerate(outline):
            cx, cy = int(round(px)), int(round(py))

            x1, x2 = max(0, cx - radius), min(width, cx + radius + 1)
            y1, y2 = max(0, cy - radius), min(height, cy + radius + 1)
            if x2 <= x1 or y2 <= y1:
                continue

            patch = inside[y1:y2, x1:x2]
            # Line the circle stencil up with the patch, in case the point sits
            # near the edge of the picture and the patch got clipped.
            sub = disc[y1 - (cy - radius): y1 - (cy - radius) + (y2 - y1),
                       x1 - (cx - radius): x1 - (cx - radius) + (x2 - x1)]
            profile[row, column] = (patch * sub).sum() / disc_area

    return profile


def describe_fin(image):
    """Do the whole job on one photo: mask, trailing edge, curvature.

    Returns a dictionary with:
        mask     the dolphin in white, water in black, same size as the photo
        outline  the trailing edge, tip first, in photo pixel coordinates
        facing   'left' or 'right' - which way the dolphin is heading
        profile  the curvature pattern along the edge

    or None if the fin could not be found. Returning None matters - a photo the
    program cannot read should be reported as unusable, not quietly given a bad
    score.
    """
    mask = fin_mask(image)
    if mask is None:
        return None

    traced = fin_outline(mask)
    if traced is None:
        return None
    outline, facing = traced

    profile = curvature_profile(mask, outline)
    if profile is None:
        return None

    return {'mask': mask, 'outline': outline, 'facing': facing, 'profile': profile}


def opposite_sides(description_a, description_b):
    """True if the two photos show the dolphin from opposite sides.

    A dolphin heading left shows the camera its right flank, and vice versa,
    so two photos with the dolphin facing different ways were taken from
    different sides.
    """
    return description_a['facing'] != description_b['facing']


# ---------------------------------------------------------------------------
# Step 4: comparing two curvature patterns
# ---------------------------------------------------------------------------

def dtw_distance(profile_a, profile_b, band=0.2, open_end=0.7):
    """Compare two curvature patterns, allowing them to stretch to fit.

    Two photos of the same fin never trace to exactly the same points, and one
    may be a bit larger or show a bit more of the base than the other. So the
    two patterns are allowed to stretch and squash a little to line up - this
    is called dynamic time warping. The answer is the leftover difference once
    the best line-up has been found. Smaller means more alike.

    Both patterns start at the tip, which is a reliable landmark. The base end
    is not: one photo may include a little more of the dolphin's back. So the
    line-up is allowed to finish early on either pattern (`open_end` says how
    much of each must be used at minimum), rather than being forced to stretch
    a short trace over a long one.

    `band` limits how far the line-up may stray, which stops the maths from
    matching the tip of one fin to the base of the other.
    """
    n, m = len(profile_a), len(profile_b)
    width = max(1, int(max(n, m) * band))

    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        low = max(1, i - width)
        high = min(m, i + width)
        for j in range(low, high + 1):
            step = np.linalg.norm(profile_a[i - 1] - profile_b[j - 1])
            cost[i, j] = step + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    # Allow the line-up to finish anywhere in the last stretch of either
    # pattern. Divide by the path length so a short line-up is not favoured
    # just for being short.
    best = np.inf
    for j in range(int(m * open_end), m + 1):
        if np.isfinite(cost[n, j]):
            best = min(best, cost[n, j] / (n + j))
    for i in range(int(n * open_end), n + 1):
        if np.isfinite(cost[i, m]):
            best = min(best, cost[i, m] / (i + m))

    return None if not np.isfinite(best) else float(best)


def curvature_similarity(profile_a, profile_b):
    """Turn the comparison into a 0.0 to 1.0 similarity.

    Returns (similarity, mirrored). Because both patterns are traced from the
    tip down the trailing edge, a fin seen from the dolphin's other side gives
    the same pattern and needs no special handling here - so `mirrored` is
    always False. Use opposite_sides() on the two descriptions to find out
    which side each photo was taken from.
    """
    distance = dtw_distance(profile_a, profile_b)
    if distance is None:
        return 0.0, False

    # Turn the distance into a 0-1 similarity where bigger means more alike.
    # ZERO_AT is the distance treated as "nothing in common". It was measured,
    # not guessed - see the calibration notes in identify.py. If you add a lot
    # more photos, measure the distances again and adjust it.
    return float(max(0.0, 1.0 - distance / ZERO_AT)), False


# ---------------------------------------------------------------------------
# Drawing, for the display
# ---------------------------------------------------------------------------

def draw_outline(image, description):
    """Draw the traced trailing edge over the photo, tip marked with a dot.

    The outline is stored in the photo's own pixel coordinates, so it lines
    up exactly. If a resized copy of the photo is passed in, the outline is
    scaled to suit.
    """
    canvas = image.copy()
    mask_height, mask_width = description['mask'].shape[:2]
    scale_x = canvas.shape[1] / float(mask_width)
    scale_y = canvas.shape[0] / float(mask_height)

    outline = description['outline'] * np.array([scale_x, scale_y])
    thickness = max(1, int(round(min(canvas.shape[:2]) / 250.0)))
    cv2.polylines(canvas, [outline.astype(np.int32)], False, (0, 255, 255), thickness)

    tip = tuple(int(round(v)) for v in outline[0])
    cv2.circle(canvas, tip, thickness * 3, (0, 80, 255), -1)
    return canvas


def draw_curvature(description, width=300, height=120):
    """Draw the curvature pattern as a graph. Notches show up as dips."""
    profile = description['profile']
    canvas = np.zeros((height, width, 3), np.uint8)

    # Curvature values cluster around 0.5, so the graph zooms in on the band
    # they actually occupy. Otherwise the interesting wiggles are a flat line.
    low, high = 0.15, 0.85

    def to_y(value):
        fraction = (np.clip(value, low, high) - low) / (high - low)
        return np.clip(height - 1 - fraction * (height - 1), 0, height - 1)

    # A straight edge sits at 0.5, so mark that as the "no notch" line.
    midline = int(to_y(0.5))
    cv2.line(canvas, (0, midline), (width, midline), (70, 70, 70), 1)

    colours = [(120, 200, 255), (120, 255, 160), (255, 160, 120)]
    for column in range(profile.shape[1]):
        values = profile[:, column]
        xs = np.linspace(0, width - 1, len(values)).astype(np.int32)
        ys = to_y(values).astype(np.int32)
        points = np.stack([xs, ys], axis=1)
        cv2.polylines(canvas, [points], False, colours[column % len(colours)], 1)

    return canvas
